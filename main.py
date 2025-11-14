# train_and_eval_hqcnn.py
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from open_image import open_images_in_path, images_to_angles_batch
from ai.hqcnn import DigitRecognizerHQCNN
from ai.quantum import N_WIRES


def train_and_eval_hqcnn(
        data_path="train_assets",
        labels_json="labels.json",
        num_epochs=100,
        lr=1e-3,
        batch_size=None,
        seed=42,
        save_model_path="hqcnn_digits.pth",
        save_angles_path="image_angles_digits.pkl",
        plot_path="training_curves_digits.png",
        test_labels_json=None
    ):
    """
    Train the HQCNN on a dataset of images and optionally evaluate on test labels.

    Args:
        data_path: Folder with training images.
        labels_json: JSON file with training labels.
        num_epochs: Number of training epochs.
        lr: Learning rate.
        batch_size: Training batch size (auto if None).
        seed: Random seed.
        save_model_path: Path to save trained model weights.
        save_angles_path: Path to save angle representations.
        plot_path: Path to save training curves plot.
        test_labels_json: JSON file with test_labels and optional true_labels.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # ---- Load training labels ----
    with open(os.path.join(data_path, labels_json), "r") as f:
        label_dict = json.load(f)["labels"]

    # ---- Load training images ----
    images = open_images_in_path(data_path)
    image_names = [os.path.basename(getattr(img, 'filename', f"image{i+1}")) for i, img in enumerate(images)]

    # ---- Map labels to 0-based class indices ----
    labels_list = [int(label_dict[name]) for name in image_names]
    unique_labels = sorted(list(set(labels_list)))
    label_map = {v: i for i, v in enumerate(unique_labels)}
    labels_tensor = torch.tensor([label_map[l] for l in labels_list], dtype=torch.long)
    num_classes = len(unique_labels)
    print(f"Number of classes: {num_classes}")
    print(f"Label mapping: {label_map}")

    # ---- Convert images to angles ----
    angles_np = images_to_angles_batch(images, n_wires=N_WIRES)
    angles_np = np.asarray(angles_np, dtype=np.float32)
    angles_np = angles_np / 255.0 * 2 * np.pi  # normalize

    # ---- Pad/truncate to N_WIRES ----
    def ensure_length_rowwise(arr, target=N_WIRES):
        fixed = []
        for row in arr:
            r = np.ravel(row).astype(np.float32)
            if r.size == target:
                fixed.append(r)
            elif r.size > target:
                fixed.append(r[:target])
            else:
                pad = np.zeros(target - r.size, dtype=np.float32)
                fixed.append(np.concatenate([r, pad]))
        return np.stack(fixed).astype(np.float32)

    angles_np = ensure_length_rowwise(angles_np, N_WIRES)
    train_tensor = torch.from_numpy(angles_np).to(torch.float32)

    # ---- Dataset / DataLoader ----
    dataset = TensorDataset(train_tensor, labels_tensor)
    if batch_size is None:
        batch_size = min(16, len(dataset))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ---- Model / optimizer / loss ----
    model = DigitRecognizerHQCNN(num_classes=num_classes, use_encoder=False).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # ---- Training loop ----
    history_loss = []
    history_acc = []

    epoch_bar = tqdm(range(1, num_epochs+1), desc="Epochs")
    for epoch in epoch_bar:
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        batch_bar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
        for angles_batch, labels_batch in batch_bar:
            angles_batch, labels_batch = angles_batch.to(device), labels_batch.to(device)
            optimizer.zero_grad()
            logits = model(angles_batch)
            loss = criterion(logits, labels_batch)
            loss.backward()
            optimizer.step()

            # Metrics
            epoch_loss += loss.item() * angles_batch.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels_batch).sum().item()
            total += labels_batch.size(0)

            batch_bar.set_postfix({
                "batch_loss": f"{loss.item():.4f}",
                "batch_acc": f"{correct/total:.3f}"
            })

        avg_loss = epoch_loss / total
        epoch_acc = correct / total
        history_loss.append(avg_loss)
        history_acc.append(epoch_acc)

        epoch_bar.set_postfix({
            "epoch_loss": f"{avg_loss:.4f}",
            "epoch_acc": f"{epoch_acc:.4f}"
        })

    # ---- Save model and angles ----
    torch.save(model.state_dict(), save_model_path)
    with open(save_angles_path, "wb") as f:
        pickle.dump(angles_np, f)
    print(f"Saved model -> {save_model_path}")
    print(f"Saved angles -> {save_angles_path}")

    # ---- Plot training curves ----
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history_loss, marker='o')
    plt.title("Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(history_acc, marker='o')
    plt.title("Training Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.show()
    print(f"Saved plot -> {plot_path}")

    # ---- Optional test evaluation ----
    if test_labels_json:
        with open(test_labels_json, "r") as f:
            test_data = json.load(f)
        test_labels_dict = test_data["test_labels"]
        image_names = list(test_labels_dict.keys())
        test_labels_raw = [int(test_labels_dict[name]) for name in image_names]

        # Map test labels to nearest training label
        unique_labels_array = np.array(unique_labels)
        mapped_test_labels = []
        for lbl in test_labels_raw:
            idx = np.argmin(np.abs(unique_labels_array - lbl))
            mapped_test_labels.append(unique_labels_array[idx])

        # Convert to angles for HQCNN input
        min_label = min(unique_labels)
        max_label = max(unique_labels)
        angles_test = (np.array(mapped_test_labels, dtype=np.float32) - min_label) / (max_label - min_label) * 2 * np.pi
        angles_test = np.stack([np.full(N_WIRES, a, dtype=np.float32) for a in angles_test])
        angles_test_tensor = torch.from_numpy(angles_test).to(torch.float32).to(device)

        # Predict
        model.eval()
        with torch.no_grad():
            logits_test = model(angles_test_tensor)
            probs_test = torch.softmax(logits_test, dim=1)
            preds_test = torch.argmax(probs_test, dim=1).cpu().numpy()

        print("\nPredictions vs Test Labels (mapped to nearest training class):")
        for i, name in enumerate(image_names):
            pred_class = preds_test[i]
            conf = probs_test[i, pred_class].item()
            original_val = test_labels_raw[i]
            mapped_val = mapped_test_labels[i]
            print(f"{name}: Predicted {pred_class} (conf {conf:.2f}), Original {original_val}, Mapped {mapped_val}")

        # Optional true labels
        if "true_labels" in test_data:
            true_labels_dict = test_data["true_labels"]
            true_labels_raw = [int(true_labels_dict[name]) for name in image_names]
            true_labels_mapped = np.array([label_map.get(lbl, -1) for lbl in true_labels_raw])
            valid_idx = true_labels_mapped != -1
            cm_test = confusion_matrix(true_labels_mapped[valid_idx], preds_test[valid_idx], labels=np.arange(num_classes))
            disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test,
                                               display_labels=[str(l) for l in unique_labels])

            plt.figure(figsize=(8,8))
            disp_test.plot(cmap=plt.cm.Reds, values_format='d')
            plt.title("Confusion Matrix on Test Set")
            plt.show()

            acc_test = np.sum(true_labels_mapped[valid_idx] == preds_test[valid_idx]) / valid_idx.sum()
            print(f"Test set accuracy: {acc_test*100:.2f}%")

    return model, label_map, unique_labels


if __name__ == "__main__":
    # Example usage
    train_and_eval_hqcnn(
        data_path="train_assets",
        labels_json="labels.json",
        num_epochs=50,
        lr=1e-3,
        test_labels_json="eval_assets/labels.json"
    )
