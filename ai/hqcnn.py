# ai/hqcnn.py
import torch
import torch.nn as nn
import numpy as np
from ai.quantum import N_WIRES, torch_qnode

class DigitRecognizerHQCNN(nn.Module):
    def __init__(self, num_classes=2, use_encoder=True):
        super().__init__()
        self.use_encoder = use_encoder
        print(f"Quantum device layer: N_WIRES={N_WIRES}, use_encoder={use_encoder}")

        # Use the provided TorchLayer (it is a torch.nn.Module)
        self.qnode = torch_qnode

        if self.use_encoder:
            # small encoder that maps (B,1,H,W) -> (B, N_WIRES) in [-pi/2, pi/2]
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(8, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((2,2)),
                nn.Flatten(),
                nn.Linear(16*2*2, 32),
                nn.ReLU(),
                nn.Linear(32, N_WIRES),
                nn.Tanh()   # outputs in [-1,1]
            )
        else:
            self.encoder = None

        self.classifier = nn.Sequential(
            nn.Linear(N_WIRES, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # Two input modes:
        # - precomputed angles: x shape (B, N_WIRES)
        # - images (B,1,H,W) when use_encoder=True
        if x.dim() == 2 and x.shape[1] == N_WIRES:
            angles = x
        elif x.dim() == 4 and self.use_encoder:
            enc = self.encoder(x)  # (B, N_WIRES) in [-1,1]
            angles = enc * (np.pi/2)  # map to radians
        else:
            raise ValueError(f"DigitRecognizerHQCNN.forward: expected (B,{N_WIRES}) angles or (B,1,H,W) images; got {tuple(x.shape)}")

        # enforce shape/dtype before calling qnode
        if angles.dim() != 2 or angles.shape[1] != N_WIRES:
            raise ValueError(f"Angles must be shaped (B, {N_WIRES}); got {tuple(angles.shape)}")

        # Ensure dtype matches torch_qnode (TorchLayer built with float32)
        if angles.dtype != torch.float32:
            angles = angles.to(torch.float32)

        q_out = self.qnode(angles)  # (B, N_WIRES)
        logits = self.classifier(q_out)
        return logits
