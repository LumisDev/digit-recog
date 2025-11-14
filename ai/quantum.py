# ai/quantum.py
import pennylane as qml
import pennylane.numpy as pnp
import torch
from pennylane import qnn

# --- Config
DEV_NAME = "default.qubit"
N_WIRES = 16
Q_LAYERS = 2

# device
dev = qml.device(DEV_NAME, wires=N_WIRES, shots=None)


# Define a qnode expecting exactly N_WIRES inputs and weights shaped (Q_LAYERS, N_WIRES, 3)
def qnode_fn(inputs, weights):
    """
    inputs: length-N_WIRES array (angles)
    weights: shape (Q_LAYERS, N_WIRES, 3)
    """
    # validation at QNode entry (helps catching wrong shapes early)
    if inputs is None:
        raise ValueError("qnode: inputs is None")
    # inputs may be a 1D array (for single sample) or shape (N_WIRES,)
    if len(inputs) != N_WIRES:
        raise ValueError(f"qnode expected {N_WIRES} inputs, got {len(inputs)}")

    # encoding
    for i in range(N_WIRES):
        qml.RY(inputs[i], wires=i)

    # variational ansatz
    qml.templates.StronglyEntanglingLayers(weights, wires=list(range(N_WIRES)))

    # measure PauliZ expectations on each wire
    return [qml.expval(qml.PauliZ(i)) for i in range(N_WIRES)]


# Create a template shape: weight shape for StronglyEntanglingLayers
weight_shape = (Q_LAYERS, N_WIRES, 3)

# qnode wrapper for Pennylane workflow
_qnode = qml.qnode(dev, interface="torch", diff_method="backprop")(qnode_fn)

# Build TorchLayer factory (call build_torch_qnode() to get a module)
def build_torch_qnode(dtype=torch.float32):
    """
    Returns a torch.nn.Module (TorchLayer) that maps (B, N_WIRES) -> (B, N_WIRES).
    Call: torch_qnode = build_torch_qnode() and use torch_qnode(angles_tensor).
    """
    # the weight shapes (the TorchLayer will register these as torch parameters)
    in_shape = (N_WIRES,)      # single-sample input shape
    weight_shapes = {"weights": weight_shape}

    # use a wrapper that casts inputs to the specified dtype
    torch_layer = qnn.TorchLayer(_qnode, weight_shapes)

    # Optional: enforce dtype by wrapping forward (so user doesn't need to cast)
    class TorchQNodeWrapper(torch.nn.Module):
        def __init__(self, layer, dtype):
            super().__init__()
            self.layer = layer
            self.dtype = dtype

        def forward(self, x):
            # x expected shape (B, N_WIRES)
            if x.dim() != 2 or x.shape[1] != N_WIRES:
                raise ValueError(f"TorchQNodeWrapper expected input shape (B, {N_WIRES}), got {tuple(x.shape)}")
            if x.dtype != self.dtype:
                x = x.to(self.dtype)
            return self.layer(x)

    return TorchQNodeWrapper(torch_layer, dtype)

# For convenience, provide a default instance (float32)
torch_qnode = build_torch_qnode(torch.float32)
