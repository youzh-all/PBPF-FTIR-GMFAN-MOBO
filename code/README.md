# Public custom-code components

This directory provides the central custom-code components used in the PBPF FT-IR/GMFAN workflow.

## Files

| File | Purpose |
|---|---|
| `gmfan_architecture.py` | PyTorch implementation of the FT-IR encoder, formulation-metadata encoder, gated multimodal fusion module, and model heads used by the GMFAN architecture. |
| `preprocessing.py` | FT-IR/DSC/UTM preprocessing utilities, composition-ID parsing, and formulation metadata encoding. |
| `curve_uncertainty_metrics.py` | Residual-quantile prediction intervals and curve-level PICP, interval-width, and interval-score calculations. |

## Installation

Create a Python environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Minimal architecture check

```bash
python - <<'PY'
import torch
from code.gmfan_architecture import AttentionBackbone

model = AttentionBackbone(meta_input_dim=11)
ftir = torch.zeros((2, 1800))
metadata = torch.zeros((2, 11))
print(model(ftir, metadata).shape)
PY
```

Expected output:

```text
torch.Size([2, 256])
```

## Data access and scope

The public repository provides representative input-format samples and manuscript-aligned metadata. The full experimental FT-IR, DSC, and UTM datasets can be requested from the corresponding authors. The code is released under the repository's MIT License.
