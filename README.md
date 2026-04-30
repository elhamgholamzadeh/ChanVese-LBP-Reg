# MRI Registration Training

Git-ready reproduction of the original Colab/Jupyter MRI registration code.

## Structure

```text
src/
  spatial_transformer.py   # Dense3DSpatialTransformer layer
  model.py                 # U-Net / VoxelMorph-style model
  losses.py                # gradient, statistical, LBP, NCC losses
  data.py                  # NIfTI loading, cropping/padding, generators
  train.py                 # training entry point
```

## Install

```bash
pip install -r requirements.txt
```

## Colab setup

In Colab only:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Then run:

```bash
python -m src.train \
  --data-root "/content/drive/MyDrive/Colab Notebooks/azizkhani" \
  --initial-weights "/content/drive/MyDrive/Colab Notebooks/azizkhani/models/vm1_cc.h5" \
  --output-dir "/content/drive/MyDrive/Colab Notebooks/azizkhani/models" \
  --n-iterations 2000 \
  --model-save-iter 100
```

## Notes

Large data and model weights are ignored by Git. Keep datasets and `.h5` weights outside commits.


## References

This work builds upon the VoxelMorph framework:

Balakrishnan, G., Zhao, A., Sabuncu, M. R., Guttag, J., & Dalca, A. V. (2019).  
VoxelMorph: A Learning Framework for Deformable Medical Image Registration.  
IEEE Transactions on Medical Imaging.

https://arxiv.org/abs/1809.05231
