import os
from pathlib import Path

import nibabel as nib
import numpy as np


def _list_sorted(folder):
    return sorted(str(Path(folder) / f) for f in os.listdir(folder))


def read_template_images(data_root):
    folder = Path(data_root) / "dataset/OASIS-TRT-20/OASIS-TRT_brains_to_OASIS_Atropos_template"
    return _list_sorted(folder)


def read_template_labels(data_root):
    folder = Path(data_root) / "dataset/OASIS-TRT-20/OASIS-TRT_labels_to_OASIS_Atropos_template"
    return _list_sorted(folder)


def read_move_data(data_root):
    folder = Path(data_root) / "dataset/OASIS-TRT-20/OASIS-TRT_brains_in_MNI152"
    return _list_sorted(folder)


def read_move_labels(data_root):
    folder = Path(data_root) / "dataset/OASIS-TRT-20/OASIS-TRT_labels_in_MNI152"
    return _list_sorted(folder)


def center_crop_or_pad_3d(image, target_shape=(160, 192, 224)):
    """Center crop or pad a 3D image to target_shape."""
    result = image

    for axis, target in enumerate(target_shape):
        size = result.shape[axis]
        diff = target - size

        if diff > 0:
            pad_before = diff // 2
            pad_after = diff - pad_before
            pad_width = [(0, 0)] * result.ndim
            pad_width[axis] = (pad_before, pad_after)
            result = np.pad(result, pad_width, mode="constant")
        elif diff < 0:
            crop = -diff
            crop_before = crop // 2
            crop_after = crop - crop_before
            slices = [slice(None)] * result.ndim
            slices[axis] = slice(crop_before, size - crop_after)
            result = result[tuple(slices)]

    return result


def read_mri(path, target_shape=(160, 192, 224), normalize=True):
    image = nib.load(path).get_fdata()

    if normalize:
        max_value = image.max()
        if max_value > 0:
            image = image / max_value

    image = center_crop_or_pad_3d(image, target_shape)
    image = image.astype("float32")

    return np.reshape(image, (1,) + image.shape + (1,))


def example_gen(move_images, template_images, move_labels, template_labels, vol_size):
    while True:
        idx = np.random.randint(len(move_images))

        x = read_mri(move_images[idx], target_shape=vol_size, normalize=True)
        y = read_mri(template_images[idx], target_shape=vol_size, normalize=True)

        x_label = read_mri(move_labels[idx], target_shape=vol_size, normalize=True)
        y_label = read_mri(template_labels[idx], target_shape=vol_size, normalize=True)

        y = np.concatenate([y, y_label], axis=-1)

        yield x, y, x_label
