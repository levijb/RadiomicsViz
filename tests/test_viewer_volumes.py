"""Viewer volumes: geometry (T2) and robustness of conversion and windows (T6). No Flask."""

from __future__ import annotations

import gzip
import math

import nibabel as nib
import numpy as np
import pytest
import SimpleITK as sitk

import _viewer_support as support
from radiomicviz.viewer._volumes import (
    VolumeError,
    VolumeStore,
    display_window,
    nifti_bytes_from_array,
    read_array,
    read_geometry,
)


def _rotation(z_degrees: float, x_degrees: float) -> tuple[float, ...]:
    a, b = math.radians(z_degrees), math.radians(x_degrees)
    rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]])
    return tuple((rz @ rx).reshape(-1))


def _converted(path):
    array, geometry = read_array(path)
    return nib.Nifti1Image.from_bytes(nifti_bytes_from_array(array, geometry)), geometry


@pytest.mark.parametrize("direction", [None, _rotation(30, 15)])
def test_nrrd_conversion_matches_simpleitk_world_coordinates(tmp_path, direction):
    values = np.zeros((7, 6, 5), dtype=np.float64)  # (z, y, x)
    values[2, 3, 1] = 100.0  # asymmetric marker, left of centre in x
    values[5, 1, 4] = 50.0
    path = support.write_volume(tmp_path / "map.nrrd", values, spacing=(0.8, 1.1, 2.5),
                                origin=(10.0, -20.0, 35.0), direction=direction)
    image = sitk.ReadImage(str(path))
    converted, geometry = _converted(path)
    data = np.asanyarray(converted.dataobj)
    assert data.shape == image.GetSize()
    for index in [(0, 0, 0), (1, 3, 2), (4, 1, 5), (4, 5, 6), (2, 2, 3)]:
        lps = np.array(image.TransformIndexToPhysicalPoint(index))
        ras = converted.affine @ np.array([*index, 1.0])
        assert np.allclose(ras[:3], [-lps[0], -lps[1], lps[2]], atol=1e-4)
        assert data[index] == values[index[2], index[1], index[0]]
    assert geometry.oblique is (direction is not None)
    assert geometry.sheared is False


def test_cropped_map_keeps_its_own_origin_and_lies_on_the_region(tmp_path):
    image = support.write_volume(tmp_path / "image.nii.gz", support.texture(3), spacing=(0.9, 1.2, 2.0),
                                 origin=(-5.0, 7.0, 3.0), direction=_rotation(20, 0))
    mask = support.write_volume(tmp_path / "mask.nii.gz", support.block_mask((5, 3, 2), 40), spacing=(0.9, 1.2, 2.0),
                                origin=(-5.0, 7.0, 3.0), direction=_rotation(20, 0))
    maps = support.FakeVoxelExtractor().execute(str(image), str(mask), label=1, voxelBased=True)
    map_path = tmp_path / "original_firstorder_Mean.nrrd"
    sitk.WriteImage(maps["original_firstorder_Mean"], str(map_path))
    converted, _ = _converted(map_path)
    data = np.asanyarray(converted.dataobj)
    finite = np.argwhere(np.isfinite(data))
    map_world = (converted.affine @ np.c_[finite, np.ones(len(finite))].T).T[:, :3]
    mask_image = sitk.ReadImage(str(mask))
    mask_index = np.argwhere(sitk.GetArrayFromImage(mask_image) == 1)[:, ::-1]
    mask_world = np.array([mask_image.TransformIndexToPhysicalPoint(tuple(int(v) for v in index)) for index in mask_index])
    mask_world[:, :2] *= -1
    assert len(finite) == len(mask_index) == 40
    assert np.allclose(map_world.mean(axis=0), mask_world.mean(axis=0), atol=1e-4)
    low, high = mask_world.min(axis=0) - 1e-4, mask_world.max(axis=0) + 1e-4
    assert np.all(map_world >= low) and np.all(map_world <= high)


def test_display_windows_are_robust():
    constant = display_window(np.full((3, 3, 3), 4.0), kind="map")
    assert constant["constant"] and constant["cal_min"] < 4.0 < constant["cal_max"]
    nan = display_window(np.full((3, 3, 3), np.nan), kind="map")
    assert nan["all_nonfinite"] and nan["n_finite"] == 0
    values = np.array([-5.0, -1.0, 0.0, 2.0, np.inf, -np.inf, np.nan] * 3)
    mixed = display_window(values, kind="map")
    assert mixed["n_positive_infinite"] == 3 and mixed["n_negative_infinite"] == 3
    assert mixed["n_nonfinite"] == 9 and mixed["min"] == -5.0 and mixed["max"] == 2.0
    image = display_window(np.array([0.0] * 50 + [10.0, 20.0, 30.0]), kind="image")
    assert image["min"] == 10.0, "image windows ignore exact-zero background"


def test_bad_files_give_clean_errors(tmp_path):
    store = VolumeStore(32)
    good = support.write_volume(tmp_path / "good.nii.gz", support.texture(1))
    truncated = tmp_path / "truncated.nii.gz"
    truncated.write_bytes(good.read_bytes()[: len(good.read_bytes()) // 2])
    with pytest.raises(VolumeError) as error:
        store.layer(truncated)
    assert error.value.code == "unreadable"
    corrupt = tmp_path / "corrupt.nrrd"
    corrupt.write_bytes(b"NRRD0004\ntype: double\ndimension: 3\nsizes: 4 4 4\nencoding: raw\n\nxx")
    with pytest.raises(VolumeError):
        store.layer(corrupt)
    with pytest.raises(VolumeError) as missing:
        store.layer(tmp_path / "missing.nrrd")
    assert missing.value.code == "missing"
    huge = tmp_path / "huge.nii.gz"
    header = nib.Nifti1Header()
    header.set_data_shape((4000, 4000, 4000))
    header.set_data_dtype(np.float32)
    header["vox_offset"] = 352
    header["magic"] = b"n+1"
    with gzip.open(huge, "wb") as handle:
        handle.write(header.binaryblock + b"\x00" * 4 + b"\x00" * 64)
    with pytest.raises(VolumeError) as large:
        store.layer(huge)
    assert large.value.code == "too_large"
    kind, payload, geometry = store.layer(good)
    assert kind == "file" and payload == good and geometry.size == (12, 12, 12)


def test_thin_and_border_regions_convert(tmp_path):
    values = np.full((6, 6, 6), np.nan)
    values[0, :, 2] = np.arange(6.0)  # one-voxel-thick region touching the border
    path = support.write_volume(tmp_path / "thin.nrrd", values)
    converted, _ = _converted(path)
    assert int(np.isfinite(np.asanyarray(converted.dataobj)).sum()) == 6
    stats = VolumeStore(16).stats(path, "feature_map")
    assert stats["n_finite"] == 6 and stats["cal_min"] < stats["cal_max"]
    assert read_geometry(path).size == (6, 6, 6)
