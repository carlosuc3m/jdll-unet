import json
import shutil
from dataclasses import asdict
from unittest.mock import Mock

import numpy as np
import pytest
import tifffile
import torch
from PIL import Image

from jdll_unet import image_reading
from jdll_unet.config import ArchitectureConfig
from jdll_unet.errors import DataFormatError
from jdll_unet.geometry import DomainReader, _array_info, inspect_pair, load_domain_image, load_domain_mask
from jdll_unet.image_reading import ImageReadSession, image_reading_session
from jdll_unet.infer import _load_input, infer
from jdll_unet.io import ImageMaskPair, discover_dataset, load_array, load_image, load_mask
from jdll_unet.model import build_unet
from jdll_unet.planning import read_spacing
from jdll_unet.trainer import train


def write_image(path, array, fmt="TIFF", **kwargs):
    if fmt == "TIFF":
        tifffile.imwrite(path, array, photometric="minisblack", **kwargs)
    else:
        Image.fromarray(array).save(path, format=fmt, **kwargs)
    return path


def track_readers(monkeypatch):
    opened = Mock(wraps=image_reading._open_format)
    probe = Mock(wraps=image_reading._signature_format)
    monkeypatch.setattr(image_reading, "_open_format", opened)
    monkeypatch.setattr(image_reading, "_signature_format", probe)
    return opened, probe


@pytest.mark.parametrize("fmt,suffix", [("TIFF", ".tif"), ("PNG", ".png"), ("BMP", ".bmp"), ("JPEG", ".jpg")])
@pytest.mark.parametrize("rgb", [False, True])
def test_normal_path_preserves_samples_without_signature_probe(tmp_path, monkeypatch, fmt, suffix, rgb):
    array = np.full((12, 13, 3) if rgb else (12, 13), 17, dtype=np.uint8)
    path = tmp_path / ("image" + suffix)
    if fmt == "TIFF":
        tifffile.imwrite(path, array, photometric="rgb" if rgb else "minisblack")
    else:
        write_image(path, array, fmt)
    opened, probe = track_readers(monkeypatch)
    decoded = Mock(wraps=image_reading.decode_pixels)
    monkeypatch.setattr(image_reading, "decode_pixels", decoded)
    np.testing.assert_array_equal(load_array(path), array)
    assert opened.call_count == decoded.call_count == 1
    probe.assert_not_called()
    loaded = load_image(path)
    assert loaded.shape == ((3, 12, 13) if rgb else (1, 12, 13))


@pytest.mark.parametrize("fmt,suffix", [("PNG", ".png"), ("TIFF", ".tif")])
def test_uint16_mask_labels(tmp_path, fmt, suffix):
    labels = np.resize(np.array([0, 256, 1024, 65535], dtype=np.uint16), (12, 13))
    path = write_image(tmp_path / ("mask" + suffix), labels, fmt)
    np.testing.assert_array_equal(load_mask(path), labels)
    assert load_array(path).dtype.itemsize >= 2


def test_learned_preferences_and_mixed_formats(tmp_path, monkeypatch):
    array = np.arange(5 * 24 * 28, dtype=np.uint16).reshape(5, 24, 28)
    paths = [write_image(tmp_path / f"{i}.png", array, metadata={"axes": "ZYX"}) for i in range(3)]
    real_png = write_image(tmp_path / "real.png", array[0], "PNG")
    opened, probe = track_readers(monkeypatch)
    events = []
    with image_reading_session(lambda kind, **event: events.append(event)):
        np.testing.assert_array_equal(load_array(paths[0]), array)
        assert [call.args[1] for call in opened.call_args_list] == ["PNG", "TIFF"]
        opened.reset_mock()
        np.testing.assert_array_equal(load_array(paths[1]), array)
        assert [call.args[1] for call in opened.call_args_list] == ["TIFF"]
        np.testing.assert_array_equal(load_array(real_png), array[0])
        np.testing.assert_array_equal(load_array(paths[2]), array)
    assert probe.call_count == 3
    assert [event["actual_format"] for event in events] == ["TIFF", "PNG"]


@pytest.mark.parametrize("suffix", [".tif", ".tiff", ".TIF", ".TIFF"])
def test_reverse_preference_learning(tmp_path, monkeypatch, suffix):
    array = np.arange(80, dtype=np.uint8).reshape(8, 10)
    paths = [write_image(tmp_path / (str(i) + suffix), array, "PNG") for i in range(2)]
    opened, _ = track_readers(monkeypatch)
    session = ImageReadSession()
    for path in paths:
        np.testing.assert_array_equal(load_array(path, session=session), array)
    assert [call.args[1] for call in opened.call_args_list] == ["TIFF", "PNG", "PNG"]


def test_preferences_are_isolated_by_folder_role_and_run(tmp_path, monkeypatch):
    (tmp_path / "other").mkdir()
    data = np.ones((8, 9), dtype=np.uint8)
    path = write_image(tmp_path / "source.png", data)
    other = write_image(tmp_path / "other" / "source.png", data)
    opened, _ = track_readers(monkeypatch)
    with image_reading_session():
        load_array(path)
        opened.reset_mock()
        load_mask(path)
        load_array(other)
        assert [call.args[1] for call in opened.call_args_list] == ["PNG", "TIFF", "PNG", "TIFF"]
    opened.reset_mock()
    with image_reading_session():
        load_array(path)
    assert [call.args[1] for call in opened.call_args_list] == ["PNG", "TIFF"]


@pytest.mark.parametrize("suffix", [".png", ".bmp", ".jpg"])
@pytest.mark.parametrize(
    "axes,shape", [("ZYX", (5, 24, 28)), ("CZYX", (2, 5, 12, 13)), ("ZYXC", (5, 12, 13, 2)), ("ZCYX", (5, 2, 12, 13))]
)
def test_renamed_tiff_volume_all_read_paths(tmp_path, suffix, axes, shape):
    array = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 17
    image = write_image(tmp_path / ("image" + suffix), array, metadata={"axes": axes})
    spatial = tuple(shape[axes.index(axis)] for axis in "ZYX")
    labels = np.ones(spatial, dtype=np.uint16) * 1024
    mask = write_image(tmp_path / "mask.bmp", labels, metadata={"axes": "ZYX"})
    np.testing.assert_array_equal(load_array(image), array)
    pair, info = inspect_pair(ImageMaskPair(image, mask, "case"))
    assert pair.source_kind == "volume" and pair.spatial_shape == spatial
    assert info["image_axes_provenance"] == "explicit_tiff_axes"
    expected = array.transpose(tuple(axes.index(axis) for axis in "CZYX")) if "C" in axes else array[None]
    np.testing.assert_array_equal(load_domain_image(pair), expected)
    np.testing.assert_array_equal(_load_input({"image_path": image}, "3d"), expected)
    np.testing.assert_array_equal(load_domain_mask(pair, "3d"), labels)
    np.testing.assert_array_equal(load_mask(mask, "3d"), labels)


@pytest.mark.parametrize("kind", ["ome", "imagej"])
def test_renamed_spacing_axes_and_sidecar_precedence(tmp_path, kind):
    array = np.ones((5, 12, 13), dtype=np.uint16)
    options = (
        {"ome": True, "metadata": {"axes": "ZYX", "PhysicalSizeZ": 2.5, "PhysicalSizeY": 0.5, "PhysicalSizeX": 0.25}}
        if kind == "ome"
        else {"imagej": True, "metadata": {"axes": "ZYX", "spacing": 2.5}, "resolution": (4, 2)}
    )
    original = write_image(tmp_path / "original.tif", array, **options)
    renamed = tmp_path / "renamed.bmp"
    shutil.copyfile(original, renamed)
    assert _array_info(original) == _array_info(renamed)
    assert read_spacing(original) == read_spacing(renamed) == ((2.5, 0.5, 0.25), "embedded_metadata")
    renamed.with_suffix(".json").write_text(json.dumps({"spacing": [3, 2, 1]}))
    assert read_spacing(renamed) == ((3, 2, 1), "sidecar")


@pytest.mark.parametrize("fmt", ["PNG", "BMP"])
def test_palette_ids_survive_direct_and_prepared_domain_reads(tmp_path, fmt):
    labels = np.resize(np.array([0, 1, 2], dtype=np.uint8), (8, 9))
    palette = Image.new("P", (9, 8))
    palette.putdata(labels.ravel())
    palette.putpalette([0, 0, 0, 255, 0, 0, 255, 0, 0] + [0] * (768 - 9))
    path = tmp_path / "palette.tif"
    palette.save(path, format=fmt, **({"transparency": bytes([255, 0, 127])} if fmt == "PNG" else {}))
    reader = DomainReader()
    image = reader.read(path)
    assert image.shape == (8, 9, 3)
    np.testing.assert_array_equal(reader.read(path, role="mask"), labels)
    assert len(reader.cache) == 2
    np.testing.assert_array_equal(load_mask(path), labels)
    pair, info = inspect_pair(ImageMaskPair(path, path, "palette"), reader=reader)
    assert info["mask_channels"] == 1 and pair.label_values == (1, 2)
    np.testing.assert_array_equal(load_domain_mask(pair, reader=reader), labels)


def test_disguised_jpeg_mask_rejected_even_with_prepared_axes(tmp_path):
    path = write_image(tmp_path / "mask.png", np.ones((8, 9), dtype=np.uint8), "JPEG")
    pair = ImageMaskPair(path, path, "jpeg", image_axes="YX", mask_axes="YX")
    for read in (lambda: load_mask(path), lambda: load_domain_mask(pair), lambda: inspect_pair(pair)):
        with pytest.raises(DataFormatError, match="JPEG masks"):
            read()


def test_real_multichannel_mask_and_invalid_labels(tmp_path):
    labels = np.full((2, 5, 12, 13), 1024, dtype=np.uint16)
    labels[1] = 256
    path = write_image(tmp_path / "mask.png", labels, metadata={"axes": "CZYX"})
    with pytest.warns(RuntimeWarning, match="first channel"):
        np.testing.assert_array_equal(load_mask(path, "3d"), labels[0])
    for value, message in [(float("nan"), "non-finite"), (0.5, "non-integer"), (-1, "nonnegative")]:
        path = write_image(tmp_path / "invalid.bmp", np.full((8, 9), value, dtype=np.float32), metadata={"axes": "YX"})
        with pytest.raises(DataFormatError, match=message):
            load_domain_mask(ImageMaskPair(path, path, "invalid", image_axes="YX", mask_axes="YX"))


@pytest.mark.parametrize(
    "suffix,header", [(".png", b"\x89PNG\r\n\x1a\n"), (".tif", b"II*\x00"), (".png", b"unknown"), (".png", b"II*\x00")]
)
def test_failed_reads_do_not_poison_preferences(tmp_path, monkeypatch, suffix, header):
    path = tmp_path / ("corrupt" + suffix)
    path.write_bytes(header)
    opened, _ = track_readers(monkeypatch)
    events = []
    session = ImageReadSession(lambda kind, **event: events.append(event))
    with pytest.raises(DataFormatError, match="corrupt") as error:
        load_array(path, session=session)
    assert error.value.__cause__ is not None
    assert not session.preferences and not events
    formats = [call.args[1] for call in opened.call_args_list]
    assert formats == (
        ["PNG", "TIFF"] if suffix == ".png" and header == b"II*\x00" else ["TIFF" if suffix == ".tif" else "PNG"]
    )


@pytest.mark.parametrize("suffix", [".tif", ".png"])
def test_missing_codec_and_unrelated_exceptions_do_not_trigger_alternate_decoders(tmp_path, monkeypatch, suffix):
    path = write_image(tmp_path / ("compressed" + suffix), np.ones((8, 9), dtype=np.uint16))
    opened, probe = track_readers(monkeypatch)
    session = ImageReadSession()
    for error in (ValueError("requires missing codec"), MemoryError("out of memory"), TypeError("programming error")):
        monkeypatch.setattr(image_reading, "decode_pixels", Mock(side_effect=error))
        with pytest.raises(DataFormatError if isinstance(error, ValueError) else type(error)) as raised:
            load_array(path, session=session)
        assert str(error) in str(raised.value)
        assert [call.args[1] for call in opened.call_args_list] == (["TIFF"] if suffix == ".tif" else ["PNG", "TIFF"])
        assert probe.call_count == (suffix == ".png")
        assert not session.preferences and not session.notices
        opened.reset_mock()
        probe.reset_mock()


@pytest.mark.parametrize("bigtiff", [False, True])
@pytest.mark.parametrize("byteorder", ["<", ">"])
@pytest.mark.parametrize("compression", [None, "deflate"])
def test_tiff_variants_and_bounded_memmap_reads(tmp_path, bigtiff, byteorder, compression):
    array = np.arange(5 * 24 * 28, dtype=np.uint16).reshape(5, 24, 28)
    path = write_image(
        tmp_path / "volume.PNG",
        array,
        metadata={"axes": "ZYX"},
        bigtiff=bigtiff,
        byteorder=byteorder,
        compression=compression,
    )
    reader = DomainReader(max_bytes=1)
    loaded = reader.read(path)
    np.testing.assert_array_equal(loaded, array)
    assert reader.bytes == 0 and not reader.cache
    if compression is None:
        assert isinstance(loaded, np.memmap)


def test_metadata_inspection_does_not_decode_pixels(tmp_path, monkeypatch):
    png = write_image(tmp_path / "image.png", np.ones((8, 9), dtype=np.uint8), "PNG")
    tiff = write_image(tmp_path / "volume.tif", np.ones((5, 8, 9), dtype=np.uint16), metadata={"axes": "ZYX"})
    monkeypatch.setattr(tifffile.TiffPageSeries, "asarray", Mock(side_effect=AssertionError("decoded TIFF")))
    monkeypatch.setattr(Image.Image, "tobytes", Mock(side_effect=AssertionError("decoded raster")))
    _, probe = track_readers(monkeypatch)
    assert _array_info(png)[0] == (8, 9)
    assert _array_info(tiff)[0] == (5, 8, 9)
    probe.assert_not_called()


def test_multiple_series_and_animated_rasters_are_rejected(tmp_path):
    tiff = tmp_path / "series.png"
    with tifffile.TiffWriter(tiff) as writer:
        writer.write(np.ones((8, 9), dtype=np.uint8))
        writer.write(np.ones((4, 5), dtype=np.uint8))
    png = tmp_path / "animation.png"
    frames = [Image.new("L", (8, 9), value) for value in (1, 2)]
    frames[0].save(png, save_all=True, append_images=frames[1:])
    for path, message in ((tiff, "multiple TIFF series"), (png, "multiple raster frames")):
        session = ImageReadSession()
        with pytest.raises(DataFormatError, match=message):
            load_array(path, session=session)
        assert not session.preferences and not session.notices


def test_pyramid_levels_are_not_independent_series(tmp_path):
    path = tmp_path / "pyramid.png"
    with tifffile.TiffWriter(path) as writer:
        writer.write(np.ones((16, 16), dtype=np.uint16), subifds=1)
        writer.write(np.ones((8, 8), dtype=np.uint16), subfiletype=1)
    assert load_array(path).shape == (16, 16)


def test_training_preparation_and_file_inference_match_renamed_contents(tmp_path):
    data = tmp_path / "data"
    for folder in ("images", "masks"):
        (data / folder).mkdir(parents=True)
    for i in range(2):
        labels = np.zeros((24, 24), dtype=np.uint8)
        labels[5:19, 5:19] = 1
        write_image(data / "images" / f"{i}.png", labels.astype(np.float32), metadata={"axes": "YX"})
        write_image(data / "masks" / f"{i}.tif", labels, "PNG")
    events = []
    result = train(
        {
            "model_name": "readers",
            "output_dir": tmp_path / "model",
            "dataset_path": data,
            "architecture": "tiny-2d",
            "device": "cpu",
            "epochs": 1,
            "steps_per_epoch": 1,
            "patch_size": [16, 16],
            "effective_batch_size": 1,
            "preview_count": 0,
            "validation": {"mode": "light", "light_steps": 1},
        },
        task=events.append,
    )
    corrections = [event for event in events if event.get("reason") == "image_reader_correction"]
    assert len(corrections) == 2 and {event["role"] for event in corrections} == {"image", "mask"}
    original = data / "images" / "0.png"
    correctly_named = tmp_path / "correct.tif"
    shutil.copyfile(original, correctly_named)
    correct_mask = tmp_path / "correct_mask.png"
    shutil.copyfile(data / "masks" / "0.tif", correct_mask)
    prepared, _ = inspect_pair(discover_dataset(data).train[0])
    reference, _ = inspect_pair(ImageMaskPair(correctly_named, correct_mask, "correct"))
    assert prepared.spatial_shape == reference.spatial_shape
    assert prepared.label_values == reference.label_values
    np.testing.assert_array_equal(load_domain_image(prepared), load_domain_image(reference))
    np.testing.assert_array_equal(load_domain_mask(prepared), load_domain_mask(reference))
    config = {"model_path": result["model_path"], "device": "cpu", "tile_size": [16, 16]}
    expected = infer(config, {"image_path": correctly_named})
    events.clear()
    actual = infer(config, {"image_path": original}, callback=events.append)
    np.testing.assert_array_equal(
        actual["outputs"]["foreground_probability"], expected["outputs"]["foreground_probability"]
    )
    assert sum(event.get("reason") == "image_reader_correction" for event in events) == 1
    assert len(discover_dataset(data).train) == 2


def test_cpu_volume_file_inference_preserves_planes_after_renaming(tmp_path):
    architecture = ArchitectureConfig(name="tiny-3d", dimensions="3d", depth=2, channels=(2, 4), normalization="none")
    model = build_unet(architecture)
    config = {
        "format": "jdll-unet",
        "format_version": 1,
        "task": "binary_semantic",
        "architecture_config": asdict(architecture),
        "label_values": [1],
        "normalization": {"type": "none"},
    }
    model_path = tmp_path / "model.pt"
    torch.save(
        {"state_dict": model.state_dict(), "model_config": config, "architecture_config": asdict(architecture)},
        model_path,
    )
    array = np.arange(5 * 24 * 28, dtype=np.uint16).reshape(5, 24, 28)
    original = write_image(tmp_path / "volume.tif", array, metadata={"axes": "ZYX"})
    renamed = tmp_path / "volume.png"
    shutil.copyfile(original, renamed)
    request = {"model_path": model_path, "device": "cpu", "tile_size": [8, 16, 16]}
    expected = infer(request, {"image_path": original})["outputs"]["foreground_probability"]
    actual = infer(request, {"image_path": renamed})["outputs"]["foreground_probability"]
    assert expected.shape == (5, 24, 28)
    np.testing.assert_array_equal(actual, expected)
