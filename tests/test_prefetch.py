import threading
from contextvars import ContextVar

import numpy as np
import pytest
import tifffile
import torch

from jdll_unet import train, trainer
from jdll_unet.config import parse_training_config
from jdll_unet.image_reading import ImageReadSession, current_read_session
from jdll_unet.prefetch import BatchPrefetcher


def test_order_context_and_repeated_epochs():
    value = ContextVar("test_prefetch_value", default="missing")
    value.set("run-context")
    main = threading.get_ident()

    class Loader:
        def __iter__(self):
            assert threading.get_ident() == main
            return iter((index, value.get()) for index in range(6))

    loader = BatchPrefetcher(Loader())
    try:
        for _ in range(2):
            assert list(loader) == [(index, "run-context") for index in range(6)]
    finally:
        loader.close()
    loader.close()
    with pytest.raises(RuntimeError, match="closed"):
        list(loader)


def test_producer_exception_is_propagated():
    def source():
        yield 1
        raise ValueError("source failed")

    loader = BatchPrefetcher(source())
    try:
        iterator = iter(loader)
        assert next(iterator) == 1
        with pytest.raises(ValueError, match="source failed"):
            next(iterator)
    finally:
        loader.close()


def test_iterator_creation_failure_does_not_leave_loader_active():
    class Loader:
        failed = False

        def __iter__(self):
            if not self.failed:
                self.failed = True
                raise ValueError("iterator creation failed")
            return iter([1])

    loader = BatchPrefetcher(Loader())
    try:
        with pytest.raises(ValueError, match="iterator creation failed"):
            list(loader)
        assert list(loader) == [1]
    finally:
        loader.close()


def test_image_reading_events_stay_on_consumer_thread():
    owner = threading.get_ident()
    events = []

    def emit(name, **kwargs):
        assert threading.get_ident() == owner
        events.append((name, kwargs))

    session = ImageReadSession(emit)

    def source():
        session.emit("warning", reason="region_fallback")
        yield 1

    loader = BatchPrefetcher(source(), read_session=session)
    try:
        assert list(loader) == [1]
        session.emit("warning", reason="main_thread")
    finally:
        loader.close()
    assert events == [("warning", {"reason": "region_fallback"}), ("warning", {"reason": "main_thread"})]
    assert session.emit is emit


def test_early_close_waits_for_inflight_batch():
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    def source():
        yield 1
        entered.set()
        assert release.wait(5)
        finished.set()
        yield 2

    loader = BatchPrefetcher(source())
    iterator = iter(loader)
    assert next(iterator) == 1
    assert entered.wait(5)
    closer = threading.Thread(target=loader.close)
    closer.start()
    release.set()
    closer.join(5)
    iterator.close()
    assert finished.is_set() and not closer.is_alive()


@pytest.mark.parametrize("fail_training", [False, True])
def test_cleanup_does_not_replace_training_error(monkeypatch, fail_training):
    events = []
    state = {}

    def callback(event):
        events.append(event["type"])
        if event["type"] == "warning":
            raise ValueError("warning callback failed")

    def fake_train(config, task, *, control, _runtime):
        session = current_read_session()
        state["session"] = session
        state["emit"] = session.emit
        loader = BatchPrefetcher([], read_session=session)
        _runtime["batch_prefetcher"] = loader
        state["loader"] = loader
        loader.executor.submit(session.emit, "warning").result(timeout=5)
        if fail_training:
            raise RuntimeError("original training failure")
        return {"cancelled": False}

    monkeypatch.setattr(trainer, "_train", fake_train)
    expected = RuntimeError if fail_training else ValueError
    message = "original training failure" if fail_training else "warning callback failed"
    with pytest.raises(expected, match=message):
        train({}, task=callback)
    assert events == (["error"] if fail_training else ["warning", "error"])
    assert state["loader"].closed
    assert state["session"].emit is state["emit"]
    assert not any(thread.name.startswith("jdll-batch") for thread in threading.enumerate())


@pytest.mark.parametrize("setting", [True, False, "auto"])
def test_prefetch_config_round_trip(tmp_path, setting):
    config = parse_training_config({"dataset_path": str(tmp_path), "output_dir": str(tmp_path),
                                    "model_name": "test", "batch_prefetch": setting})
    assert config.batch_prefetch == setting


def test_invalid_prefetch_config_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="batch_prefetch"):
        parse_training_config({"dataset_path": str(tmp_path), "output_dir": str(tmp_path),
                               "model_name": "test", "batch_prefetch": "sometimes"})


def test_prefetch_preserves_training_weights(tmp_path):
    images, masks = tmp_path / "data/images", tmp_path / "data/masks"
    images.mkdir(parents=True)
    masks.mkdir()
    rng = np.random.default_rng(4)
    for index in range(4):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[8:17, 8:17] = 2
        mask[20:27, 20:27] = 7
        tifffile.imwrite(images / f"{index}.tif", rng.random(mask.shape, dtype=np.float32))
        tifffile.imwrite(masks / f"{index}.tif", mask)
    states = []
    for prefetch in (False, True):
        output = tmp_path / str(prefetch)
        train({"dataset_path": images.parent, "output_dir": output, "model_name": "prefetch-check",
               "architecture": "resenc-tiny-2d", "task": "instance_friendly", "device": "cpu",
               "epochs": 2, "steps_per_epoch": 3, "batch_size": 2, "effective_batch_size": 2,
               "patch_size": [32, 32], "preview_count": 0, "batch_prefetch": prefetch,
               "validation": {"mode": "light", "light_steps": 1}})
        states.append(torch.load(output / "weights_last.pt", map_location="cpu", weights_only=False)["state_dict"])
    assert states[0].keys() == states[1].keys()
    for name in states[0]:
        assert torch.equal(states[0][name], states[1][name]), name
    assert not any(thread.name.startswith("jdll-batch") for thread in threading.enumerate())
