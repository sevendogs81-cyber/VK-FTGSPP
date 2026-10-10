from pathlib import Path

import numpy as np
import pytest

from ftgspp.io.model import NativeGaussians
from ftgspp.io.ply import export_4dgs


def model():
    return NativeGaussians(
        means=np.array([[1, 2, 3]], np.float32), log_scales=np.zeros((1, 3)),
        quaternions=np.array([[1, 0, 0, 0]]), opacity_logits=np.zeros((1, 1)),
        sh_coefficients=np.zeros((1, 16, 3)), times=np.zeros((1, 1)),
        duration_raw=np.zeros((1, 1)), velocities=np.zeros((1, 3)),
        gate_raw=np.zeros((1, 1)), max_duration=1.,
    )


@pytest.mark.parametrize("exists", [False, True])
def test_failed_ply_write_never_publishes_partial_output(tmp_path, monkeypatch, exists):
    target = tmp_path / "model.ply"
    if exists:
        target.write_bytes(b"previous model")
    original = Path.open

    class FailingWriter:
        def __init__(self, path):
            self.path = path
        def __enter__(self):
            self.stream = original(self.path, "wb")
            return self
        def __exit__(self, *args):
            self.stream.close()
        def write(self, data):
            self.stream.write(data)
            raise OSError("injected disk error")

    def patched(path, *args, **kwargs):
        return FailingWriter(path) if args == ("wb",) else original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", patched)
        with pytest.raises(OSError, match="injected"):
            export_4dgs(target, model())
    if exists:
        assert target.read_bytes() == b"previous model"
    else:
        assert not target.exists()
    assert sorted(tmp_path.iterdir()) == ([target] if exists else [])
    # A retry publishes a complete binary PLY without any stale temporary file.
    export_4dgs(target, model())
    header, payload = target.read_bytes().split(b"end_header\n", 1)
    property_count = header.count(b"property float ")
    assert len(payload) == property_count * 4
    np.testing.assert_array_equal(np.frombuffer(payload, dtype="<f4")[:3], [1, 2, 3])
    assert list(tmp_path.iterdir()) == [target]
