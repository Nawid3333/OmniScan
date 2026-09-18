"""Tests for the turbo (CPU baseline) JPEG codec."""

from __future__ import annotations

import pytest
import torch

from omniscan.gpu.codec.turbo import TurboCodec
from omniscan.gpu.device import resolve_device
from tests.fixtures.images import plain_jpeg, rotated_jpeg

pytest.importorskip("torch")


def _read(path) -> bytes:
    return path.read_bytes()


def _mean(tensor: torch.Tensor) -> tuple[float, float, float]:
    """Per-channel mean of a uint8 [3, H, W] tensor, as floats."""
    return tuple(t.item() for t in tensor.float().mean(dim=(1, 2)))  # type: ignore[return-value]


def test_info_plain_baseline(tmp_path) -> None:
    data = _read(plain_jpeg(tmp_path / "a.jpg"))
    info = TurboCodec("cpu").info(data)
    assert (info.width, info.height, info.components) == (400, 300, 3)
    assert info.subsampling in {"444", "422", "420"}
    assert info.progressive is False


def test_info_exif_rotated(tmp_path) -> None:
    # info() is header-only and reports the *stored* dimensions; orientation is applied at decode time
    data = _read(rotated_jpeg(tmp_path / "r.jpg", size=(300, 400)))
    codec = TurboCodec("cpu")
    assert (codec.info(data).width, codec.info(data).height) == (400, 300)
    assert codec.decode(data).shape == (3, 400, 300)


def test_decode_plain(tmp_path) -> None:
    data = _read(plain_jpeg(tmp_path / "a.jpg", color=(200, 60, 60)))
    tensor = TurboCodec("cpu").decode(data)
    assert tensor.dtype == torch.uint8
    assert tensor.shape == (3, 300, 400)
    r, g, b = _mean(tensor)
    assert abs(r - 200) <= 3 and abs(g - 60) <= 3 and abs(b - 60) <= 3


def test_decode_into_stacks_at_offsets(tmp_path) -> None:
    datas = [
        _read(plain_jpeg(tmp_path / f"{i}.jpg", size=(400, 100 + 10 * i), color=c))
        for i, c in enumerate([(200, 60, 60), (60, 200, 60), (60, 60, 200)])
    ]
    heights = [100, 110, 120]
    offsets = [0, 100, 210]
    out = torch.empty((3, 330, 400), dtype=torch.uint8)
    TurboCodec("cpu").decode_into(datas, out, offsets)
    for y, h, c in zip(offsets, heights, [(200, 60, 60), (60, 200, 60), (60, 60, 200)], strict=True):
        r, g, b = _mean(out[:, y + h // 2 : y + h // 2 + 1, :])
        assert abs(r - c[0]) <= 3 and abs(g - c[1]) <= 3 and abs(b - c[2]) <= 3


def test_decode_into_width_mismatch_names_index(tmp_path) -> None:
    datas = [
        _read(plain_jpeg(tmp_path / "0.jpg", size=(400, 50))),
        _read(plain_jpeg(tmp_path / "1.jpg", size=(399, 50))),
    ]
    out = torch.empty((3, 100, 400), dtype=torch.uint8)
    with pytest.raises(ValueError, match=r"\b1\b"):
        TurboCodec("cpu").decode_into(datas, out, [0, 50])


def test_encode_decode_round_trip(tmp_path) -> None:
    codec = TurboCodec("cpu")
    source = torch.full((3, 64, 64), 0, dtype=torch.uint8)
    source[0], source[1], source[2] = 180, 90, 40
    encoded = codec.encode(source, quality=92, subsampling="420")
    back = codec.decode(encoded)
    r, g, b = _mean(back)
    assert abs(r - 180) <= 3 and abs(g - 90) <= 3 and abs(b - 40) <= 3
    assert codec.info(encoded).subsampling == "420"


def test_available_and_cpu_fallback(tmp_path) -> None:
    codec = TurboCodec("cpu")
    assert codec.available() is True
    data = _read(plain_jpeg(tmp_path / "a.jpg"))
    tensor = codec.decode(data)
    assert tensor.shape == (3, 300, 400)


def test_close_twice_then_decode(tmp_path) -> None:
    codec = TurboCodec("cpu")
    codec.close()
    codec.close()  # idempotent
    # documented behaviour: the pool is lazily recreated, so decode still works after close()
    data = _read(plain_jpeg(tmp_path / "a.jpg"))
    assert codec.decode(data).shape == (3, 300, 400)


@pytest.mark.gpu
def test_decode_into_gpu_strip(tmp_path) -> None:
    n = 40
    datas = [_read(plain_jpeg(tmp_path / f"{i}.jpg", size=(400, 300), color=(120, 40, 90))) for i in range(n)]
    out = torch.empty((3, 300 * n, 400), dtype=torch.uint8, device=resolve_device())
    TurboCodec(resolve_device()).decode_into(datas, out, [300 * i for i in range(n)])
    assert out.is_cuda
    cpu = out.cpu()
    for i in range(n):
        row = cpu[:, 300 * i + 150, :]
        r, g, b = (row[0].float().mean().item(), row[1].float().mean().item(), row[2].float().mean().item())
        assert abs(r - 120) <= 3 and abs(g - 40) <= 3 and abs(b - 90) <= 3


@pytest.mark.gpu
def test_decode_on_cuda(tmp_path) -> None:
    data = _read(plain_jpeg(tmp_path / "a.jpg", size=(400, 300), color=(200, 60, 60)))
    tensor = TurboCodec(resolve_device()).decode(data)
    assert tensor.is_cuda
    assert tensor.shape == (3, 300, 400)
