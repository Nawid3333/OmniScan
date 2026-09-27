"""Tests for omniscan.interchange.psd — layered PSD pages, read back with a small PSD reader (torch-free)."""

from __future__ import annotations

import hashlib
import io
import struct
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from omniscan.core.paths import ChapterPaths
from omniscan.core.schemas import BBox, IngestArtifact, LayoutItem, SourceFile
from omniscan.interchange.psd import PsdLayer, packbits, page_psd, write_psd
from omniscan.typeset.page_preview import render_page


def unpackbits(data: bytes, size: int) -> bytes:
    """PackBits decoding, as a PSD reader does it."""
    out = bytearray()
    i = 0
    while len(out) < size:
        n = data[i]
        i += 1
        if n < 128:
            out += data[i : i + n + 1]
            i += n + 1
        elif n > 128:
            out += bytes([data[i]]) * (257 - n)
            i += 1
    assert len(out) == size and i == len(data)
    return bytes(out)


def read_rle(data: bytes, pos: int, planes: int, height: int, width: int) -> tuple[list[np.ndarray], int]:
    """`planes` RLE planes starting with their row byte counts at `pos`; returns the planes and the end."""
    counts = struct.unpack_from(f">{planes * height}H", data, pos)
    pos += 2 * planes * height
    out = []
    for p in range(planes):
        rows = []
        for r in range(height):
            n = counts[p * height + r]
            rows.append(np.frombuffer(unpackbits(data[pos : pos + n], width), np.uint8))
            pos += n
        out.append(np.stack(rows))
    return out, pos


def read_psd(data: bytes) -> tuple[list[tuple[str, dict[int, np.ndarray]]], np.ndarray]:
    """(name, channel id -> plane) of every layer, bottom first, and the composite (uint8 [h, w, 3])."""
    signature, version, channels, height, width, depth, mode = struct.unpack_from(">4sH6xHIIHH", data, 0)
    assert (signature, version, channels, depth, mode) == (b"8BPS", 1, 3, 8, 3)
    pos = 26
    for _ in range(2):  # color mode data, image resources
        (length,) = struct.unpack_from(">I", data, pos)
        pos += 4 + length
    (section,) = struct.unpack_from(">I", data, pos)
    end_of_section = pos + 4 + section
    (info,) = struct.unpack_from(">I", data, pos + 4)
    info_start = pos + 8
    (count,) = struct.unpack_from(">h", data, info_start)
    pos += 10
    records = []
    for _ in range(count):
        top, left, bottom, right, n = struct.unpack_from(">4iH", data, pos)
        assert (top, left, bottom, right) == (0, 0, height, width)
        pos += 18
        chans = [struct.unpack_from(">hI", data, pos + 6 * i) for i in range(n)]
        pos += 6 * n
        assert data[pos : pos + 8] == b"8BIMnorm"
        (extra,) = struct.unpack_from(">I", data, pos + 12)
        name_len = data[pos + 16 + 8]
        name = data[pos + 16 + 9 : pos + 16 + 9 + name_len].decode()
        pos += 16 + extra
        records.append((name, chans))
    layers = []
    for name, chans in records:
        planes = {}
        for channel, length in chans:
            (compression,) = struct.unpack_from(">H", data, pos)
            assert compression == 1
            (plane,), end = read_rle(data, pos + 2, 1, height, width)
            assert end == pos + length
            planes[channel] = plane
            pos = end
        layers.append((name, planes))
    assert info_start + info - pos in (
        0,
        1,
    )  # the layer info length covers the records and channels (+ padding)
    (compression,) = struct.unpack_from(">H", data, end_of_section)
    assert compression == 1
    composite, end = read_rle(data, end_of_section + 2, 3, height, width)
    assert end == len(data)
    return layers, np.stack(composite, axis=-1)


@pytest.mark.parametrize(
    "row",
    [
        np.zeros(0, np.uint8),
        np.full(300, 255, np.uint8),  # one long run: repeat packets of at most 128
        np.array([1, 2, 3, 3, 3, 3, 4, 5, 5, 6], np.uint8),
        np.random.default_rng(0).integers(0, 256, 1000, dtype=np.uint8),  # busy: literals only
        np.repeat(np.arange(40, dtype=np.uint8), 7),
    ],
)
def test_packbits_round_trips(row: np.ndarray) -> None:
    encoded = packbits(row)
    assert unpackbits(encoded, len(row)) == row.tobytes()
    if len(row) == 300:
        assert len(encoded) == 6  # three repeat packets


def test_write_psd_round_trips_layers_and_the_composite() -> None:
    rng = np.random.default_rng(1)
    rgb = rng.integers(0, 256, (20, 30, 3), dtype=np.uint8)
    alpha = np.zeros((20, 30), np.uint8)
    alpha[5:10, 5:25] = 255
    composite = rng.integers(0, 256, (20, 30, 3), dtype=np.uint8)
    layers, back = read_psd(write_psd([PsdLayer("raw", rgb), PsdLayer("text", rgb, alpha)], composite))
    assert [name for name, _ in layers] == ["raw", "text"]
    assert sorted(layers[0][1]) == [0, 1, 2] and sorted(layers[1][1]) == [-1, 0, 1, 2]
    assert np.array_equal(np.stack([layers[0][1][i] for i in range(3)], axis=-1), rgb)
    assert np.array_equal(layers[1][1][-1], alpha) and np.array_equal(back, composite)
    with pytest.raises(ValueError, match="layer 'raw'"):
        write_psd([PsdLayer("raw", rgb[:10])], composite)


def test_a_page_psd_has_raw_clean_and_text_layers(tmp_path: Path) -> None:
    paths = ChapterPaths(
        "S", "C", tmp_path / "lib" / "S" / "C", tmp_path / "w", tmp_path / "o", tmp_path / "f"
    )
    paths.raw_dir.mkdir(parents=True)
    buf = io.BytesIO()
    Image.new("RGB", (200, 120), (230, 220, 210)).save(buf, format="PNG")
    (paths.raw_dir / "001.png").write_bytes(buf.getvalue())
    ingest = IngestArtifact(
        series="S",
        chapter="C",
        strip_width=200,
        strip_height=120,
        files=[
            SourceFile(
                index=0,
                name="001.png",
                sha256=hashlib.sha256(buf.getvalue()).hexdigest(),
                width=200,
                height=120,
                y0=0,
                y1=120,
            )
        ],
    )
    item = LayoutItem(
        region_id="r0001",
        box=BBox(x0=20, y0=30, x1=180, y1=90),
        lines=["HELLO"],
        font_role="dialogue",
        font="ComicNeue-Bold.ttf",
        size_px=28,
        color=(10, 10, 10),
        stroke_px=0,
        align="center",
    )
    layers, composite = read_psd(page_psd(paths, ingest, 0, [item]))
    assert [name for name, _ in layers] == ["raw", "clean", "text"]
    raw = np.stack([layers[0][1][i] for i in range(3)], axis=-1)
    assert raw.shape == (120, 200, 3) and abs(int(raw[5, 5, 0]) - 230) <= 2
    text_alpha = layers[2][1][-1]
    assert (
        text_alpha[:25].max() == 0 and text_alpha[30:90, 20:180].max() == 255
    )  # lettering, nothing above it
    assert np.array_equal(composite, render_page(paths, ingest, 0, [item]))  # the studio preview's page


def test_psd_export_writes_one_file_per_page(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from typer.testing import CliRunner

    import omniscan.interchange.cli as interchange_cli
    from omniscan.core.config import Config, PathsConfig
    from omniscan.core.paths import SeriesPaths
    from omniscan.core.schemas import LayoutArtifact

    cfg = Config(
        paths=PathsConfig(
            library_root=tmp_path / "lib", work_root=tmp_path / "work", output_root=tmp_path / "out"
        )
    )
    monkeypatch.setattr(interchange_cli, "get_config", lambda: cfg)
    paths = SeriesPaths.from_config(cfg, "S").chapter("C")
    paths.raw_dir.mkdir(parents=True)
    files = []
    for index, colour in enumerate(((200, 10, 10), (10, 10, 200))):
        buf = io.BytesIO()
        Image.new("RGB", (100, 80), colour).save(buf, format="PNG")
        (paths.raw_dir / f"{index + 1:03d}.png").write_bytes(buf.getvalue())
        digest = hashlib.sha256(buf.getvalue()).hexdigest()
        files.append(
            SourceFile(
                index=index,
                name=f"{index + 1:03d}.png",
                sha256=digest,
                width=100,
                height=80,
                y0=80 * index,
                y1=80 * (index + 1),
            )
        )
    IngestArtifact(series="S", chapter="C", strip_width=100, strip_height=160, files=files).save(
        paths.artifact("ingest.json")
    )
    runner = CliRunner()
    result = runner.invoke(interchange_cli.psd_app, ["export", "S", "C"])
    target = tmp_path / "out" / "S" / "_psd" / "C"
    assert result.exit_code == 0 and f"psd: 2 page(s) -> {target}" in result.output
    assert "no layout.json yet" in result.output
    layers, composite = read_psd((target / "002.psd").read_bytes())
    assert composite.shape == (80, 100, 3) and abs(int(composite[40, 50, 2]) - 200) <= 2  # page 2 is blue
    assert layers[2][1][-1].max() == 0  # nothing lettered
    LayoutArtifact(items=[]).save(paths.artifact("layout.json"))
    one = runner.invoke(
        interchange_cli.psd_app, ["export", "S", "C", "--page", "0", "--out", str(tmp_path / "x")]
    )
    assert one.exit_code == 0 and sorted(p.name for p in (tmp_path / "x").iterdir()) == ["001.psd"]
    unknown = runner.invoke(interchange_cli.psd_app, ["export", "S", "C", "--page", "7"])
    assert unknown.exit_code == 2 and "no page 7" in unknown.output
