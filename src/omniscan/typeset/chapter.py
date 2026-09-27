"""A chapter's lettering from its artifacts: what the `typeset` stage writes and the studio previews live."""

from __future__ import annotations

from dataclasses import dataclass

from omniscan.core.config import Config
from omniscan.core.paths import ChapterPaths
from omniscan.core.schemas import RGB, FinalArtifact, InpaintArtifact, LayoutItem, RegionsArtifact
from omniscan.edits.store import load_edits
from omniscan.typeset.overrides import apply_layout_edits
from omniscan.typeset.plan import plan_layout


@dataclass(frozen=True, slots=True)
class ChapterLayout:
    """The typesetter's own items and the items with the chapter's hand-set lettering applied."""

    auto: list[LayoutItem]
    items: list[LayoutItem]
    orphans: int  # hand-set letterings whose region no longer exists


def chapter_layout(paths: ChapterPaths, cfg: Config) -> ChapterLayout:
    """Letter every region with an English line (ocr.json + final.json + inpaint(_lama).json), then apply
    edits.json's hand-set lettering. FileNotFoundError names the first missing artifact."""
    for name, stage in (("ocr.json", "ocr"), ("final.json", "judge"), ("inpaint.json", "inpaint")):
        if not paths.artifact(name).is_file():
            raise FileNotFoundError(f"{name} missing — run the {stage} stage first")
    regions = RegionsArtifact.load(paths.artifact("ocr.json")).regions
    texts = {line.region_id: line.text for line in FinalArtifact.load(paths.artifact("final.json")).lines}
    inpaint = InpaintArtifact.load(paths.artifact("inpaint.json")).items
    fills: dict[str, RGB] = {item.region_id: item.fill for item in inpaint if item.fill is not None}
    erased = {item.region_id for item in inpaint if item.method == "flat"}
    lama_path = paths.artifact("inpaint_lama.json")
    if lama_path.is_file():  # which sound effects LaMa erased
        erased |= {item.region_id for item in InpaintArtifact.load(lama_path).items}
    auto = plan_layout(regions, texts, fills, cfg.typeset, sfx=cfg.sfx, erased=erased)
    items, orphans = apply_layout_edits(auto, load_edits(paths), regions, texts, cfg.typeset)
    return ChapterLayout(auto=auto, items=items, orphans=orphans)
