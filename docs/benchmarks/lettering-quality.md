# Lettering quality: cleaning, layout and sound effects (2026-09-25)

Owner request: output that looks like an official release — no cleaning artifacts, lettering that fits the
balloon and matches the style, sound effects redrawn like official translations do. This note records what
was changed and the evidence. Measured on CPU in a cloud session (no RX 9070 XT, no Hugging Face access):
synthetic Korean pages (`tests/fixtures/korean_pages.py`) whose background is a painted panel crossed by ink
lines and hatching (`scripts/lettering_demo.py`), the regions as the default whole-region OCR engine hands
them on, and the **real LaMa model** (`big-lama.pt`, sha256 as in the config) on CPU. **Not yet checked on
real chapters** — see "Next" below.

Reproduce / look at it: `uv run python scripts/lettering_demo.py` → `data/lettering_demo/page-<seed>.png`
(raw | lettered) and `sfx.png`.

## What professionals do (research summary)

- **Balloons:** letterers break dialogue so line lengths follow the balloon — short lines at top and bottom,
  the widest in the middle (the "lozenge"/diamond), keep neighbouring lines similar, break at phrase and
  clause boundaries, and keep the dialogue size consistent across a chapter.
- **Fonts:** official English manga (Viz, Yen Press, Kodansha) use CC Wild Words / Anime Ace style hand
  lettering in capitals; official webtoons use clean, rounded mixed-case faces. Both are commercial.
- **Cleaning:** redrawers remove only the lettering and rebuild the art under it; tools like
  manga-image-translator and BallonsTranslator refine the detector's box into a pixel mask of the text
  (connected components + dilation by a fraction of the text size) before LaMa.
- **Sound effects:** official webtoon releases erase the original effect and redraw an English one in a
  matching style (colour, outline, tilt, weight — fat brush letters for impacts, thin ones for soft
  sounds); many manga releases instead keep the art and set a small translation next to it.

## Cleaning

| background (6 pages each) | LaMa mask px, box masks | LaMa mask px, glyph masks | change | text pixels covered |
|---|---|---|---|---|
| drawn art (ink lines, hatching) | 324 627 | 241 156 | **-26 %** | 100 % |
| gradient + noise | 330 163 | 194 871 | **-41 %** | 100 % |
| random 8 px blocks (adversarial) | 328 941 | 287 683 | -13 % | 100 % |

Coverage is measured against the fixture's pixel-exact text mask on every region of every page
(`test_pipeline_masks_cover_every_text_pixel`); SFX lettered on a flat panel are now cleaned by a flat fill
of just the glyphs instead of LaMa. Artifacts found by looking at the LaMa output and fixed:

- **Tile leak:** an effect wider than one LaMa window was tiled, each tile masking only its own part; LaMa
  copied the rest of the effect's orange back into the hole. Every window now masks all lettering it reaches
  and works on a copy with the finished regions written back — 0 remnant pixels afterwards.
- **Outline ghost:** black letters with a 10 px white outline, crossed by black art lines: the outline was
  measured around the art lines too (rings read as background) and survived as a white silhouette. Measured
  around the letters alone it is removed (`test_a_thick_outline_is_removed_even_with_ink_lines...`).
- **Cut art lines:** the flat fill around glyphs accepted a band that was flat for 90 % of its pixels, so ink
  lines crossing a caption were cut by a flat patch; glyph fills now need 98 %.

## Layout

Same balloon, same font (Mali SemiBold), old greedy fit into the inscribed rectangle vs the balloon-shaped fit
with 12 % breathing room (before the chapter-wide size limit):

| balloon | old (greedy, inscribed rectangle) | new (balloon-shaped, phrase-aware) |
|---|---|---|
| 360x230 | 26 px: "Are you okay? The / dungeon just / opened and the / monsters are / coming out!" | 24 px: "Are you okay? / The dungeon just / opened and the / monsters are / coming out!" |
| 300x300 | 29 px: "I can't / believe you / actually / came all the / way here just / to see me." | 30 px: "I can't / believe you / actually came / all the way / here just to / see me." |
| 420x170 | 23 px: "Hunter Association / emergency / announcement: / evacuate immediately." | 23 px: "Hunter Association / emergency announcement: / evacuate immediately." |

The size stays about the same (-2..+1 px); what changes is the shape (lines following the balloon, no
ragged or dangling lines) and the breaks (after the sentence and the clause, never after "The").

Cost: ~9 ms per balloon on CPU (60 random balloons in 0.55 s), ≤ 1 200 width measurements per balloon
(`test_typeset_fit_speed.py`). On a page, dialogue sizes are held within 1.15 x the chapter's median balloon
size, so a short "Huh?" no longer jumps to 48 px.

## Sound effects

Stroke weight (stroke width / letter height, `ocr/sfx.py`) on rendered Hangul effects, 110 and 190 px,
level / 15° / -25°:

| face | "쾅!" | "두근두근" | class |
|---|---|---|---|
| Nanum Gothic Bold | 0.089-0.098 | 0.096-0.108 | bold (≥ 0.08) |
| Nanum Gothic Regular | 0.057-0.065 | 0.057-0.070 | light |
| Gaegu (handwriting) | 0.069-0.072 | 0.074-0.084 | light / bold border |

The class also holds with a white outline and a coloured fill (`test_the_weight_class_survives_tilt_and_outline`).
Tilt: two or more syllables are measured within ~2° (-18° → -17..-20°, 25° → 23..26°); a lone syllable and a
vertical column stay level. Over drawn art the measurement strips differently coloured art and strokes much
thinner than the letters' own before measuring.

## Known limits

- Validated on synthetic pages only; the real Solo Leveling chapters need a GPU run and a look (the chapter's
  typical size, the lexicon's hit rate on real effects, LaMa on real screentone).
- The SFX lexicon is hand-written (ko ~200 words, ja/zh smaller); unknown effects stay free text unless
  `sfx.size_ratio` is enabled (off: big signs would pass).
- A lone-syllable effect is lettered level even when the original is tilted (one boxy glyph has no baseline).
- Effects are fitted into the original's box; a long English word over a one-syllable original comes out
  smaller than the original (the translation prompt asks for one short onomatopoeia).
- Bubble shapes are approximated by the ellipse in the bubble box; rectangular captions get an ellipse too
  unless the original text already filled the box.
