# Fonts

Open-source fonts used by the typesetter's lettering presets, the tests and the synthetic Korean pages.
Each family's licence text sits next to it: `OFL-<Family>.txt` (SIL Open Font License 1.1) or
`LICENSE-<Family>.txt` (Apache License 2.0). Source: <https://github.com/google/fonts> (`ofl/`, `apache/`),
fetched 2026-09-19 (Bangers, Comic Neue, Gaegu, Nanum Gothic, Patrick Hand) and 2026-09-25 (the rest).

## Lettering presets (`typeset/fonts.py`, `[typeset] style`)

| File | Preset use |
|---|---|
| `Mali-SemiBold.ttf` | **webtoon** dialogue and narration: the clean, rounded mixed-case look of official English webtoons |
| `Mali-Bold.ttf` | webtoon shouts and free-standing text |
| `Mali-MediumItalic.ttf` | webtoon thoughts |
| `Kalam-Bold.ttf` | **manga** dialogue and free text, in capitals: the slanted hand-lettered look official English manga get from CC Wild Words |
| `Kalam-Regular.ttf` | manga thoughts and narration |
| `Bangers-Regular.ttf` | manga shouts |
| `LuckiestGuy-Regular.ttf` | sound effects redrawn over **heavy** originals (Apache 2.0) |
| `Knewave-Regular.ttf` | sound effects over **bold** brush-stroke originals, and the default effect face |
| `PermanentMarker-Regular.ttf` | sound effects over **light**, thin-stroked originals (Apache 2.0) |

A sound effect's face is picked from the stroke weight measured on the original (`ocr/sfx.py`), so a crash
drawn in fat brush strokes is redrawn fat and a whisper of a rustle stays thin.

## Other files

| File | Use |
|---|---|
| `NanumGothic-Regular.ttf`, `NanumGothic-Bold.ttf` | Korean text for synthetic test pages (Hangul + Latin) |
| `Gaegu-Regular.ttf` | Korean handwriting style for synthetic pages (harder OCR case) |
| `ComicNeue-Regular.ttf`, `ComicNeue-Bold.ttf` | the lettering defaults before the presets; kept for tests and as an option |
| `PatrickHand-Regular.ttf` | handwritten alternative for thoughts |

## Your own lettering fonts

Commercial lettering fonts (CC Wild Words, Anime Ace, Blambot faces, ...) are supplied by the user and never
committed (question E2 in `docs/OPEN_QUESTIONS.md`). Point a role at one in `~/.config/omniscan/config.toml`
or a series' `series.toml`, by file name (looked up here, or in `$OMNISCAN_FONTS_DIR`) or absolute path:

```toml
[typeset]
font_dialogue = "C:/Users/me/Fonts/CCWildWords-Regular.ttf"
font_shout = "C:/Users/me/Fonts/CCWildWords-Bold.ttf"
```
