# Reader format (version 1)

What OmniScan writes next to its finished chapters so any reader app can open them without knowing the pipeline:
OmniScan's own reading mode today, and the planned phone/tablet/desktop reader (`docs/ROADMAP.md`, X6). The
schemas are `ReaderChapter` and `ReaderSeries` in `src/omniscan/core/schemas.py`; the writer is
`omniscan.export.reader_index`, called by the `export` stage.

```
<output_root>/
  <Series>/
    omniscan-series.json          # the chapters, in reading order
    <Chapter>/
      0001.jpg 0002.jpg ...       # the translated pages (JPEG slices of the chapter strip)
      omniscan-chapter.json       # the pages, with their pixel sizes
```

## omniscan-chapter.json

```json
{
  "schema_version": 1,
  "format": "omniscan-chapter",
  "series": "Solo Leveling",
  "chapter": "Chapter 12",
  "number": 12.0,
  "language": "en",
  "reading": "vertical",
  "pages": [{"file": "0001.jpg", "width": 800, "height": 2400}],
  "updated_at": "2026-09-27T03:30:00Z"
}
```

- `number` is parsed from the folder name and is `null` when the name has none.
- `reading` is `vertical` (webtoon: one continuous scroll; pages touch with no gap), or `rtl` / `ltr` for paged
  manga. OmniScan writes `vertical` today.
- The page sizes let a reader lay out the whole scroll before any image is loaded.

## omniscan-series.json

```json
{
  "schema_version": 1,
  "format": "omniscan-series",
  "series": "Solo Leveling",
  "chapters": [{"folder": "Chapter 12", "number": 12.0, "pages": 31, "updated_at": "2026-09-27T03:30:00Z"}],
  "updated_at": "2026-09-27T03:30:00Z"
}
```

Rebuilt after every export from the chapter folders that hold a readable `omniscan-chapter.json`, sorted like the
library (chapter number, then the folder name). Folders starting with `_` (e.g. `_filtered`) are never listed.

## Rules for readers
- Treat unknown fields as optional and ignore them; a breaking change bumps `schema_version`.
- Never write into these files; keep reading progress in the reader's own storage.
- Other readers (Mihon/Tachiyomi, Komga, Kavita) read the CBZ files `omniscan pack` builds (with `ComicInfo.xml`).
