# Translation probe (2026-09-18)

Small, hand-written evidence run to choose the prompt/parse strategy before the translation runner is built (card B29).
**Not a quality benchmark**: 10 original Korean lines (honorifics, name + particle, glossary terms, an SFX, a line-broken
bubble), one run each, temperature 0.3, reproducible with `scripts/probe_translation.py`. The lines are written for this
probe and are not from any real work. Real quality evaluation needs real OCR output and the calibration benchmark (C12).

Glossary used: 성진 → Seong-jin, 헌터 협회 → Hunter Association, 게이트 → Gate.

## Results

| Model (style) | Where | Time | Glossary respected | JSON handling | Notes |
|---|---|---|---|---|---|
| `translategemma:12b` (per line, its own template) | local GPU | 0.3 s/line after a ~10 s load, 12 s total | **No** (`Seongjin`, `Hunter's Association`, `gate`) | n/a (plain text) | Sometimes wraps the reply in straight quotes; `Brother` for 형, `Senior` for 선배님 |
| `translategemma:12b` with **pre-substituted** targets (`Seong-jin 님`, `Gate에`) | local GPU | same | **Yes** | n/a | Keeps the Latin term and reads the Korean particle around it |
| `gemma4:12b` (chat_json) | local GPU | 4 s | Yes | valid JSON when **`think: false`** | Without `think: false`: **353 s, 7518 output tokens, empty `content`** (all budget went into thinking) |
| `gemma4:31b-cloud` (chat_json) | cloud via local daemon | 1.4 s | Yes | ignores `format`; wraps JSON in a ```` ```json ```` fence | Keeps the source line break in two lines |
| `glm-5.3-flash:cloud` (chat_json) | cloud via local daemon | 5 s | Yes | ignores `format` **and** `think: false`; sometimes prose reasoning before the JSON | Good quality |
| `kimi-k3:cloud` (chat_json) | cloud via local daemon | 2.9 s | Yes | valid JSON after tolerant extraction | Good quality |

## Findings that shape the runner

1. **`think` must be controllable per profile.** Thinking models spend the whole budget reasoning if it is left on;
   `translategemma` rejects `think: true` with HTTP 400 (`does not support thinking`) but accepts `false`. So the
   client sends `think` only when the profile sets it.
2. **Cloud models ignore Ollama's `format` JSON schema** (fenced JSON, prose first). Local models honour it. The
   parser must be tolerant: fenced blocks, prose before/after, first JSON object that has a `translations` list.
3. **Glossary compliance is prompt-enforceable for chat models but not for translategemma.** For translategemma the
   locked source terms are replaced by their targets before translating (keeping the Korean particle), then verified.
4. **Source line breaks leak into the translation** (`This is just\na dream, right...?`). Regions are sent with
   whitespace collapsed to single spaces and the prompt asks for one continuous line; the typesetter re-wraps.
5. Ambiguity that only context can resolve (`헌터 협회에서 온 사람이라고?!` → "You're saying they're from…" vs "He's
   from…") shows up between models, which is exactly what the judge and story memory (C5/C12) are for.
6. Cloud models named `*-cloud` are reachable through the **local** daemon at `localhost:11434` (used above), so no API
   key is needed for them today.
