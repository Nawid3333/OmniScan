"""Probe how candidate Ollama models translate a short manhwa-style Korean script (see docs/benchmarks/translation-probe.md).

Usage: uv run python scripts/probe_translation.py <model>      # chat_json style, thinking off, tolerant JSON parse
       uv run python scripts/probe_translation.py tg           # translategemma:12b, one request per line
"""

import json
import re
import sys
import time

import httpx

BASE = "http://localhost:11434"

REGIONS = [
    {"id": "r0001", "kind": "bubble_text", "text": "성진 님, 괜찮으세요?"},
    {"id": "r0002", "kind": "bubble_text", "text": "...아무것도 아니야. 신경 쓰지 마."},
    {"id": "r0003", "kind": "bubble_text", "text": "헌터 협회에서 온 사람이라고?!"},
    {"id": "r0004", "kind": "sfx", "text": "쿵!"},
    {"id": "r0005", "kind": "bubble_text", "text": "그 녀석은 S급 게이트 안에서\n살아남은 유일한 생존자야."},
    {"id": "r0006", "kind": "bubble_text", "text": "형, 이번엔 내가 지킬게."},
    {"id": "r0007", "kind": "bubble_text", "text": "죽고 싶지 않으면 당장 물러서라."},
    {"id": "r0008", "kind": "free_text", "text": "이건 그냥\n꿈이겠지...?"},
    {"id": "r0009", "kind": "bubble_text", "text": "성진이가 게이트에 들어간 지 벌써 3일이 지났어요."},
    {"id": "r0010", "kind": "bubble_text", "text": "선배님, 저... 헌터 협회 소속이 아닙니다."},
]
GLOSSARY = [
    {"source": "성진", "target": "Seong-jin", "type": "person"},
    {"source": "헌터 협회", "target": "Hunter Association", "type": "org"},
    {"source": "게이트", "target": "Gate", "type": "term"},
]

SYSTEM = (
    "You are the translator for an official English release of a Korean manhwa. Translate every numbered "
    "region from Korean into natural, idiomatic English suited to comic lettering: concise, in-voice, no "
    "translator notes. Keep Korean honorific suffixes/titles romanized when they address a person "
    "(-nim, -ssi, hyung, noona, sunbae) unless they read badly in English. For a region of kind 'sfx' give a "
    "short English onomatopoeia. A glossary entry is binding: whenever its source term appears (with or without "
    "a particle such as 이/가/은/는/을/를/의), its target must appear in your English exactly as written. "
    'Answer with JSON only: {"translations":[{"id":"r0001","text":"..."}]} with exactly one entry per input id.'
)

SCHEMA = {
    "type": "object",
    "properties": {
        "translations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "text": {"type": "string"}},
                "required": ["id", "text"],
            },
        }
    },
    "required": ["translations"],
}


def chat(model, messages, fmt=None, options=None, timeout=600):
    body = {"model": model, "messages": messages, "stream": False, "think": False}
    if fmt:
        body["format"] = fmt
    if options:
        body["options"] = options
    t = time.time()
    r = httpx.post(f"{BASE}/api/chat", json=body, timeout=timeout)
    r.raise_for_status()
    d = r.json()
    return d["message"]["content"], time.time() - t, d.get("prompt_eval_count"), d.get("eval_count")


def json_style(model):
    user = (
        "Glossary:\n"
        + "\n".join(f"- {g['source']} -> {g['target']} ({g['type']})" for g in GLOSSARY)
        + "\n\nRegions (reading order):\n"
        + json.dumps(REGIONS, ensure_ascii=False, indent=1)
    )
    content, secs, pin, pout = chat(
        model,
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        fmt=SCHEMA,
        options={"temperature": 0.3},
    )
    print(f"\n=== {model} [chat_json] {secs:.1f}s in={pin} out={pout}")
    try:
        m = re.search(r"\{.*\}", content, re.S)
        if m is None:
            raise ValueError("no JSON object in reply")
        out = json.loads(m.group(0))["translations"]
    except (ValueError, KeyError) as e:
        print("  !! could not parse:", e, content[:300])
        return
    got = {t["id"]: t["text"] for t in out}
    for r in REGIONS:
        print(f"  {r['id']} {r['text']!r:48} -> {got.get(r['id'])!r}")
    missing = [r["id"] for r in REGIONS if r["id"] not in got]
    if missing:
        print("  !! missing ids:", missing)


def tg_style(model):
    print(f"\n=== {model} [translategemma per-line]")
    t0 = time.time()
    for r in REGIONS:
        text = r["text"].replace("\n", " ")
        prompt = (
            "You are a professional Korean (ko) to English (en) translator. Your goal is to accurately convey "
            "the meaning and nuances of the original Korean text while adhering to English grammar, vocabulary, "
            "and cultural sensitivities.\nProduce only the English translation, without any additional "
            "explanations or commentary. Please translate the following Korean text into English:\n\n\n"
            + text
        )
        content, secs, _pin, _pout = chat(
            model, [{"role": "user", "content": prompt}], options={"temperature": 0.3}
        )
        print(f"  {r['id']} {r['text']!r:48} -> {content.strip()!r}  ({secs:.1f}s)")
    print(f"  total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "tg":
        tg_style("translategemma:12b")
    else:
        json_style(which)
