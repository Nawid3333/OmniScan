"""Tests for scripts/model_watch.py (card W1): fake Fetch, tmp files, no network."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from omniscan.models.catalog import ModelEntry

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "model_watch.py"
_spec = importlib.util.spec_from_file_location("model_watch", _SCRIPT)
assert _spec is not None and _spec.loader is not None
script = importlib.util.module_from_spec(_spec)
sys.modules["model_watch"] = script  # dataclasses(slots=True) resolves its module via sys.modules
_spec.loader.exec_module(script)

REPO_ROOT = Path(__file__).resolve().parents[2]
A = "a" * 40
B = "b" * 40
GENERATED = "2026-09-23T06:00:00+00:00"


class FakeFetch:
    """Canned Hugging Face API: per-repo model responses, per (author, search term) listings."""

    def __init__(
        self,
        *,
        model: dict[str, tuple[int, Any]] | None = None,
        listing: dict[tuple[str, str], tuple[int, Any]] | None = None,
    ) -> None:
        self.model = model or {}
        self.listing = listing or {}
        self.calls: list[tuple[str, Mapping[str, str] | None]] = []

    def __call__(self, url: str, params: Mapping[str, str] | None = None) -> tuple[int, Any]:
        self.calls.append((url, params))
        if url == script.API_BASE:
            assert params is not None
            key = (params["author"], params["search"])
            return self.listing.get(key, (200, []))
        repo = url.removeprefix(script.API_BASE + "/")
        return self.model.get(repo, (500, None))


def entry(eid: str, *, fmt: str = "hf", repo: str | None = None, revision: str | None = None) -> ModelEntry:
    kwargs: dict[str, Any] = {}
    if fmt in ("hf", "zip"):
        kwargs["upstream_repo"] = repo
        kwargs["upstream_revision"] = revision
    return ModelEntry(
        id=eid,
        name=eid,
        kind="ocr",
        format=fmt,  # type: ignore[arg-type]
        size_mb=10,
        license="Apache-2.0",
        description="test entry",
        **kwargs,
    )


def catalog_toml(*specs: tuple[str, str, str]) -> str:
    """A minimal [[model]] catalog: (id, upstream_repo, upstream_revision) per entry."""
    blocks = []
    for eid, repo, revision in specs:
        blocks.append(
            f'[[model]]\nid = "{eid}"\nname = "{eid}"\nkind = "ocr"\nformat = "hf"\nsize_mb = 10\n'
            f'license = "Apache-2.0"\ndescription = "test entry"\n'
            f'upstream_repo = "{repo}"\nupstream_revision = "{revision}"\n'
        )
    return "\n".join(blocks)


def write_files(
    tmp_path: Path,
    catalog: str,
    config: str = 'ignore = []\n\n[orgs."orga"]\nsearch = ["t"]\npatterns = ["pp-*"]\n',
) -> tuple[Path, Path]:
    catalog_path, config_path = tmp_path / "models.toml", tmp_path / "model_watch.toml"
    catalog_path.write_text(catalog, encoding="utf-8")
    config_path.write_text(config, encoding="utf-8")
    return catalog_path, config_path


# --- config --------------------------------------------------------------------------------------------


def test_load_watch_config_shipped_file() -> None:
    cfg = script.load_watch_config(REPO_ROOT / "config" / "model_watch.toml")
    assert set(cfg.orgs) == {"PaddlePaddle", "kha-white", "jzhang533", "ogkalu"}
    assert cfg.orgs["PaddlePaddle"].search == ("OCR",)
    assert cfg.orgs["PaddlePaddle"].patterns == ("PP-OCRv*", "*PP-OCRv*_rec*", "PaddleOCR-VL*")
    assert cfg.orgs["kha-white"].search == ("manga-ocr",)
    assert cfg.orgs["kha-white"].patterns == ("manga-ocr*",)
    assert cfg.orgs["jzhang533"].search == ("manga-ocr",)
    assert cfg.orgs["jzhang533"].patterns == ("manga-ocr*",)
    assert cfg.orgs["ogkalu"].search == ("comic",)
    assert cfg.orgs["ogkalu"].patterns == ("comic-text*",)
    assert cfg.ignore == frozenset()


@pytest.mark.parametrize(
    ("toml_text", "field"),
    [
        ('[orgs."org"]\nsearch = ["x"]\n', "patterns"),  # missing patterns
        ('[orgs."org"]\npatterns = ["x"]\n', "search"),  # missing search
        ('[orgs."org"]\nsearch = "x"\npatterns = ["x"]\n', "search"),  # wrong type
        ('[orgs."org"]\nsearch = ["x"]\npatterns = [1]\n', "patterns"),  # wrong type
        ('[orgs.""]\nsearch = ["x"]\npatterns = ["x"]\n', "orgs"),  # empty org name
        ('[orgs."a/b"]\nsearch = ["x"]\npatterns = ["x"]\n', "orgs"),  # org id with a slash
        ("ignore = 5\n[orgs.a]\nsearch = []\npatterns = []\n", "ignore"),  # wrong type
        ("ignore = []\n", "orgs"),  # no orgs at all
        ("not toml [[[\n", "TOML"),  # not TOML
    ],
)
def test_load_watch_config_rejects_bad_input(tmp_path: Path, toml_text: str, field: str) -> None:
    path = tmp_path / "watch.toml"
    path.write_text(toml_text, encoding="utf-8")
    with pytest.raises(ValueError, match=field):
        script.load_watch_config(path)


# --- check_entries -------------------------------------------------------------------------------------


def test_check_entries_no_change_when_head_matches() -> None:
    fetch = FakeFetch(
        model={"org/same": (200, {"sha": A, "lastModified": "2026-09-01T00:00:00.000Z", "downloads": 7})}
    )
    assert script.check_entries([entry("e1", repo="org/same", revision=A)], fetch) == []
    assert len(fetch.calls) == 1


def test_check_entries_reports_updated() -> None:
    fetch = FakeFetch(
        model={"org/moved": (200, {"sha": B, "lastModified": "2026-09-02T00:00:00.000Z", "downloads": 11})}
    )
    changes = script.check_entries([entry("e1", repo="org/moved", revision=A)], fetch)
    assert len(changes) == 1
    change = changes[0]
    assert change.kind == "updated" and change.repo == "org/moved" and change.catalog_id == "e1"
    assert change.old_revision == A and change.new_revision == B
    assert change.last_modified == "2026-09-02T00:00:00.000Z" and change.downloads == 11
    assert change.note == "e1: upstream has a newer commit"


def test_check_entries_reports_missing_on_404() -> None:
    fetch = FakeFetch(model={"org/gone": (404, None)})
    changes = script.check_entries([entry("e1", repo="org/gone", revision=A)], fetch)
    assert len(changes) == 1
    change = changes[0]
    assert change.kind == "missing" and change.repo == "org/gone" and change.catalog_id == "e1"
    assert change.old_revision == A and change.new_revision is None
    assert change.note == "repo not found (renamed or removed?)"


def test_check_entries_skips_server_error_with_warning(capsys: pytest.CaptureFixture[str]) -> None:
    fetch = FakeFetch(model={"org/broken": (500, None)})
    assert script.check_entries([entry("e1", repo="org/broken", revision=A)], fetch) == []
    assert "warning" in capsys.readouterr().err


def test_check_entries_skips_transport_failure_with_warning(capsys: pytest.CaptureFixture[str]) -> None:
    def failing(url: str, params: Mapping[str, str] | None = None) -> tuple[int, Any]:
        raise script.FetchError("ConnectError for url after 2 retries")

    assert script.check_entries([entry("e1", repo="org/broken", revision=A)], failing) == []
    assert "warning" in capsys.readouterr().err


def test_check_entries_only_hf_and_zip_entries_are_watched() -> None:
    fetch = FakeFetch(model={"org/zip": (200, {"sha": B, "downloads": 3})})
    entries = [
        entry("zipdet", fmt="zip", repo="org/zip", revision=A),
        entry("lama", fmt="file"),  # GitHub upstream: never fetched
        entry("local-llm", fmt="ollama"),  # Ollama: out of scope
        entry("cloud-llm", fmt="cloud"),  # Ollama Cloud: out of scope
    ]
    changes = script.check_entries(entries, fetch)
    assert [(c.kind, c.repo) for c in changes] == [("updated", "org/zip")]
    assert len(fetch.calls) == 1


def test_check_entries_one_request_per_repo_two_changes() -> None:
    fetch = FakeFetch(model={"org/shared": (200, {"sha": B, "downloads": 5})})
    changes = script.check_entries(
        [entry("e1", repo="org/shared", revision=A), entry("e2", repo="org/shared", revision="c" * 40)], fetch
    )
    assert len(fetch.calls) == 1
    assert sorted(c.catalog_id for c in changes) == ["e1", "e2"]
    assert all(c.kind == "updated" and c.new_revision == B for c in changes)


# --- find_new ------------------------------------------------------------------------------------------


def test_find_new_filters_patterns_catalog_and_ignore() -> None:
    cfg = script.WatchConfig(
        orgs={
            "orga": script.OrgWatch(search=("term-a",), patterns=("pp-*", "orga/*")),
            "orgb": script.OrgWatch(search=("t1", "t2"), patterns=("manga*",)),
        },
        ignore=frozenset({"orga/pp-ignored"}),
    )
    fetch = FakeFetch(
        listing={
            ("orga", "term-a"): (
                200,
                [
                    {"id": "orga/pp-new", "lastModified": "2026-09-02T00:00:00.000Z", "downloads": 5},
                    {"id": "orga/pp-known", "lastModified": "2026-09-01T00:00:00.000Z", "downloads": 6},
                    {"id": "orga/pp-ignored", "lastModified": "2026-09-02T00:00:00.000Z", "downloads": 7},
                    {"id": "orga/no-match", "lastModified": "2026-09-06T00:00:00.000Z", "downloads": 8},
                    {"id": "orga/PP-CAPS", "lastModified": "2026-09-03T00:00:00.000Z", "downloads": 9},
                ],
            ),
            ("orgb", "t1"): (
                200,
                [{"id": "orgb/manga-ocr-x", "lastModified": "2026-09-04T00:00:00.000Z", "downloads": 10}],
            ),
            # same repo as t1: deduped across terms
            ("orgb", "t2"): (
                200,
                [{"id": "orgb/manga-ocr-x", "lastModified": "2026-09-04T00:00:00.000Z", "downloads": 10}],
            ),
        },
    )
    entries = [entry("known", repo="orga/pp-known", revision=A)]
    changes = script.find_new(entries, cfg, fetch)
    assert [c.repo for c in changes] == ["orgb/manga-ocr-x", "orga/PP-CAPS", "orga/pp-new"]  # newest first
    for change in changes:
        assert change.kind == "new" and change.catalog_id is None and change.note == "not in the catalog"
    assert len(fetch.calls) == 3  # one per (org, term)


def test_find_new_failing_listing_is_a_warning_only(capsys: pytest.CaptureFixture[str]) -> None:
    cfg = script.WatchConfig(
        orgs={"orga": script.OrgWatch(search=("t",), patterns=("pp-*",))}, ignore=frozenset()
    )
    fetch = FakeFetch(listing={("orga", "t"): (500, None)})
    assert script.find_new([], cfg, fetch) == []
    assert "warning" in capsys.readouterr().err


def test_find_new_transport_failure_is_a_warning_only(capsys: pytest.CaptureFixture[str]) -> None:
    def failing(url: str, params: Mapping[str, str] | None = None) -> tuple[int, Any]:
        raise script.FetchError("ReadError for url after 2 retries")

    cfg = script.WatchConfig(
        orgs={"orga": script.OrgWatch(search=("t",), patterns=("pp-*",))}, ignore=frozenset()
    )
    assert script.find_new([], cfg, failing) == []
    assert "warning" in capsys.readouterr().err


# --- render_report -------------------------------------------------------------------------------------


def make_change(kind: str, **overrides: Any) -> Any:  # Change is loaded dynamically; not a static type
    base: dict[str, Any] = {
        "repo": "org/repo",
        "catalog_id": "catalog-id",
        "old_revision": A,
        "new_revision": B,
        "last_modified": "2026-09-01T00:00:00.000Z",
        "downloads": 10,
        "note": "note",
    }
    return script.Change(kind, **{**base, **overrides})  # type: ignore[arg-type]


TABLE = [
    "| repo | catalog id | old → new | last modified | downloads | note |",
    "|---|---|---|---|---|---|",
]

GOLDEN = (
    "\n".join(
        [
            "# Model watch report",
            "",
            f"Generated: {GENERATED}",
            "",
            "## Updated upstream",
            "",
            *TABLE,
            "| `PaddlePaddle/PP-OCRv6_tiny_rec_safetensors` | `ocr-rec-ppocrv6-tiny` | `6f2d2d51` → `024cad6a` "
            "| 2026-09-20T10:00:00.000Z | 1234 | ocr-rec-ppocrv6-tiny: upstream has a newer commit |",
            "",
            "## New upstream models",
            "",
            *TABLE,
            "| `kha-white/manga-ocr-large` | — | — | 2026-09-21T00:00:00.000Z | 456 | not in the catalog |",
            "",
            "## Missing upstream",
            "",
            *TABLE,
            "| `jzhang533/manga-ocr-base-2025` | `ocr-rec-manga-ocr-2025` | `1e64d5be` → — | — | — "
            "| repo not found (renamed or removed?) |",
            "",
            "## What to do",
            "",
            "- run scripts/qualify_ocr.py --only <candidate> on a GPU machine",
            "- update config/models.toml only if the qualification improves",
            "- add the repo to the ignore list in config/model_watch.toml to silence it",
        ]
    )
    + "\n"
)


def make_golden_changes() -> list[Any]:
    return [
        make_change(
            "updated",
            repo="PaddlePaddle/PP-OCRv6_tiny_rec_safetensors",
            catalog_id="ocr-rec-ppocrv6-tiny",
            old_revision="6f2d2d51b4b4226d7a2329a02f416f4994106f3a",
            new_revision="024cad6a831de75c2c3c26e711ba8c4a82ccd24b",
            last_modified="2026-09-20T10:00:00.000Z",
            downloads=1234,
            note="ocr-rec-ppocrv6-tiny: upstream has a newer commit",
        ),
        make_change(
            "new",
            repo="kha-white/manga-ocr-large",
            catalog_id=None,
            old_revision=None,
            new_revision=None,
            last_modified="2026-09-21T00:00:00.000Z",
            downloads=456,
            note="not in the catalog",
        ),
        make_change(
            "missing",
            repo="jzhang533/manga-ocr-base-2025",
            catalog_id="ocr-rec-manga-ocr-2025",
            old_revision="1e64d5beb41ee6bee72c200838ddb93ad541482a",
            new_revision=None,
            last_modified=None,
            downloads=None,
            note="repo not found (renamed or removed?)",
        ),
    ]


def test_render_report_golden() -> None:
    assert script.render_report(make_golden_changes(), generated=GENERATED) == GOLDEN


def test_render_report_is_deterministic() -> None:
    changes = make_golden_changes()
    assert script.render_report(list(reversed(changes)), generated=GENERATED) == GOLDEN


def test_render_report_without_changes() -> None:
    report = script.render_report([], generated=GENERATED)
    assert report == f"# Model watch report\n\nGenerated: {GENERATED}\n\nNo changes.\n"


# --- main ----------------------------------------------------------------------------------------------


def run_main(*args: str, fetch: Any = None) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = script.main(list(args), fetch=fetch)
    return code, out.getvalue(), err.getvalue()


def test_main_offline_validates_both_files_without_network(tmp_path: Path) -> None:
    catalog_path, config_path = write_files(tmp_path, catalog_toml(("e1", "org/x", A)))

    def explode(url: str, params: Mapping[str, str] | None = None) -> tuple[int, Any]:
        raise AssertionError("network call in --offline mode")

    code, out, _err = run_main(
        "--catalog", str(catalog_path), "--config", str(config_path), "--offline", fetch=explode
    )
    assert code == 0
    assert out.strip() == "ok"


def test_main_fail_on_change_exit_codes(tmp_path: Path) -> None:
    catalog_path, config_path = write_files(
        tmp_path, catalog_toml(("e1", "org/moved", A), ("e2", "org/stable", A))
    )
    common = ("--catalog", str(catalog_path), "--config", str(config_path))
    changed = FakeFetch(model={"org/moved": (200, {"sha": B}), "org/stable": (200, {"sha": A})})
    code, out, _err = run_main(*common, "--fail-on-change", fetch=changed)
    assert code == 3
    assert "# Model watch report" in out
    unchanged = FakeFetch(model={"org/moved": (200, {"sha": A}), "org/stable": (200, {"sha": A})})
    code, _out, _err = run_main(*common, "--fail-on-change", fetch=unchanged)
    assert code == 0
    code, _out, _err = run_main(*common, fetch=changed)  # without --fail-on-change: report only
    assert code == 0


def test_main_writes_markdown_and_json_even_without_changes(tmp_path: Path) -> None:
    fetch = FakeFetch(model={"org/stable": (200, {"sha": A, "downloads": 2})})
    catalog_path, config_path = write_files(tmp_path, catalog_toml(("e1", "org/stable", A)))
    markdown, json_out = tmp_path / "report.md", tmp_path / "report.json"
    code, out, _err = run_main(
        "--catalog",
        str(catalog_path),
        "--config",
        str(config_path),
        "--markdown",
        str(markdown),
        "--json",
        str(json_out),
        fetch=fetch,
    )
    assert code == 0
    report = markdown.read_text(encoding="utf-8")
    assert report == out and "No changes." in report
    assert json.loads(json_out.read_text(encoding="utf-8")) == []


def test_main_json_lists_changes_as_dicts(tmp_path: Path) -> None:
    fetch = FakeFetch(
        model={"org/moved": (200, {"sha": B, "lastModified": "2026-09-02T00:00:00.000Z", "downloads": 9})}
    )
    catalog_path, config_path = write_files(tmp_path, catalog_toml(("e1", "org/moved", A)))
    json_out = tmp_path / "report.json"
    code, _out, _err = run_main(
        "--catalog",
        str(catalog_path),
        "--config",
        str(config_path),
        "--json",
        str(json_out),
        "--fail-on-change",
        fetch=fetch,
    )
    assert code == 3
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert len(payload) == 1
    assert set(payload[0]) == {
        "kind",
        "repo",
        "catalog_id",
        "old_revision",
        "new_revision",
        "last_modified",
        "downloads",
        "note",
    }
    assert payload[0]["kind"] == "updated" and payload[0]["new_revision"] == B


def test_main_prints_report_on_a_legacy_windows_console(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fetch = FakeFetch(model={"org/moved": (200, {"sha": B, "downloads": 1})})
    catalog_path, config_path = write_files(tmp_path, catalog_toml(("e1", "org/moved", A)))
    buffer = io.BytesIO()
    console = io.TextIOWrapper(buffer, encoding="cp1252")  # the default on a German/Spanish Windows console
    monkeypatch.setattr(sys, "stdout", console)
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        code = script.main(
            ["--catalog", str(catalog_path), "--config", str(config_path), "--fail-on-change"], fetch=fetch
        )
    assert code == 3
    console.flush()
    output = buffer.getvalue().decode("utf-8")  # the reconfigured console emits UTF-8, not cp1252
    assert "`aaaaaaaa` → `bbbbbbbb`" in output


def test_main_bad_config_exits_2(tmp_path: Path) -> None:
    catalog_path, config_path = write_files(tmp_path, catalog_toml(("e1", "org/x", A)), config="broken [[[\n")
    code, _out, err = run_main("--catalog", str(catalog_path), "--config", str(config_path))
    assert code == 2
    assert "error" in err
    code, _out, err = run_main("--catalog", str(catalog_path), "--config", str(tmp_path / "missing.toml"))
    assert code == 2
    assert "error" in err


def test_main_bad_catalog_exits_2(tmp_path: Path) -> None:
    _catalog_path, config_path = write_files(tmp_path, "")
    code, _out, _err = run_main("--catalog", str(tmp_path / "missing.toml"), "--config", str(config_path))
    assert code == 2


def test_main_load_catalog_entries_minimal(tmp_path: Path) -> None:
    catalog_path, _config_path = write_files(tmp_path, catalog_toml(("e1", "org/x", A)))
    entries = script.load_catalog_entries(catalog_path)
    assert entries == [script.CatalogEntry(id="e1", format="hf", upstream_repo="org/x", upstream_revision=A)]


def test_default_fetch_sends_bearer_token_from_hf_token(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("authorization"))
        return httpx.Response(200, json={"sha": A})

    monkeypatch.setenv("HF_TOKEN", "secret-token-123")
    status, _payload = script.http_fetch(
        "https://huggingface.co/api/models/org/x",
        transport=httpx.MockTransport(handler),
        sleep=lambda _s: None,
    )
    assert status == 200
    assert seen == ["Bearer secret-token-123"]


def test_default_fetch_sends_no_authorization_without_hf_token(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("authorization"))
        return httpx.Response(200, json={"sha": A})

    monkeypatch.delenv("HF_TOKEN", raising=False)
    script.http_fetch(
        "https://huggingface.co/api/models/org/x",
        transport=httpx.MockTransport(handler),
        sleep=lambda _s: None,
    )
    assert seen == [None]


def test_main_hf_token_never_appears_in_output_or_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", "supersecret-token")
    fetch = FakeFetch(model={"org/moved": (500, None)})
    catalog_path, config_path = write_files(tmp_path, catalog_toml(("e1", "org/moved", A)))
    markdown = tmp_path / "report.md"
    code, out, err = run_main(
        "--catalog",
        str(catalog_path),
        "--config",
        str(config_path),
        "--markdown",
        str(markdown),
        "--fail-on-change",
        fetch=fetch,
    )
    assert code == 0  # a failing upstream is a warning, never a change
    assert "supersecret-token" not in out + err + markdown.read_text(encoding="utf-8")


def test_default_fetch_retries_503_then_succeeds() -> None:
    attempts: list[str] = []
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(str(request.url))
        return httpx.Response(503) if len(attempts) <= 2 else httpx.Response(200, json={"sha": A})

    status, payload = script.http_fetch(
        "https://huggingface.co/api/models/org/x", transport=httpx.MockTransport(handler), sleep=sleeps.append
    )
    assert (status, payload) == (200, {"sha": A})
    assert len(attempts) == 3
    assert sleeps == [0.5, 1.0]


def test_default_fetch_gives_up_after_two_retries() -> None:
    attempts: list[str] = []
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(str(request.url))
        return httpx.Response(503)

    with pytest.raises(script.FetchError):
        script.http_fetch(
            "https://huggingface.co/api/models/org/x",
            transport=httpx.MockTransport(handler),
            sleep=sleeps.append,
        )
    assert len(attempts) == 3
    assert sleeps == [0.5, 1.0]


def test_default_fetch_gives_up_on_transport_errors() -> None:
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    with pytest.raises(script.FetchError):
        script.http_fetch(
            "https://huggingface.co/api/models/org/x",
            transport=httpx.MockTransport(handler),
            sleep=sleeps.append,
        )
    assert sleeps == [0.5, 1.0]


# --- workflow ------------------------------------------------------------------------------------------


def test_model_watch_workflow() -> None:
    workflow_path = REPO_ROOT / ".github" / "workflows" / "model-watch.yml"
    raw = workflow_path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    triggers = data.get("on", data.get(True))  # YAML 1.1 parses the bare `on` key as boolean True
    assert set(triggers) == {"schedule", "workflow_dispatch"}
    assert triggers["schedule"] == [{"cron": "0 6 * * 1"}]
    assert data["permissions"] == {"contents": "read", "issues": "write"}
    assert len(data["jobs"]) == 1
    job = next(iter(data["jobs"].values()))
    assert job["runs-on"] == "ubuntu-latest"
    steps_text = "\n".join(str(step.get("run", "")) + str(step.get("uses", "")) for step in job["steps"])
    assert "scripts/model_watch.py" in steps_text
    assert "gh issue create" in steps_text
    assert "gh issue comment" in steps_text
    assert "actions/upload-artifact" in steps_text
    assert "continue-on-error" not in raw
    assert "secrets.GITHUB_TOKEN" in raw
    assert "secrets.HF_TOKEN" in raw
