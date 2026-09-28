import os

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"  # see the README: stale bytecode can poison verdicts

P = "src/omniscan/qa/problems.py"
CLI = "src/omniscan/edits/cli.py"
WEB = "src/omniscan/web/app.py"
MUTANTS = [
    (
        P,
        'status.get(issue.region_id, "todo"), word=issue.word',
        'status.get(issue.region_id, "todo")',
        "a typo loses its word",
    ),
    (
        P,
        '            status.get(issue.region_id, "todo"),\n            finished_page=True,',
        '            "todo",\n            finished_page=True,',
        "finished-page problems forget the line's review state",
    ),
    (P, "            finished_page=True,\n", "", "a finished-page problem reads as an artifact check"),
    (P, "            read=issue.read,\n", "", "what the re-read saw is lost"),
    (
        P,
        "        leftover = load_issues(paths)\n    except OSError, ValueError:\n        leftover = []",
        "        leftover = load_issues(paths)\n    except OSError:\n        leftover = []",
        "a damaged qa.json breaks the list",
    ),
    (P, "    problems += [", "    problems = [", "the checks are dropped when a re-read ran"),
    (
        P,
        "    return sorted(problems, key=lambda problem: order.get(problem.region_id, len(order)))",
        "    return problems",
        "not in reading order",
    ),
    (
        P,
        "order.get(problem.region_id, len(order))",
        "order.get(problem.region_id, 0)",
        "a gone region's problem comes first",
    ),
    (CLI, 'if not (unchecked and p.status == "checked")', "if True", "--unchecked is ignored"),
    (
        CLI,
        '        where = " (finished page)" if problem.finished_page else ""',
        '        where = ""',
        "finished-page problems are not told apart",
    ),
    (
        CLI,
        '        checked = " [checked]" if problem.status == "checked" else ""',
        '        checked = ""',
        "checked lines are not marked",
    ),
    (
        WEB,
        '        return {"problems": [asdict(problem) for problem in chapter_problems(chapter_paths(series, chapter))]}',
        '        return {"problems": []}',
        "the web answers nothing",
    ),
]
