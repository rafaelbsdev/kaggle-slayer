"""Turns a run's ledger into the evidence a reader can check: a Markdown report and one chart."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from slayer.ledger import Ledger

TOP_EXPERIMENTS = 10
REJECTIONS_SHOWN = 10


def _number(value, digits=5) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _table(headers: list[str], rows: list[list]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    return lines + ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]


def _results(entries: list[dict]) -> list[str]:
    submissions = [e for e in entries if e["kind"] == "submission"]
    if not submissions:
        return ["## Result", "", "No submissions yet.", ""]
    scored = [s for s in submissions if s["public_score"] is not None]
    lines = ["## Result", ""]
    if scored:
        lines.append(f"- Best public score: **{max(s['public_score'] for s in scored):.5f}**")
    snapshots = [e for e in entries if e["kind"] == "phase1_snapshot"]
    boards = [e for e in entries if e["kind"] == "leaderboard" and e["rank"] is not None]
    if snapshots:  # the board shows a team's best, which may come from outside this ledger (decision 0011)
        own = snapshots[-1]
        lines.append(f"- Rank of this score: **{own['rank']} of {own['teams']}** (top {own['top_percent']:.1f}%), read at {own['taken_at']}")
        lines.append(f"- Public score needed: top 20%: {own['top_20_needs']:.5f}, top 10%: {own['top_10_needs']:.5f}, top 5%: {own['top_5_needs']:.5f}")
        lines.append("- Kaggle's board shows a higher score for this team, which uses other people's predictions. It has its own ledger and is left out here.")
    elif boards:
        board = boards[-1]
        lines.append(f"- Rank: **{board['rank']} of {board['team_count']}** (top {board['percentile']:.1f}%)")
        cutoffs = ", ".join(f"{name.replace('_', ' ')}: {score:.5f}" for name, score in board["cutoffs"].items())
        lines.append(f"- Public score needed: {cutoffs}")
    lines += ["- These are public leaderboard numbers. Final standings come when the competition closes.", ""]
    lines += ["## Submissions", ""]
    rows = [[s["ts"], s["candidate"], _number(s["cv"]), _number(s["public_score"]), _number(s["gap"])]
            for s in submissions]
    return lines + _table(["When (UTC)", "Candidate", "CV", "Public", "CV minus public"], rows) + [""]


def _experiments(entries: list[dict]) -> list[str]:
    experiments = sorted((e for e in entries if e["kind"] == "experiment"), key=lambda e: -e["cv"])
    if not experiments:
        return []
    rows = [[e["id"], e["round"], e["family"], _number(e["cv"]), _number(e["std"]), e["seconds"], e["hypothesis"]]
            for e in experiments[:TOP_EXPERIMENTS]]
    lines = [f"## Best experiments ({len(experiments)} ran)", ""]
    return lines + _table(["Id", "Round", "Family", "CV", "Fold std", "Seconds", "Hypothesis"], rows) + [""]


def _team(entries: list[dict]) -> list[str]:
    decisions = [e for e in entries if e["kind"] == "agent_decision"]
    fallbacks = [e for e in entries if e["kind"] == "fallback"]
    errors = [e for e in entries if e["kind"] == "agent_error"]
    roles = sorted({e["role"] for e in decisions + fallbacks + errors})
    if not roles:
        return []
    rows = [[role,
             sum(e["role"] == role for e in decisions),
             sum(e["role"] == role and not e["passed"] for e in decisions),
             sum(e["role"] == role for e in fallbacks),
             sum(e["role"] == role for e in errors)] for role in roles]
    lines = ["## The team", ""] + _table(["Agent", "Answers", "Rejected", "Heuristic took over", "Errors"], rows) + [""]

    rejected = [e for e in decisions if not e["passed"]]
    if rejected:
        lines += ["## What the verifier stopped", ""]
        lines += [f"- `{e['role']}` ({e['layer']}): {e['reason']}" for e in rejected[-REJECTIONS_SHOWN:]]
        lines.append("")
    return lines


def _lessons(entries: list[dict]) -> list[str]:
    lessons = [e for e in entries if e["kind"] == "lesson" and e["lessons"]]
    if not lessons:
        return []
    lines = ["## Lessons the team passed forward", ""]
    for entry in lessons:
        who = " (operator)" if entry.get("source") == "operator" else ""
        lines += [f"- Round {entry['round']}{who}: {lesson}" for lesson in entry["lessons"]]
    return lines + [""]


def _closing(entries: list[dict]) -> list[str]:
    lines = []
    canaries = [e for e in entries if e["kind"] == "leak_canary"]
    if canaries:
        passed = sum(e["passed"] for e in canaries)
        lines.append(f"- Leak canary (shuffled labels): {passed} of {len(canaries)} feature plans passed.")
    spend = [e for e in entries if e["kind"] == "llm_spend"]
    if spend:
        lines.append(f"- LLM spend: ${spend[-1]['spent_usd']:.2f} of ${spend[-1]['limit_usd']:.2f}.")
    stops = [e for e in entries if e["kind"] == "stop"]
    if stops:
        detail = f" ({stops[-1]['detail']})" if stops[-1]["detail"] else ""
        lines.append(f"- Last stop reason: `{stops[-1]['reason']}`{detail}.")
    return ["## Run", ""] + lines + [""] if lines else []


def _chart(entries: list[dict], path: Path) -> bool:
    scored = [e for e in entries if e["kind"] == "submission" and e["public_score"] is not None]
    if not scored:
        return False
    positions = range(1, len(scored) + 1)
    figure, axis = plt.subplots(figsize=(7, 3.5))
    axis.plot(positions, [s["cv"] for s in scored], marker="o", label="Cross-validation")
    axis.plot(positions, [s["public_score"] for s in scored], marker="o", label="Public leaderboard")
    axis.set_xlabel("Submission")
    axis.set_ylabel("AUC")
    axis.set_xticks(list(positions))
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return True


def write_report(ledger_path) -> Path:
    ledger_path = Path(ledger_path)
    entries = Ledger(ledger_path).read()
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# Run report: {ledger_path.parent.name}", "", *_results(entries)]
    if _chart(entries, ledger_path.parent / "cv_vs_public.png"):
        lines += ["![CV against public score](cv_vs_public.png)", ""]
    lines += [*_experiments(entries), *_team(entries), *_lessons(entries), *_closing(entries)]
    path = ledger_path.parent / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
