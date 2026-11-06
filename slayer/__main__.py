"""Command line: `python -m slayer check|run|report --competition <slug>`."""
import argparse
import os
import sys
from pathlib import Path

from slayer.brains import Budget, ClaudeBrain, HeuristicBrain
from slayer.kaggle_client import KaggleClient, KaggleError
from slayer.ledger import Ledger
from slayer.orchestrator import Orchestrator, RunConfig
from slayer.report import write_report
from slayer.verifier import Verifier

DEFAULT_AGENT_MODEL = "claude-opus-5-5"
DEFAULT_CHECK_MODEL = "claude-haiku-5-5"
CLEAN_STOPS = {"rounds_done", "no_gain", "target_reached", "time_limit", "budget", "unsupported"}


def load_env_file(path) -> None:
    """Fills missing variables from .env. Real environment variables win, and empty values are ignored."""
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip("'\"")
        if value:
            os.environ.setdefault(key.strip(), value)


def spent_so_far(root, slug: str) -> float:
    entries = Ledger(Path(root) / "runs" / slug / "ledger.jsonl").read("llm_spend")
    return entries[-1]["spent_usd"] if entries else 0.0


def _build_team(args):
    if args.brain == "heuristic":
        return HeuristicBrain(), Verifier()
    import anthropic

    client = anthropic.Anthropic()
    budget = Budget(args.budget_usd, spent=spent_so_far(args.root, args.competition))
    agents = ClaudeBrain(client, os.environ.get("SLAYER_AGENT_MODEL", DEFAULT_AGENT_MODEL), budget, effort=args.effort)
    checker = ClaudeBrain(client, os.environ.get("SLAYER_CHECK_MODEL", DEFAULT_CHECK_MODEL), budget, effort="low")
    return agents, Verifier(checker)


def _check(args) -> int:
    kaggle = KaggleClient()
    try:
        info = kaggle.info(args.competition)
        if not info["entered"]:
            print(f"joined: False. Accept the rules on https://www.kaggle.com/competitions/{args.competition}/rules")
            return 1
        print("joined: True")
        print(f"teams: {info['team_count']}")
        print(f"your rank: {info['user_rank']}")
        print(f"quota left today: {kaggle.quota_left(args.competition)}")
    except KaggleError as exc:
        print(f"kaggle error: {exc}")
        return 1
    return 0


def _run(args) -> int:
    brain, verifier = _build_team(args)
    config = RunConfig(
        slug=args.competition, root=args.root, rounds=args.rounds, allow_submit=args.allow_submit,
        min_gain=args.min_gain, max_trees=args.max_trees, patience=args.patience, device=args.device,
        target_percentile=args.target_percentile, max_minutes=args.max_minutes, no_gain_rounds=args.no_gain_rounds,
        notes=args.note)
    reason = Orchestrator(config, KaggleClient(), brain, verifier).run()
    report = write_report(Path(args.root) / "runs" / args.competition / "ledger.jsonl")
    print(f"stopped: {reason}")
    print(f"report: {report}")
    return 0 if reason in CLEAN_STOPS else 1


def _report(args) -> int:
    print(write_report(Path(args.root) / "runs" / args.competition / "ledger.jsonl"))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="slayer", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def add_command(name, help_text):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--competition", required=True, help="competition slug from its Kaggle URL")
        command.add_argument("--root", default=".", help="project folder holding data, runs and artifacts")
        return command

    add_command("check", "confirm Kaggle access, joined rules and quota. Costs nothing")
    add_command("report", "rebuild the report from the ledger")
    run = add_command("run", "compete")
    run.add_argument("--brain", choices=["heuristic", "claude"], default="heuristic")
    run.add_argument("--budget-usd", type=float, help="hard cap on LLM spend. Required with --brain claude")
    run.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    run.add_argument("--rounds", type=int, default=3)
    run.add_argument("--allow-submit", action="store_true", help="without this flag the run only rehearses")
    run.add_argument("--min-gain", type=float, default=1e-4, help="CV gain that justifies a new submission")
    run.add_argument("--max-trees", type=int, default=8000)
    run.add_argument("--patience", type=int, default=200)
    run.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    run.add_argument("--target-percentile", type=float, help="stop once the rank reaches this top percentage")
    run.add_argument("--max-minutes", type=float, help="start no new round after this many minutes")
    run.add_argument("--no-gain-rounds", type=int, default=2, help="stop after this many rounds with no CV gain")
    run.add_argument("--note", action="append", default=[],
                     help="tell the agents something from outside the run. Replaces earlier lessons. Never paste rows")

    args = parser.parse_args(argv)
    if args.command == "run" and args.brain == "claude" and args.budget_usd is None:
        parser.error("--brain claude needs --budget-usd, so spend is always capped")
    load_env_file(Path(args.root) / ".env")
    return {"check": _check, "run": _run, "report": _report}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
