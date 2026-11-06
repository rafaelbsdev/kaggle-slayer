"""The loop. All mechanics live here, and a verifier stands between every pair of steps.

Everything a run learns goes to the ledger first, so a crash or a new day resumes where it stopped.
"""
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import get_args

import numpy as np

from slayer.brains import BrainRejected, BrainUnavailable, BudgetExceeded, HeuristicBrain
from slayer.contracts import CONTRACTS, FeaturePlan, ModelPlan, ValidationPlan, experiment_id
from slayer.data import build_profile, load_competition, temporal_hint
from slayer.ensemble import blend
from slayer.kaggle_client import KaggleError, NotAuthenticated, RulesNotAccepted
from slayer.ledger import Ledger
from slayer.privacy import PayloadRejected
from slayer.submission import validate_submission, write_submission
from slayer.training import leak_canary, load_predictions, run_experiment, save_predictions, tokenizer_available
from slayer.verifier import Verifier

MAX_ATTEMPTS = 2
OVERVIEW_CHARS = 6_000
EXPERIMENTS_SHOWN = 20
LESSONS_SHOWN = 10
BLEND_VERSION = 3  # raise it when a blend's CV or its test predictions stop being comparable with earlier ones
FEATURE_DEFAULTS = {name: field.default for name, field in FeaturePlan.model_fields.items() if name != "rationale"}
OTHER_METRICS = re.compile(r"\b(rmse|rmsle|mae|mape|log ?loss|accuracy|f1|map@\d+|dice|iou)\b", re.I)


class StepFailed(RuntimeError):
    """Neither the agent nor the heuristic produced something that passes the checks."""


def plan_key(experiment: dict) -> str:
    """What an experiment runs apart from its seed. An option added later counts as its default."""
    features = {**FEATURE_DEFAULTS, **experiment["features"]}
    return json.dumps([experiment["family"], experiment["params"], features, experiment["recency_half_life"]], sort_keys=True)


@dataclass
class RunConfig:
    slug: str
    root: Path = Path(".")
    rounds: int = 3
    allow_submit: bool = False
    min_gain: float = 1e-4
    max_trees: int = 8000
    patience: int = 200
    device: str = "auto"
    no_gain_rounds: int = 2
    target_percentile: float | None = None
    max_minutes: float | None = None
    seed: int = 0
    notes: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.root = Path(self.root)


def detect_metric(overview: str) -> str | None:
    lowered = (overview or "").lower()
    if "auc" in lowered or "area under" in lowered:
        return "auc"
    other = OTHER_METRICS.search(overview or "")
    return other.group(0).lower() if other else None


class Orchestrator:
    def __init__(self, config: RunConfig, kaggle, brain, verifier: Verifier):
        self.config, self.kaggle, self.brain, self.verifier = config, kaggle, brain, verifier
        self.fallback = HeuristicBrain()
        root, slug = config.root, config.slug
        self.ledger = Ledger(root / "runs" / slug / "ledger.jsonl")
        self.raw_dir = root / "data" / slug / "raw"
        self.work_dir = root / "data" / slug / "work"
        self.board_dir = root / "data" / slug / "leaderboard"
        self.artifacts = root / "artifacts" / slug
        self.submissions = root / "submissions" / slug
        budget = getattr(brain, "budget", None)
        if budget is not None:
            budget.on_charge = lambda _: self._record_spend()  # so a hard kill loses at most the call in flight

    # --- entry point ------------------------------------------------------------------------
    def run(self) -> str:
        try:
            reason = self._run()
            detail = ""
        except BudgetExceeded as exc:
            reason, detail = "budget", str(exc)
        except PayloadRejected as exc:
            reason, detail = "privacy_guard", str(exc)
        except BrainUnavailable as exc:
            reason, detail = "brain_unavailable", str(exc)
        except StepFailed as exc:
            reason, detail = "step_failed", str(exc)
        except (NotAuthenticated, RulesNotAccepted) as exc:
            reason, detail = "kaggle_access", str(exc)
        except KaggleError as exc:
            reason, detail = "kaggle_error", str(exc)
        except BaseException as exc:  # Ctrl-C or a bug: leave the ledger honest, then let it surface
            self._record_spend()
            self.ledger.append("stop", reason="crashed", detail=repr(exc)[:300])
            raise
        self._record_spend()
        self.ledger.append("stop", reason=reason, detail=detail)
        return reason

    def _run(self) -> str:
        started = time.monotonic()
        self._prepare()
        if self.config.notes:
            self.ledger.append("lesson", round=len(self.ledger.read("round_start")), diagnosis="Note from the operator.",
                               lessons=self.config.notes, source="operator")
        brief = self._plan("scout", self._scout_request)
        if not brief.supported:
            return "unsupported"
        self.validation = self._plan("validation", self._validation_request)

        rounds_without_gain = 0
        for _ in range(self.config.rounds):
            if self.config.max_minutes is not None and (time.monotonic() - started) / 60 >= self.config.max_minutes:
                return "time_limit"
            gained = self._round()
            self._record_spend()
            rounds_without_gain = 0 if gained else rounds_without_gain + 1
            if self._reached_target():
                return "target_reached"
            if rounds_without_gain >= self.config.no_gain_rounds:
                return "no_gain"
        return "rounds_done"

    # --- step 1: collect ----------------------------------------------------------------------
    def _prepare(self) -> None:
        slug = self.config.slug
        info = self.kaggle.info(slug)
        if not info["entered"]:
            raise RulesNotAccepted(
                f"This account has not joined {slug}. Accept its rules on "
                f"https://www.kaggle.com/competitions/{slug}/rules")
        try:
            self.overview = self.kaggle.overview(slug)[:OVERVIEW_CHARS]
        except KaggleError as exc:  # the overview only informs the agents; the profile still measures drift
            self.ledger.append("overview_unavailable", error=str(exc)[:300])
            self.overview = ""
        if not (self.raw_dir / "train.csv").exists():
            self.kaggle.download(slug, self.raw_dir)
        self.data = load_competition(self.raw_dir, self.work_dir)
        self.profile = build_profile(self.data, seed=self.config.seed)
        self.temporal = temporal_hint(self.overview)
        self.ledger.append("data_ready", n_train=self.profile["n_train"], n_test=self.profile["n_test"],
                           n_features=self.profile["n_features"], team_count=info["team_count"])
        self.ledger.append("profile", **self.profile)

    # --- asking agents ------------------------------------------------------------------------
    def _ask(self, role: str, payload: dict, ctx: dict):
        schema = CONTRACTS[role]
        feedback = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            request = {**payload, "rejected_because": feedback} if feedback else payload
            try:
                answer = self.brain.decide(role, request, schema)
            except (BudgetExceeded, BrainUnavailable, PayloadRejected):
                raise
            except BrainRejected as exc:
                feedback = str(exc)[:500]
                self.ledger.append("agent_decision", role=role, attempt=attempt, passed=False, layer="schema",
                                   reason=feedback, notes=[], output=None)
                continue
            except Exception as exc:  # a provider failure must not end a run that can still compete
                self.ledger.append("agent_error", role=role, attempt=attempt, error=str(exc)[:300])
                continue
            verdict = self.verifier.check(role, answer, ctx, evidence=request)
            self.ledger.append("agent_decision", role=role, attempt=attempt, passed=verdict.passed,
                               layer=verdict.layer, reason=verdict.reason, notes=verdict.notes,
                               output=answer.model_dump())
            if verdict.passed:
                return verdict.output
            feedback = verdict.reason

        answer = self.fallback.decide(role, payload, schema)
        verdict = Verifier().check(role, answer, ctx)
        self.ledger.append("fallback", role=role, passed=verdict.passed, reason=verdict.reason,
                           output=answer.model_dump())
        if not verdict.passed:
            raise StepFailed(f"{role}: {verdict.reason}")
        return verdict.output

    def _plan(self, role: str, build_request):
        """Plans that define the run are decided once and kept, so results stay comparable."""
        for entry in reversed(self.ledger.read("plan")):
            if entry["role"] == role:
                return CONTRACTS[role].model_validate(entry["output"])
        payload, ctx = build_request()
        output = self._ask(role, payload, ctx)
        self.ledger.append("plan", role=role, output=output.model_dump())
        return output

    def _scout_request(self):
        metric = detect_metric(self.overview)
        payload = {"competition": self.config.slug, "overview": self.overview, "profile": self.profile,
                   "target_is_binary": self.profile["target_is_binary"], "metric": metric}
        return payload, {"target_is_binary": self.profile["target_is_binary"], "metric": metric}

    def _validation_request(self):
        payload = {"profile": self.profile, "temporal_hint": self.temporal}
        keys = ("adversarial_auc", "row_order_drift_auc", "id_order_drift_auc")
        return payload, {"temporal_hint": self.temporal, "drift_clock_available": self.profile["drift_clock"] is not None,
                         **{k: self.profile[k] for k in keys}}

    # --- state read back from the ledger ------------------------------------------------------
    def _experiments(self) -> list[dict]:
        return [e for e in self.ledger.read("experiment") if (self.artifacts / f"{e['id']}.npz").exists()]

    def _lessons(self) -> list[str]:
        """An operator's note replaces earlier lessons and stays in view: a person writes one when they stopped holding."""
        entries = self.ledger.read("lesson")
        start = max((i for i, e in enumerate(entries) if e.get("source") == "operator"), default=None)
        notes = [] if start is None else entries[start]["lessons"]
        since = entries if start is None else entries[start + 1:]
        lessons = list(dict.fromkeys(lesson for e in since for lesson in e["lessons"]))
        return notes + lessons[-LESSONS_SHOWN:]

    def _experiment_summaries(self) -> list[dict]:
        keys = ("id", "family", "params", "features", "recency_half_life", "seed", "cv", "std", "best_iters",
                "seconds", "round", "hypothesis")
        families = get_args(ModelPlan.model_fields["family"].annotation)
        current = [e for e in self._experiments()
                   if set(e["features"]) <= set(FeaturePlan.model_fields) and e["family"] in families]  # no retired options
        best = sorted(current, key=lambda e: -e["cv"])[:EXPERIMENTS_SHOWN]
        return [{k: e[k] for k in keys} for e in best]

    def _best_cv(self) -> float | None:
        scores = [e["cv"] for e in self._experiments()] + [e["cv"] for e in self.ledger.read("blend")]
        return max(scores, default=None)

    # --- one round ----------------------------------------------------------------------------
    def _round(self) -> bool:
        step = len(self.ledger.read("round_start"))
        self.ledger.append("round_start", round=step)
        previous_best = self._best_cv()

        shared = {"round": step, "profile": self.profile, "temporal": self._is_temporal(),
                  "n_features": self.profile["n_features"], "lessons": self._lessons(),
                  "experiments": self._experiment_summaries()}
        features = self._ask("features", {**shared, "tokenizer_available": tokenizer_available()},
                             {"n_features": self.profile["n_features"], "tokenizer_available": tokenizer_available()})
        canary = self._canary(features)
        known = {e["id"] for e in self.ledger.read("experiment")}
        batch = self._ask("tuner", {**shared, "feature_plan": features.model_dump()},
                          {"known_experiments": known, "feature_plan": features, "validation": self.validation})
        for model in batch.models:
            self._train(step, features, model, canary)

        if not self._experiments():
            self.ledger.append("round_empty", round=step)
            return False
        self._blend_and_submit(step)
        self._monitor()
        self._learn(step)

        best = self._best_cv()
        return previous_best is None or (best is not None and best - previous_best >= self.config.min_gain)

    def _is_temporal(self) -> bool:
        return self.validation.scheme == "time_kfold"

    def _canary(self, features: FeaturePlan) -> bool:
        spec = json.dumps(features.model_dump(exclude={"rationale"}), sort_keys=True).encode()
        key = hashlib.sha256(spec).hexdigest()[:12]
        for entry in self.ledger.read("leak_canary"):
            if entry["plan"] == key:
                return entry["passed"]
        result = leak_canary(self.validation, features, self.data, seed=self.config.seed)
        self.ledger.append("leak_canary", plan=key, **result)
        return result["passed"]

    def _train(self, step: int, features: FeaturePlan, model: ModelPlan, canary_passed: bool) -> None:
        exp_id = experiment_id(self.validation, features, model)
        try:
            result = run_experiment(self.validation, features, model, self.data, max_trees=self.config.max_trees,
                                    patience=self.config.patience, device=self.config.device)
        except Exception as exc:  # one failing model (a GPU out of memory, say) must not end the run
            self.ledger.append("experiment_failed", id=exp_id, family=model.family, error=str(exc)[:300])
            return
        save_predictions(self.artifacts / f"{exp_id}.npz", result)
        self.ledger.append(
            "experiment", id=exp_id, round=step, family=model.family, params=model.params.model_dump(),
            features=features.model_dump(exclude={"rationale"}), recency_half_life=model.recency_half_life,
            seed=model.seed, hypothesis=model.hypothesis, validation_scheme=self.validation.scheme,
            canary_passed=canary_passed, cv=result.cv, std=result.std, fold_scores=result.fold_scores,
            best_iters=result.best_iters, seconds=result.seconds)

    # --- blending and the submission gate -----------------------------------------------------
    def _blend_and_submit(self, step: int) -> None:
        experiments = self._experiments()
        clean = [e for e in experiments if e["canary_passed"]]
        pool = clean or experiments  # keeps the run measuring; the critic's gate still blocks submitting

        oofs, tests, mask = {}, {}, None
        for entry in pool:
            oof, test, mask = load_predictions(self.artifacts / f"{entry['id']}.npz")
            oofs[entry["id"]], tests[entry["id"]] = oof, test
        plans = {e["id"]: plan_key(e) for e in pool}
        families = {e["id"]: e["family"] for e in pool}
        blended = blend(oofs, tests, self.data.y, mask, seed=self.config.seed, plans=plans, families=families)
        blend_id = f"blend-r{step}"
        self.ledger.append("blend", id=blend_id, round=step, cv=blended["cv"], weights=blended["weights"])

        best_single = max(pool, key=lambda e: e["cv"])
        candidates = {blend_id: {"cv": blended["cv"], "std": best_single["std"]},
                      best_single["id"]: {"cv": best_single["cv"], "std": best_single["std"]}}
        predictions = {blend_id: blended["test"], best_single["id"]: tests[best_single["id"]]}

        problems = []
        for candidate, values in predictions.items():
            path = write_submission(self.data.sample_path, self.data.test_ids, values,
                                    self.submissions / f"{candidate}.csv")
            problems += validate_submission(path, self.data.sample_path)
        if problems:
            self.ledger.append("submission_file_invalid", problems=problems)

        comparable = [s for s in self.ledger.read("submission") if s.get("blend_version") == BLEND_VERSION]
        best_submitted = max((s["cv"] for s in comparable), default=None)
        quota = self.kaggle.quota_left(self.config.slug)
        facts = {"candidates": candidates, "best_submitted_cv": best_submitted, "quota_left": quota,
                 "min_gain": self.config.min_gain, "canary_passed": bool(clean), "file_ok": not problems}
        payload = {**facts, "recent_submissions": self._submission_summaries(), "lessons": self._lessons(),
                   "weights": blended["weights"]}
        try:
            decision = self._ask("critic", payload, facts)
        except StepFailed as exc:
            self.ledger.append("submission_held", reason=str(exc)[:300])
            return
        if decision.action != "submit":
            self.ledger.append("submission_held", reason=decision.reason)
            return
        self._submit(decision.candidate_id, candidates[decision.candidate_id]["cv"], decision.reason)

    def _submit(self, candidate: str, cv: float, reason: str) -> None:
        if not self.config.allow_submit:
            self.ledger.append("dry_run", candidate=candidate, cv=cv, reason=reason)
            return
        message = f"{candidate} cv={cv:.5f}"
        try:
            result = self.kaggle.submit(self.config.slug, self.submissions / f"{candidate}.csv", message)
        except (NotAuthenticated, RulesNotAccepted):
            raise
        except KaggleError as exc:  # it may still have counted: the quota is read fresh every round
            self.ledger.append("submission_failed", candidate=candidate, error=str(exc)[:300])
            return
        score = result["public_score"]
        self.ledger.append("submission", ref=result["ref"], candidate=candidate, cv=cv, public_score=score,
                           gap=None if score is None else cv - score, reason=reason, error=result.get("error"),
                           blend_version=BLEND_VERSION)

    def _submission_summaries(self) -> list[dict]:
        keys = ("candidate", "cv", "public_score", "gap")
        return [{k: s[k] for k in keys} for s in self.ledger.read("submission")[-10:]]

    # --- monitoring and learning --------------------------------------------------------------
    def _monitor(self) -> None:
        try:
            info = self.kaggle.info(self.config.slug)
            board = self.kaggle.leaderboard(self.config.slug, self.board_dir)
        except (NotAuthenticated, RulesNotAccepted):
            raise
        except KaggleError as exc:  # monitoring must not stop the competing
            self.ledger.append("monitor_failed", error=str(exc)[:300])
            return
        rank, teams = info["user_rank"], info["team_count"]
        scores = sorted((row["score"] for row in board), reverse=True)
        cutoffs = {}
        if scores:
            for percent in (1, 5, 20):
                position = max(math.floor(percent / 100 * len(scores)), 1)
                cutoffs[f"top_{percent}"] = scores[position - 1]
        self.ledger.append("leaderboard", rank=rank, team_count=teams, cutoffs=cutoffs,
                           percentile=None if rank is None else 100 * rank / teams)

    def _latest_gap(self) -> float | None:
        scored = [s for s in self.ledger.read("submission") if s["gap"] is not None]
        return scored[-1]["gap"] if scored else None

    def _learn(self, step: int) -> None:
        gap = self._latest_gap()
        if gap is None:
            return
        boards = self.ledger.read("leaderboard")
        payload = {"round": step, "observed_gap": gap, "recent_submissions": self._submission_summaries(),
                   "leaderboard": boards[-1] if boards else None, "lessons": self._lessons(),
                   "experiments": self._experiment_summaries()}
        try:
            report = self._ask("analyst", payload, {"observed_gap": gap})
        except StepFailed as exc:
            self.ledger.append("analysis_skipped", reason=str(exc)[:300])
            return
        self.ledger.append("lesson", round=step, diagnosis=report.diagnosis, lessons=report.lessons)

    def _reached_target(self) -> bool:
        target = self.config.target_percentile
        boards = self.ledger.read("leaderboard")
        if target is None or not boards or boards[-1]["percentile"] is None:
            return False
        return boards[-1]["percentile"] <= target

    def _record_spend(self) -> None:
        budget = getattr(self.brain, "budget", None)
        if budget is not None:
            self.ledger.append("llm_spend", spent_usd=round(budget.spent, 6), limit_usd=budget.limit)
