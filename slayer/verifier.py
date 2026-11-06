"""Verifier between every pair of steps.

It attests that an agent's output can move forward and never fixes anything. Cheapest layer first:
  1. schema        - Pydantic contract
  2. deterministic - checks against observed facts, no LLM
  3. semantic      - cheap model reads the reasoning against the facts (optional)
"""
from dataclasses import dataclass, field

from pydantic import BaseModel, ValidationError

from slayer.contracts import CONTRACTS, MAX_JOINT_SCAN_COLUMNS, SemanticCheck, experiment_id

DRIFT_AUC = 0.75         # train-vs-test AUC above this means random folds would lie
TEMPORAL_EVIDENCE_AUC = 0.6
ORDER_EVIDENCE_AUC = 0.6 # first-vs-last rows AUC above this means the order carries time
MIN_FEATURES_KEPT = 0.2
CATBOOST_MAX_DEPTH = 10  # memory grows exponentially with depth on an 8 GB GPU
GAP_TOLERANCE = 0.0005

CRITERIA = {
    "scout": "The summary and risks must be consistent with the facts given.",
    "validation": "The rationale must be consistent with the drift score and the temporal hint.",
    "features": "The rationale must follow from the data profile and past experiments.",
    "tuner": "Each hypothesis must be consistent with past experiments and not repeat a failed idea. Trying a "
             "family or tree size no past experiment measured is allowed even if a lesson discourages it.",
    "critic": "The reason must be consistent with the candidate's CV, its noise and the remaining quota.",
    "analyst": "The diagnosis and lessons must follow from the scores given, with no invented facts.",
}


@dataclass
class Verdict:
    passed: bool
    layer: str
    reason: str = ""
    output: BaseModel | None = None
    notes: list[str] = field(default_factory=list)


def _failed(checks: list[tuple[str, bool, str]]) -> str:
    return "; ".join(f"{name}: {detail}" for name, ok, detail in checks if not ok)


def _scout(out, ctx):
    if not out.supported:
        return []
    metric = (ctx.get("metric") or "auc").lower()
    return [
        ("target_is_binary", ctx["target_is_binary"], "this system only handles binary targets"),
        ("metric_is_auc", "auc" in metric or "area under" in metric, f"unsupported metric {metric!r}"),
    ]


def _order_evidence(ctx) -> dict[str, bool]:
    """Which orderings of the training rows were measured to follow time."""
    return {
        "row_order": ctx["row_order_drift_auc"] >= ORDER_EVIDENCE_AUC,
        "id": ctx["id_order_drift_auc"] >= ORDER_EVIDENCE_AUC,
        "drift_clock": ctx["drift_clock_available"],
    }


def _validation(out, ctx):
    temporal = ctx["temporal_hint"]
    drift = ctx["adversarial_auc"]
    facts = f"temporal_hint={temporal}, train-vs-test AUC={drift:.3f}"
    if out.scheme == "stratified_kfold":
        ok = drift < DRIFT_AUC and not (temporal and any(_order_evidence(ctx).values()))
        return [("random_folds_on_drift", ok, facts)]
    ok = temporal or drift >= TEMPORAL_EVIDENCE_AUC
    return [
        ("time_folds_need_evidence", ok, facts),
        ("order_is_time", _order_evidence(ctx)[out.order_by],
         f"nothing measured shows {out.order_by} follows time, so these folds would be random and CV would lie"),
    ]


def _features(out, ctx):
    n_features = ctx["n_features"]
    kept = n_features - out.drop_drifting_top_k
    checks = [("keeps_enough_features", kept >= MIN_FEATURES_KEPT * n_features, f"keeps {kept} of {n_features}")]
    if out.number_tokens or out.shared_token_rates:
        checks.append(("tokenizer_present", ctx["tokenizer_available"], "number_tokens and shared_token_rates need the GPT-2 tokenizer file"))
    if out.joint_keys:
        checks.append(("joint_scan_fits", n_features <= MAX_JOINT_SCAN_COLUMNS,
                       f"joint_keys scans every pair of columns, so it takes at most {MAX_JOINT_SCAN_COLUMNS} features"))
    return checks


def _tuner(out, ctx):
    checks = []
    seen = set(ctx["known_experiments"])
    for model in out.models:
        exp_id = experiment_id(ctx["validation"], ctx["feature_plan"], model)
        checks.append(("no_repeated_experiment", exp_id not in seen, f"{exp_id} already ran"))
        seen.add(exp_id)
        params = model.params
        if model.family == "lightgbm":
            checks.append(("leaves_fit_depth", params.num_leaves <= 2 ** params.max_depth,
                           f"num_leaves={params.num_leaves} with max_depth={params.max_depth}"))
        if model.family == "catboost":
            checks.append(("catboost_depth", params.max_depth <= CATBOOST_MAX_DEPTH,
                           f"max_depth={params.max_depth} above {CATBOOST_MAX_DEPTH}"))
        if model.recency_half_life is not None:
            checks.append(("recency_needs_time_order", ctx["validation"].scheme == "time_kfold",
                           "recency weights need time folds to be measured honestly"))
    return checks


def _critic(out, ctx):
    if out.action == "hold":
        return []
    candidate = ctx["candidates"].get(out.candidate_id)
    best = ctx["best_submitted_cv"]
    gain_ok = candidate is None or best is None or candidate["cv"] - best >= ctx["min_gain"]
    return [
        ("candidate_exists", candidate is not None, f"unknown candidate {out.candidate_id!r}"),
        ("quota_left", ctx["quota_left"] > 0, "no submissions left today"),
        ("leak_canary_passed", ctx["canary_passed"], "the leak canary did not pass"),
        ("submission_file_valid", ctx["file_ok"], "the submission file did not validate"),
        ("cv_gain_above_noise", gain_ok, f"best submitted CV is {best}, minimum gain {ctx['min_gain']}"),
    ]


def _analyst(out, ctx):
    observed = ctx["observed_gap"]
    if observed is None:
        return []
    cited = out.cv_lb_gap_cited
    ok = cited is not None and abs(cited - observed) <= GAP_TOLERANCE
    return [("gap_matches_ledger", ok, f"cited {cited}, observed {observed:.5f}")]


DETERMINISTIC = {
    "scout": _scout,
    "validation": _validation,
    "features": _features,
    "tuner": _tuner,
    "critic": _critic,
    "analyst": _analyst,
}


class Verifier:
    def __init__(self, check_brain=None):
        self.check_brain = check_brain

    def check(self, role: str, raw: dict | BaseModel, ctx: dict, evidence: dict | None = None) -> Verdict:
        """`evidence` is what the agent was given, so the semantic layer judges it against the same facts."""
        schema = CONTRACTS[role]
        try:
            data = raw.model_dump() if isinstance(raw, BaseModel) else raw
            output = schema.model_validate(data)
        except ValidationError as exc:
            errors = [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()]
            return Verdict(False, "schema", "; ".join(errors))

        failed = _failed(DETERMINISTIC[role](output, ctx))
        if failed:
            return Verdict(False, "deterministic", failed, output)

        if self.check_brain is None:
            return Verdict(True, "deterministic", output=output)
        return self._semantic(role, output, ctx, evidence)

    def _semantic(self, role, output, ctx, evidence) -> Verdict:
        payload = {
            "role": role,
            "criteria": CRITERIA[role],
            "output": output.model_dump(),
            "facts": {k: v for k, v in ctx.items() if isinstance(v, (int, float, str, bool, type(None)))},
            "agent_input": evidence,
        }
        try:
            answer = self.check_brain.decide("semantic_check", payload, SemanticCheck)
        except Exception as exc:  # a provider outage must not stop a run that passed the real checks
            return Verdict(True, "deterministic", output=output, notes=["semantic check skipped", str(exc)[:200]])
        if not answer.ok:
            return Verdict(False, "semantic", "; ".join(answer.notes) or "rejected", output)
        return Verdict(True, "semantic", output=output, notes=answer.notes)
