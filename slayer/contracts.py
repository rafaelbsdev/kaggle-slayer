"""What each specialist agent is allowed to say.

Every agent answers with exactly one of these schemas. Fields are flat and closed on purpose: an
agent can only choose among things the orchestrator knows how to run (see decision 0003).
"""
import hashlib
import json
from typing import Literal

from pydantic import BaseModel, Field

MAX_JOINT_SCAN_COLUMNS = 40  # ponytail: pairs grow with the square of the columns; past this the scan would dominate a run


class CompetitionBrief(BaseModel):
    """Scout: can this system compete here at all?"""
    supported: bool
    summary: str = Field(min_length=10)
    risks: list[str] = []


class ValidationPlan(BaseModel):
    """Validation Strategist: how CV must mimic the leaderboard."""
    scheme: Literal["stratified_kfold", "time_kfold"]
    n_splits: int = Field(ge=2, le=10)
    order_by: Literal["row_order", "id", "drift_clock"] = "row_order"
    rationale: str = Field(min_length=10)


class FeaturePlan(BaseModel):
    """Feature Engineer: which transformations to apply."""
    drop_drifting_top_k: int = Field(default=0, ge=0)
    row_stats: bool = False
    frequency_encode_max_cardinality: int = Field(default=0, ge=0)
    target_encode_max_cardinality: int = Field(default=0, ge=0)
    target_encode_bags: int = Field(default=1, ge=1, le=5)
    target_encode_inner_folds: int = Field(default=5, ge=2, le=10)
    target_encode_smoothing: float = Field(default=20.0, ge=1.0, le=100000.0)
    number_keys: bool = False
    number_tokens: bool = False
    joint_keys: int = Field(default=0, ge=0, le=20)
    joint_by_value: bool = False
    key_counts: bool = False
    shared_token_rates: bool = False
    logistic_tolerance: float = Field(default=1e-4, ge=1e-7, le=1e-2)
    logistic_start_penalty: float = Field(default=1.0, ge=0.01, le=1000.0)
    rationale: str = Field(min_length=10)


class ModelParams(BaseModel):
    learning_rate: float = Field(ge=0.003, le=0.3)
    max_depth: int = Field(ge=2, le=12)
    num_leaves: int = Field(ge=4, le=512)
    subsample: float = Field(ge=0.3, le=1.0)
    colsample: float = Field(ge=0.02, le=1.0)
    reg_lambda: float = Field(ge=0.0, le=100.0)
    min_child: int = Field(ge=1, le=5000)


class ModelPlan(BaseModel):
    family: Literal["lightgbm", "xgboost", "catboost", "logistic", "logistic_boost"]
    params: ModelParams
    recency_half_life: float | None = Field(default=None, ge=0.05, le=5.0)
    seed: int = Field(ge=0)
    hypothesis: str = Field(min_length=10)


class ModelBatch(BaseModel):
    """Model Tuner: the next experiments to run."""
    models: list[ModelPlan] = Field(min_length=1, max_length=4)


class SubmissionDecision(BaseModel):
    """Submission Critic: spend a daily submission or not."""
    action: Literal["submit", "hold"]
    candidate_id: str
    reason: str = Field(min_length=10)


class LeaderboardReport(BaseModel):
    """Leaderboard Analyst: what the score teaches the next round."""
    diagnosis: str = Field(min_length=10)
    cv_lb_gap_cited: float | None = None
    lessons: list[str] = Field(default=[], max_length=5)


class SemanticCheck(BaseModel):
    """Answer of the cheap model in the verifier's third layer."""
    ok: bool
    notes: list[str] = []


CONTRACTS: dict[str, type[BaseModel]] = {
    "scout": CompetitionBrief,
    "validation": ValidationPlan,
    "features": FeaturePlan,
    "tuner": ModelBatch,
    "critic": SubmissionDecision,
    "analyst": LeaderboardReport,
}


def experiment_id(validation: ValidationPlan, features: FeaturePlan, model: ModelPlan) -> str:
    """Identity of an experiment: what it runs, never how an agent explained it."""
    spec = {
        "validation": validation.model_dump(exclude={"rationale"}),
        "features": features.model_dump(exclude={"rationale"}),
        "model": model.model_dump(exclude={"hypothesis"}),
    }
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]
