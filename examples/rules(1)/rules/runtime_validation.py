import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from . import config
from .opponent_belief import STATE_NAMES, UNKNOWN


@dataclass(frozen=True)
class ValidationReport:
    valid: bool
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    checked_files: list = field(default_factory=list)


def validate_runtime_configuration(base_dir=None) -> ValidationReport:
    root = Path(base_dir) if base_dir else Path(__file__).parent
    errors = []
    warnings = []
    checked_files = ["config.py"]
    files = {
        "belief": root / "belief_params.json",
        "scorer": root / "scorer_params.json",
        "llm": root / "llm_params.json",
        "manager": root / "strategy_manager_params.json",
        "diagnostics": root / "diagnostics_params.json",
    }
    data = {}
    for key, path in files.items():
        checked_files.append(path.name)
        try:
            data[key] = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            errors.append(f"{path.name} does not exist")
        except json.JSONDecodeError as error:
            errors.append(f"{path.name} is not valid JSON: {error}")
        except OSError as error:
            errors.append(f"{path.name} cannot be read: {error}")
    if errors:
        return ValidationReport(False, errors, warnings, checked_files)

    _validate_config(errors, warnings)
    _validate_belief(data["belief"], errors)
    _validate_scorer(data["scorer"], errors)
    _validate_llm(data["llm"], errors)
    _validate_manager(data["manager"], errors, warnings)
    _validate_diagnostics(data["diagnostics"], errors)
    _validate_cross_constraints(data["llm"], data["manager"], errors, warnings)
    return ValidationReport(not errors, errors, warnings, checked_files)


def assert_runtime_configuration_valid(base_dir=None):
    report = validate_runtime_configuration(base_dir)
    if not report.valid:
        detail = "; ".join(report.errors)
        raise RuntimeError(f"Baiyang runtime configuration is invalid: {detail}")
    return report


def _validate_config(errors, warnings):
    numeric_positive = {
        "EARTH_RADIUS_M": config.EARTH_RADIUS_M,
        "FIRE_RANGE_M": config.FIRE_RANGE_M,
        "FIRE_COOLDOWN_S": config.FIRE_COOLDOWN_S,
        "HISTORY_WINDOW_S": config.HISTORY_WINDOW_S,
        "TRACK_COASTING_TIMEOUT_S": config.TRACK_COASTING_TIMEOUT_S,
    }
    for name, value in numeric_positive.items():
        if not _positive(value):
            errors.append(f"config.{name} must be positive")
    for name in (
        "ENGAGEMENT_HOT_RANGE_M",
        "ENGAGEMENT_FLANK_RANGE_M",
        "ENGAGEMENT_COLD_RANGE_M",
        "ENGAGEMENT_HOT_ASPECT_MAX_DEG",
        "ENGAGEMENT_FLANK_ASPECT_MAX_DEG",
        "ENGAGEMENT_NORMAL_HEADING_ERROR_MAX_DEG",
        "ENGAGEMENT_COUNTER_HEADING_ERROR_MAX_DEG",
        "ENGAGEMENT_MAX_PENDING_SHOTS_PER_SHOOTER",
        "ENGAGEMENT_MAX_TEAM_PENDING_SHOTS_PER_TARGET",
        "ENGAGEMENT_PENDING_SHOT_TIMEOUT_S",
    ):
        if not _positive(getattr(config, name)):
            errors.append(f"config.{name} must be positive")
    if not (
        config.ENGAGEMENT_HOT_RANGE_M
        >= config.ENGAGEMENT_FLANK_RANGE_M
        >= config.ENGAGEMENT_COLD_RANGE_M
        > 0.0
    ):
        errors.append("engagement ranges must satisfy hot >= flank >= cold > 0")
    if not (
        0.0
        < config.ENGAGEMENT_HOT_ASPECT_MAX_DEG
        < config.ENGAGEMENT_FLANK_ASPECT_MAX_DEG
        <= 180.0
    ):
        errors.append("engagement aspect thresholds must satisfy 0 < hot < flank <= 180")
    for name in ("ENGAGEMENT_NORMAL_HEADING_ERROR_MAX_DEG", "ENGAGEMENT_COUNTER_HEADING_ERROR_MAX_DEG"):
        if not 0.0 < float(getattr(config, name)) <= 180.0:
            errors.append(f"config.{name} must be in (0, 180]")
    for name in ("ENGAGEMENT_MAX_PENDING_SHOTS_PER_SHOOTER", "ENGAGEMENT_MAX_TEAM_PENDING_SHOTS_PER_TARGET"):
        if int(getattr(config, name)) < 1:
            errors.append(f"config.{name} must be at least 1")
    if config.TRACK_COASTING_TIMEOUT_S >= config.HISTORY_WINDOW_S:
        warnings.append("TRACK_COASTING_TIMEOUT_S is not shorter than HISTORY_WINDOW_S")
    if config.NO_TARGET_MIN_ALTITUDE_M > config.NO_TARGET_MAX_ALTITUDE_M:
        errors.append("no-target altitude range is inverted")
    if config.TARGET_MIN_ALTITUDE_M > config.TARGET_MAX_ALTITUDE_M:
        errors.append("target altitude range is inverted")
    for name in ("CRUISE_MACH", "CHASE_FAR_MACH", "CHASE_NEAR_MACH", "DEFEND_MACH", "PRESS_MACH", "DISENGAGE_MACH"):
        value = getattr(config, name)
        if not 0.2 <= float(value) <= 2.0:
            errors.append(f"config.{name} must be within action Mach range")


def _validate_belief(params, errors):
    _require_source(params, "belief_params.json", errors)
    states = tuple(params.get("state_names", ()))
    if states != STATE_NAMES:
        errors.append("belief_params.json state_names must match the seven code states")
    if UNKNOWN in states:
        errors.append("belief_params.json must not include UNKNOWN as a hidden state")
    count = len(STATE_NAMES)
    _probability_vector(params.get("initial_prior"), count, "belief initial_prior", errors)
    matrix = params.get("transition_matrix")
    if not isinstance(matrix, list) or len(matrix) != count:
        errors.append("belief transition_matrix row count is invalid")
    else:
        for index, row in enumerate(matrix):
            _probability_vector(row, count, f"belief transition_matrix[{index}]", errors)
    reg = params.get("covariance_regularization")
    if reg is None or float(reg) < 0.0:
        errors.append("belief covariance_regularization must be non-negative")
    for key in ("min_report_probability", "max_report_entropy"):
        if not _between(params.get(key), 0.0, 1.0):
            errors.append(f"belief {key} must be in [0, 1]")
    groups = params.get("feature_groups", {})
    if not isinstance(groups, dict) or not groups:
        errors.append("belief feature_groups must be a non-empty object")
        return
    for name, group in groups.items():
        features = group.get("feature_names", [])
        dimension = len(features)
        if dimension <= 0:
            errors.append(f"belief feature group {name} has no features")
            continue
        if len(group.get("scale", [])) != dimension or any(not _positive(value) for value in group.get("scale", [])):
            errors.append(f"belief feature group {name} scale is invalid")
        means = group.get("means", {})
        if set(means) != set(STATE_NAMES):
            errors.append(f"belief feature group {name} means must cover all states")
        for state, values in means.items():
            if len(values) != dimension:
                errors.append(f"belief feature group {name} mean dimension mismatch for {state}")
        covariance = group.get("covariance", [])
        if len(covariance) != dimension or any(len(row) != dimension for row in covariance):
            errors.append(f"belief feature group {name} covariance dimension mismatch")


def _validate_scorer(params, errors):
    _require_source(params, "scorer_params.json", errors)
    prediction = params.get("prediction", {})
    for key in (
        "horizon_s",
        "dt_s",
        "max_turn_rate_deg_s",
        "max_climb_rate_mps",
        "max_descent_rate_mps",
        "max_acceleration_mps2",
        "min_planning_speed_mps",
        "max_planning_speed_mps",
        "mach_to_mps",
    ):
        if not _positive(prediction.get(key)):
            errors.append(f"scorer prediction.{key} must be positive")
    if prediction.get("min_planning_speed_mps", 0) >= prediction.get("max_planning_speed_mps", 0):
        errors.append("scorer min_planning_speed_mps must be less than max_planning_speed_mps")
    utility_weights = params.get("utility_weights", {})
    expected_utility = {
        "attack_opportunity",
        "own_fire_opportunity",
        "survivability",
        "terminal_survival",
        "exchange_value",
        "coordination",
        "counter_effect",
        "local_advantage",
        "bracket_risk",
        "unpressed_enemy_risk",
        "enemy_fire_window_risk",
        "separation_risk",
        "ammo_waste",
        "duplicate_attack_waste",
    }
    if set(utility_weights) != expected_utility:
        errors.append("scorer utility_weights keys are incomplete")
    for key, value in utility_weights.items():
        if value is None or float(value) < 0.0:
            errors.append(f"scorer utility weight {key} must be non-negative")
    aggregation = params.get("aggregation", {})
    for key in ("expected_weight", "worst_case_weight", "switch_cost_weight"):
        if key not in aggregation or float(aggregation[key]) < 0.0:
            errors.append(f"scorer aggregation.{key} must be non-negative")
    if not math.isclose(sum(float(aggregation.get(key, 0.0)) for key in ("expected_weight", "worst_case_weight", "switch_cost_weight")), 1.0, rel_tol=1e-6, abs_tol=1e-6):
        errors.append("scorer aggregation weights must sum to 1")
    thresholds = params.get("risk_thresholds", {})
    if not _between(thresholds.get("strong_alignment"), -1.0, 1.0):
        errors.append("scorer risk_thresholds.strong_alignment must be in [-1, 1]")
    for key in ("dangerous_distance_m", "dangerous_closure_mps", "formation_min_distance_m", "formation_max_distance_m"):
        if not _positive(thresholds.get(key)):
            errors.append(f"scorer risk_thresholds.{key} must be positive")
    if thresholds.get("formation_min_distance_m", 0) >= thresholds.get("formation_max_distance_m", 0):
        errors.append("scorer formation distance thresholds are inverted")
    fire = params.get("fire_window", {})
    for key in (
        "range_m",
        "near_range_m",
        "closing_scale_mps",
        "time_to_fire_scale_s",
        "time_to_impact_scale_s",
        "hit_probability_scale",
        "high_pending_hit_probability",
        "threat_prediction_horizon_s",
        "predicted_launch_probability_threshold",
        "prelaunch_hit_probability_scale",
        "time_to_fire_discount",
        "heading_error_max_deg",
    ):
        if not _positive(fire.get(key)):
            errors.append(f"scorer fire_window.{key} must be positive")
    if not _between(fire.get("heading_error_max_deg"), 0.0, 180.0):
        errors.append("scorer fire_window.heading_error_max_deg must be in [0, 180]")
    for key in ("alignment_min", "alignment_strong", "coasting_multiplier", "pressed_threat_discount"):
        if not _between(fire.get(key), 0.0, 1.0):
            errors.append(f"scorer fire_window.{key} must be in [0, 1]")
    for key in ("predicted_launch_probability_threshold", "prelaunch_hit_probability_scale"):
        if not _between(fire.get(key), 0.0, 1.0):
            errors.append(f"scorer fire_window.{key} must be in [0, 1]")
    if fire.get("near_range_m", 0) >= fire.get("range_m", 0):
        errors.append("scorer fire_window.near_range_m must be below range_m")


def _validate_llm(params, errors):
    _require_source(params, "llm_params.json", errors)
    if params.get("provider") != "openai_compatible":
        errors.append("llm provider must be openai_compatible")
    for key in ("request_timeout_s", "connect_timeout_s"):
        if not _positive(params.get(key)):
            errors.append(f"llm {key} must be positive")
    if int(params.get("max_candidates", 0)) < 2:
        errors.append("llm max_candidates must be at least 2")
    if int(params.get("min_valid_for_steps", 0)) <= 0:
        errors.append("llm min_valid_for_steps must be positive")
    if int(params.get("min_valid_for_steps", 0)) > int(params.get("max_valid_for_steps", 0)):
        errors.append("llm valid_for_steps range is inverted")
    if int(params.get("max_response_chars", 0)) <= 0:
        errors.append("llm max_response_chars must be positive")
    if int(params.get("max_inflight_requests", 0)) != 1:
        errors.append("llm max_inflight_requests must be 1")
    if int(params.get("llm_ready_max_age_steps", 0)) <= 0:
        errors.append("llm llm_ready_max_age_steps must be positive")
    if int(params.get("retry_backoff_initial_steps", 0)) <= 0:
        errors.append("llm retry_backoff_initial_steps must be positive")
    if float(params.get("retry_backoff_multiplier", 0.0)) < 1.0:
        errors.append("llm retry_backoff_multiplier must be at least 1")
    if int(params.get("retry_backoff_max_steps", 0)) < int(params.get("retry_backoff_initial_steps", 0)):
        errors.append("llm retry_backoff_max_steps must be >= retry_backoff_initial_steps")


def _validate_manager(params, errors, warnings):
    _require_source(params, "strategy_manager_params.json", errors)
    for key in (
        "review_interval_steps",
        "minimum_hold_steps",
        "belief_shift_required_steps",
        "score_drop_required_steps",
        "llm_wait_max_steps",
        "llm_preplan_margin_steps",
        "llm_primary_wait_steps",
        "decision_history_size",
        "leader_required_reviews",
        "strong_event_required_reviews",
        "belief_label_stable_steps",
        "score_degrade_required_reviews",
        "risk_event_cooldown_steps",
    ):
        if int(params.get(key, 0)) <= 0:
            errors.append(f"manager {key} must be positive")
    for key in (
        "belief_tv_threshold",
        "emergency_alignment",
        "enemy_fire_window_enter_threshold",
        "enemy_fire_window_exit_threshold",
        "unpressed_enemy_enter_threshold",
        "unpressed_enemy_exit_threshold",
    ):
        if not _between(params.get(key), 0.0, 1.0):
            errors.append(f"manager {key} must be in [0, 1]")
    for key in (
        "switch_score_advantage",
        "switch_absolute_advantage",
        "switch_relative_advantage",
        "score_delta_epsilon",
        "worst_case_degradation_limit",
        "target_lost_timeout_s",
        "emergency_distance_m",
        "emergency_closure_mps",
        "role_swap_advantage",
    ):
        if float(params.get(key, -1.0)) < 0.0:
            errors.append(f"manager {key} must be non-negative")
    if not 0.0 < float(params.get("strong_event_threshold_multiplier", 0.0)) <= 1.0:
        errors.append("manager strong_event_threshold_multiplier must be in (0, 1]")
    if params.get("leader_identity_mode") not in {"semantic", "plan_id"}:
        errors.append("manager leader_identity_mode must be semantic or plan_id")
    if str(params.get("candidate_policy", "RULE_PRIMARY")).upper() not in {"LLM_ONLY", "LLM_PRIMARY", "RULE_PRIMARY", "RULE_ONLY"}:
        errors.append("manager candidate_policy must be LLM_ONLY, LLM_PRIMARY, RULE_PRIMARY, or RULE_ONLY")
    if float(params.get("enemy_fire_window_exit_threshold", 0.0)) >= float(params.get("enemy_fire_window_enter_threshold", 0.0)):
        errors.append("manager enemy_fire_window_exit_threshold must be below enter threshold")
    if float(params.get("unpressed_enemy_exit_threshold", 0.0)) >= float(params.get("unpressed_enemy_enter_threshold", 0.0)):
        errors.append("manager unpressed_enemy_exit_threshold must be below enter threshold")
    if float(params.get("target_lost_timeout_s", 0.0)) < config.TRACK_COASTING_TIMEOUT_S:
        warnings.append("manager target_lost_timeout_s is shorter than TRACK_COASTING_TIMEOUT_S")


def _validate_diagnostics(params, errors):
    _require_source(params, "diagnostics_params.json", errors)
    for key in ("memory_history_size", "flush_every_steps", "max_error_text_chars", "max_rationale_items", "max_ranked_plans_logged", "max_candidates_logged"):
        if int(params.get(key, 0)) <= 0:
            errors.append(f"diagnostics {key} must be positive")
    for key in ("enabled_by_default", "record_candidate_breakdown"):
        if not isinstance(params.get(key), bool):
            errors.append(f"diagnostics {key} must be boolean")


def _validate_cross_constraints(llm, manager, errors, warnings):
    if float(config.FIRE_COOLDOWN_S) < float(config.TRACK_COASTING_TIMEOUT_S):
        warnings.append("FIRE_COOLDOWN_S is shorter than TRACK_COASTING_TIMEOUT_S")
    if float(manager.get("target_lost_timeout_s", 0.0)) > float(config.HISTORY_WINDOW_S):
        warnings.append("manager target_lost_timeout_s exceeds memory history window")


def _require_source(params, filename, errors):
    source = params.get("parameter_source")
    if source != "expert_initialized":
        errors.append(f"{filename} parameter_source must be expert_initialized")


def _probability_vector(values, length, name, errors):
    if not isinstance(values, list) or len(values) != length:
        errors.append(f"{name} dimension mismatch")
        return
    if any((not _finite(value)) or float(value) < 0.0 for value in values):
        errors.append(f"{name} must contain finite non-negative values")
        return
    total = sum(float(value) for value in values)
    if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        errors.append(f"{name} must sum to 1.0")


def _positive(value):
    return _finite(value) and float(value) > 0.0


def _between(value, low, high):
    return _finite(value) and low <= float(value) <= high


def _finite(value):
    try:
        return math.isfinite(float(value))
    except Exception:
        return False
