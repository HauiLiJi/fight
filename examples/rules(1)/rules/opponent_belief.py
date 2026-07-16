import json
import math
from dataclasses import dataclass
from pathlib import Path


STATE_NAMES = (
    "FOCUS_BLUE_1",
    "FOCUS_BLUE_2",
    "SPLIT_ATTACK",
    "BRACKET",
    "ATTACK_SUPPORT",
    "BAIT_COUNTER",
    "DISENGAGE",
)
UNKNOWN = "UNKNOWN"
EPSILON = 1.0e-12


@dataclass(frozen=True)
class BeliefState:
    posterior: dict
    predicted_prior: dict
    map_state: str
    report_label: str
    max_probability: float
    normalized_entropy: float
    valid_feature_groups: tuple
    evidence: tuple
    update_mode: str
    insufficient_team_observation: bool


class BeliefParameterError(ValueError):
    pass


class OpponentBelief:
    def __init__(self, params_path=None):
        self._params_path = Path(params_path) if params_path else Path(__file__).with_name("belief_params.json")
        self._params = _load_params(self._params_path)
        self._states = tuple(self._params["state_names"])
        self._prior = list(self._params["initial_prior"])
        self._posterior = list(self._prior)
        self._pressure_history = []
        self._last_state = self._make_state(
            predicted=self._posterior,
            posterior=self._posterior,
            valid_groups=(),
            group_logs={},
            features={},
            update_mode="reset",
            insufficient_team_observation=True,
        )

    def reset(self):
        self._posterior = list(self._prior)
        self._pressure_history.clear()
        self._last_state = self._make_state(
            predicted=self._posterior,
            posterior=self._posterior,
            valid_groups=(),
            group_logs={},
            features={},
            update_mode="reset",
            insufficient_team_observation=True,
        )

    def update(self, observation, situation):
        predicted = self._predict()
        predicted = self._apply_feasibility_mask(predicted, observation)
        features, valid_groups, insufficient_team_observation = self._extract_features(
            observation,
            situation,
        )
        if not valid_groups:
            posterior = predicted
            group_logs = {}
            update_mode = "predict_only"
        else:
            posterior, group_logs = self._update_with_likelihood(predicted, features, valid_groups)
            posterior = self._apply_feasibility_mask(posterior, observation)
            update_mode = "bayes_update"

        self._posterior = posterior
        self._remember_pressure(observation.sim_time, features, situation)
        self._last_state = self._make_state(
            predicted=predicted,
            posterior=posterior,
            valid_groups=valid_groups,
            group_logs=group_logs,
            features=features,
            update_mode=update_mode,
            insufficient_team_observation=insufficient_team_observation,
        )
        return self._last_state

    def current_state(self):
        return self._last_state

    def _predict(self):
        transition = self._params["transition_matrix"]
        predicted = []
        for j in range(len(self._states)):
            value = 0.0
            for i, previous_probability in enumerate(self._posterior):
                value += transition[i][j] * previous_probability
            predicted.append(max(value, EPSILON))
        return _normalize_probabilities(predicted)

    def _update_with_likelihood(self, predicted, features, valid_groups):
        log_predicted = [math.log(max(value, EPSILON)) for value in predicted]
        group_logs = {}
        log_posterior = []
        for state_index, state in enumerate(self._states):
            total = log_predicted[state_index]
            for group_name in valid_groups:
                group_log = self._group_log_likelihood(group_name, state, features[group_name])
                group_logs.setdefault(group_name, {})[state] = group_log
                total += group_log
            log_posterior.append(total)
        log_norm = _log_sum_exp(log_posterior)
        posterior = [math.exp(value - log_norm) for value in log_posterior]
        return _normalize_probabilities(posterior), group_logs

    def _group_log_likelihood(self, group_name, state, values):
        group = self._params["feature_groups"][group_name]
        scaled = [
            value / scale
            for value, scale in zip(values, group["scale"])
        ]
        mean = group["means"][state]
        diff = [value - expected for value, expected in zip(scaled, mean)]
        inverse = group["_inverse_covariance"]
        quadratic = _quadratic_form(diff, inverse)
        dimension = len(diff)
        return -0.5 * (
            quadratic
            + group["_log_determinant"]
            + dimension * math.log(2.0 * math.pi)
        )

    def _apply_feasibility_mask(self, probabilities, observation):
        own_ids = list(observation.controlled_platform_ids)
        live_ids = {unit.platform_id for unit in observation.own_units}
        masked = list(probabilities)
        if len(own_ids) > 0 and own_ids[0] not in live_ids:
            masked[self._states.index("FOCUS_BLUE_1")] = 0.0
        if len(own_ids) > 1 and own_ids[1] not in live_ids:
            masked[self._states.index("FOCUS_BLUE_2")] = 0.0
        if sum(masked) <= EPSILON:
            return _normalize_probabilities(probabilities)
        return _normalize_probabilities(masked)

    def _extract_features(self, observation, situation):
        observed_tracks = [track for track in situation.tracks if track.is_observed]
        observed_target_ids = {track.target_id for track in observed_tracks}
        features = {}

        target_pressure = _target_pressure_features(observation, observed_tracks)
        if target_pressure is not None:
            features["target_pressure"] = target_pressure

        enemy_formation = _enemy_formation_features(situation, observed_target_ids)
        if enemy_formation is not None:
            features["enemy_formation"] = enemy_formation

        centroid_engagement = _centroid_engagement_features(situation, observed_tracks)
        if centroid_engagement is not None:
            features["centroid_engagement"] = centroid_engagement

        temporal_pattern = self._temporal_pattern_features(observation.sim_time, features, situation)
        if temporal_pattern is not None:
            features["temporal_pattern"] = temporal_pattern

        valid_groups = tuple(
            group_name
            for group_name in self._params["feature_groups"]
            if group_name in features
        )
        insufficient_team_observation = len(observed_target_ids) < 2
        return features, valid_groups, insufficient_team_observation

    def _temporal_pattern_features(self, sim_time, features, situation):
        if "target_pressure" not in features:
            return None
        previous = _latest_history_in_window(self._pressure_history, sim_time, 3.0, 5.0)
        current_target = _pressure_target_from_features(features["target_pressure"])
        if previous is None or current_target is None:
            return None
        if situation.trends.distance_delta_m is None or situation.enemy_depth_delta_m is None:
            return None
        if previous["enemy_depth_delta_m"] is None:
            return None
        changed = 1.0 if current_target != previous["pressure_target"] else 0.0
        return [
            changed,
            situation.trends.distance_delta_m,
            situation.enemy_depth_delta_m - previous["enemy_depth_delta_m"],
        ]

    def _remember_pressure(self, sim_time, features, situation):
        if "target_pressure" not in features:
            return
        pressure_target = _pressure_target_from_features(features["target_pressure"])
        if pressure_target is None:
            return
        self._pressure_history.append(
            {
                "sim_time": sim_time,
                "pressure_target": pressure_target,
                "enemy_depth_delta_m": situation.enemy_depth_delta_m,
            }
        )
        self._pressure_history = [
            entry
            for entry in self._pressure_history
            if sim_time - entry["sim_time"] <= 20.0
        ]

    def _make_state(
        self,
        predicted,
        posterior,
        valid_groups,
        group_logs,
        features,
        update_mode,
        insufficient_team_observation,
    ):
        posterior = _normalize_probabilities(posterior)
        predicted = _normalize_probabilities(predicted)
        max_index = max(range(len(posterior)), key=lambda index: posterior[index])
        map_state = self._states[max_index]
        max_probability = posterior[max_index]
        entropy = _normalized_entropy(posterior)
        report_label = map_state
        if (
            max_probability < self._params["min_report_probability"]
            or entropy > self._params["max_report_entropy"]
            or len(valid_groups) < self._params["min_valid_feature_groups_for_report"]
        ):
            report_label = UNKNOWN
        return BeliefState(
            posterior=dict(zip(self._states, posterior)),
            predicted_prior=dict(zip(self._states, predicted)),
            map_state=map_state,
            report_label=report_label,
            max_probability=max_probability,
            normalized_entropy=entropy,
            valid_feature_groups=tuple(valid_groups),
            evidence=tuple(self._evidence(max_index, posterior, valid_groups, group_logs, features)),
            update_mode=update_mode,
            insufficient_team_observation=insufficient_team_observation,
        )

    def _evidence(self, max_index, posterior, valid_groups, group_logs, features):
        if len(valid_groups) == 0 or len(posterior) < 2:
            return []
        sorted_indices = sorted(
            range(len(posterior)),
            key=lambda index: posterior[index],
            reverse=True,
        )
        h1 = self._states[sorted_indices[0]]
        h2 = self._states[sorted_indices[1]]
        contributions = []
        for group_name in valid_groups:
            contribution = group_logs[group_name][h1] - group_logs[group_name][h2]
            contributions.append(
                {
                    "group": group_name,
                    "contribution": contribution,
                    "message": _evidence_message(group_name, features[group_name]),
                }
            )
        return sorted(
            contributions,
            key=lambda item: abs(item["contribution"]),
            reverse=True,
        )[:3]


def _load_params(path):
    try:
        params = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise BeliefParameterError(f"cannot read belief parameters: {path}") from error
    _validate_params(params)
    return params


def _validate_params(params):
    required = {
        "state_names",
        "initial_prior",
        "transition_matrix",
        "feature_groups",
        "covariance_regularization",
        "min_report_probability",
        "max_report_entropy",
    }
    missing = sorted(required - set(params))
    if missing:
        raise BeliefParameterError(f"belief parameters missing keys: {missing}")
    states = tuple(params["state_names"])
    if states != STATE_NAMES:
        raise BeliefParameterError("belief state_names must match the seven supported tactical states")
    if UNKNOWN in states:
        raise BeliefParameterError("UNKNOWN must not be part of the hidden state set")
    _validate_probability_vector(params["initial_prior"], "initial_prior", len(states))
    transition = params["transition_matrix"]
    if len(transition) != len(states):
        raise BeliefParameterError("transition_matrix row count does not match state count")
    for row_index, row in enumerate(transition):
        _validate_probability_vector(row, f"transition_matrix[{row_index}]", len(states))
    regularization = float(params["covariance_regularization"])
    if regularization < 0.0:
        raise BeliefParameterError("covariance_regularization must be non-negative")
    for name, group in params["feature_groups"].items():
        _validate_feature_group(name, group, states, regularization)
    params.setdefault("min_valid_feature_groups_for_report", 1)


def _validate_feature_group(name, group, states, regularization):
    required = {"feature_names", "scale", "means", "covariance"}
    missing = sorted(required - set(group))
    if missing:
        raise BeliefParameterError(f"feature group {name} missing keys: {missing}")
    dimension = len(group["feature_names"])
    if dimension <= 0:
        raise BeliefParameterError(f"feature group {name} must contain at least one feature")
    if len(group["scale"]) != dimension or any(float(value) <= 0.0 for value in group["scale"]):
        raise BeliefParameterError(f"feature group {name} scale dimension or values are invalid")
    means = group["means"]
    if set(means) != set(states):
        raise BeliefParameterError(f"feature group {name} means must exist for every state")
    for state, mean in means.items():
        if len(mean) != dimension:
            raise BeliefParameterError(f"feature group {name} mean dimension mismatch for {state}")
    covariance = group["covariance"]
    if len(covariance) != dimension or any(len(row) != dimension for row in covariance):
        raise BeliefParameterError(f"feature group {name} covariance dimension mismatch")
    regularized = [
        [
            float(covariance[row][col]) + (regularization if row == col else 0.0)
            for col in range(dimension)
        ]
        for row in range(dimension)
    ]
    inverse, determinant = _invert_matrix(regularized)
    if determinant <= 0.0 or not math.isfinite(determinant):
        raise BeliefParameterError(f"feature group {name} covariance is not positive invertible after regularization")
    group["_inverse_covariance"] = inverse
    group["_log_determinant"] = math.log(determinant)


def _validate_probability_vector(values, name, expected_length):
    if len(values) != expected_length:
        raise BeliefParameterError(f"{name} dimension mismatch")
    if any((not math.isfinite(float(value))) or float(value) < 0.0 for value in values):
        raise BeliefParameterError(f"{name} must contain finite non-negative probabilities")
    total = sum(float(value) for value in values)
    if not math.isclose(total, 1.0, rel_tol=1.0e-6, abs_tol=1.0e-6):
        raise BeliefParameterError(f"{name} must sum to 1.0, got {total}")


def _target_pressure_features(observation, observed_tracks):
    own_ids = list(observation.controlled_platform_ids[:2])
    if len(own_ids) < 2:
        return None
    features = []
    for own_id in own_ids:
        pairs = [track for track in observed_tracks if track.ownship_id == own_id]
        if not pairs:
            return None
        features.extend(
            [
                min(track.pair.distance_3d_m for track in pairs),
                max(track.pair.closing_speed_mps for track in pairs),
                max(track.pair.alignment for track in pairs),
            ]
        )
    return _finite_or_none(features)


def _enemy_formation_features(situation, observed_target_ids):
    if len(observed_target_ids) < 2 or situation.enemy_formation is None:
        return None
    values = [
        situation.enemy_formation.spacing_m,
        situation.trends.enemy_spacing_delta_m,
        situation.enemy_formation.heading_delta_deg,
        situation.enemy_formation.altitude_delta_m,
        situation.enemy_depth_delta_m,
        situation.enemy_turn_rate_delta_dps,
    ]
    return _finite_or_none(values)


def _centroid_engagement_features(situation, observed_tracks):
    if not observed_tracks:
        return None
    if situation.centroid_distance_m is None or situation.centroid_closing_speed_mps is None:
        return None
    alignments = [track.pair.alignment for track in observed_tracks if track.pair.alignment is not None]
    if not alignments:
        return None
    return _finite_or_none(
        [
            situation.centroid_distance_m,
            situation.centroid_closing_speed_mps,
            sum(alignments) / len(alignments),
        ]
    )


def _pressure_target_from_features(values):
    if len(values) != 6:
        return None
    blue1_score = -values[0] / 50000.0 + values[1] / 500.0 + values[2]
    blue2_score = -values[3] / 50000.0 + values[4] / 500.0 + values[5]
    return "FOCUS_BLUE_1" if blue1_score >= blue2_score else "FOCUS_BLUE_2"


def _latest_history_in_window(history, sim_time, min_age_s, max_age_s):
    candidates = [
        entry
        for entry in history
        if min_age_s <= sim_time - entry["sim_time"] <= max_age_s
    ]
    return candidates[-1] if candidates else None


def _finite_or_none(values):
    if any(value is None or not math.isfinite(float(value)) for value in values):
        return None
    return [float(value) for value in values]


def _evidence_message(group_name, values):
    if group_name == "target_pressure":
        target = _pressure_target_from_features(values)
        label = "blue_fighter_01" if target == "FOCUS_BLUE_1" else "blue_fighter_02"
        return f"target_pressure used observed range, closing speed, and alignment pressure toward {label}"
    if group_name == "enemy_formation":
        return "enemy_formation used observed enemy spacing, heading separation, depth, and turn-rate difference"
    if group_name == "centroid_engagement":
        return "centroid_engagement used centroid range, centroid closing speed, and observed enemy alignment"
    if group_name == "temporal_pattern":
        return "temporal_pattern used recent pressure-target change, range trend, and depth trend"
    return f"{group_name} contributed from valid observed features"


def _normalize_probabilities(values):
    cleaned = [
        max(float(value), EPSILON) if math.isfinite(float(value)) else EPSILON
        for value in values
    ]
    total = sum(cleaned)
    if total <= EPSILON:
        cleaned = [1.0 for _ in cleaned]
        total = sum(cleaned)
    normalized = [value / total for value in cleaned]
    correction = 1.0 - sum(normalized)
    normalized[-1] += correction
    return normalized


def _normalized_entropy(probabilities):
    count = len(probabilities)
    if count <= 1:
        return 0.0
    entropy = -sum(
        probability * math.log(max(probability, EPSILON))
        for probability in probabilities
    )
    return entropy / math.log(count)


def _log_sum_exp(values):
    maximum = max(values)
    if not math.isfinite(maximum):
        return math.log(1.0 / len(values))
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def _quadratic_form(vector, matrix):
    total = 0.0
    for row_index, row_value in enumerate(vector):
        for col_index, col_value in enumerate(vector):
            total += row_value * matrix[row_index][col_index] * col_value
    return total


def _invert_matrix(matrix):
    size = len(matrix)
    work = [
        [float(value) for value in row]
        + [1.0 if row_index == col_index else 0.0 for col_index in range(size)]
        for row_index, row in enumerate(matrix)
    ]
    determinant = 1.0
    for col in range(size):
        pivot = max(range(col, size), key=lambda row: abs(work[row][col]))
        pivot_value = work[pivot][col]
        if abs(pivot_value) <= EPSILON:
            raise BeliefParameterError("covariance matrix is singular")
        if pivot != col:
            work[col], work[pivot] = work[pivot], work[col]
            determinant *= -1.0
        pivot_value = work[col][col]
        determinant *= pivot_value
        for item in range(2 * size):
            work[col][item] /= pivot_value
        for row in range(size):
            if row == col:
                continue
            factor = work[row][col]
            for item in range(2 * size):
                work[row][item] -= factor * work[col][item]
    inverse = [row[size:] for row in work]
    return inverse, determinant
