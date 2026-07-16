import hashlib
import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from queue import Empty, Queue
import threading
import time
import urllib.error
import urllib.request

from .opponent_belief import STATE_NAMES
from .chain_logger import log_event, plan_summary
from .strategy import (
    PlanSource,
    Role,
    StrategyMode,
    Tactic,
    TeamPlan,
    validate_plan,
)


class LLMStatus(Enum):
    DISABLED = "DISABLED"
    IDLE = "IDLE"
    PENDING = "PENDING"
    READY = "READY"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    STALE = "STALE"


@dataclass(frozen=True)
class LLMGenerationResult:
    request_id: object
    status: LLMStatus
    request_step: object
    completed_step: object = None
    summary_hash: object = None
    request_plan_id: object = None
    response_age_steps: object = None
    candidates: tuple = ()
    candidate_ids: tuple = ()
    rejected_candidates: tuple = ()
    error: object = None
    latency_ms: float = 0.0
    raw_response_digest: object = None
    consumed: bool = False
    stale_reasons: tuple = ()


@dataclass(frozen=True)
class _WorkerResult:
    request_id: str
    request_step: int
    generation: int
    summary_hash: str
    request_plan_id: object
    raw_response: object = None
    error: object = None
    timeout: bool = False
    latency_ms: float = 0.0


class TacticalSummaryBuilder:
    def build(
        self,
        observation,
        memory_snapshot,
        situation,
        belief,
        current_plan,
        current_plan_score,
        rule_candidates,
    ):
        controlled_ids = list(observation.controlled_platform_ids)
        own_units = [
            unit
            for unit in observation.own_units
            if unit.platform_id in set(controlled_ids)
        ]
        return {
            "step": observation.step_index,
            "sim_time": observation.sim_time,
            "own_slots": [
                {
                    "slot": index,
                    "platform_id": platform_id,
                }
                for index, platform_id in enumerate(controlled_ids)
            ],
            "own_units": [
                {
                    "platform_id": unit.platform_id,
                    "latitude": unit.position.latitude,
                    "longitude": unit.position.longitude,
                    "altitude_m": unit.position.altitude_m,
                    "speed_mps": _speed_mps(unit.velocity),
                    "heading_deg": unit.attitude.heading_deg,
                    "aam_medium_remaining": _weapon_remaining(unit, "aam_medium"),
                }
                for unit in own_units
            ],
            "enemy_tracks": [
                _track_summary(target_id, track)
                for target_id, track in sorted(memory_snapshot.tracks.items())
                if track.status in {"OBSERVED", "COASTING"}
            ],
            "pair_geometry": [
                {
                    "ownship_id": track.ownship_id,
                    "target_id": track.target_id,
                    "is_observed": track.is_observed,
                    "distance_3d_m": track.pair.distance_3d_m,
                    "closing_speed_mps": track.pair.closing_speed_mps,
                    "enemy_alignment": track.pair.alignment,
                    "own_alignment": track.pair.own_alignment,
                }
                for track in situation.tracks
            ],
            "formation": _formation_summary(situation),
            "belief": {
                "posterior": dict(getattr(belief, "posterior", {})),
                "report_label": getattr(belief, "report_label", None),
                "normalized_entropy": getattr(belief, "normalized_entropy", None),
            },
            "current_plan": _plan_summary(current_plan),
            "current_plan_score": _score_summary(current_plan_score),
            "allowed": {
                "strategy_modes": [mode.value for mode in StrategyMode],
                "roles": [role.value for role in Role],
                "tactics": [tactic.value for tactic in Tactic],
                "platform_ids": controlled_ids,
                "target_ids": [
                    target_id
                    for target_id, track in sorted(memory_snapshot.tracks.items())
                    if track.status in {"OBSERVED", "COASTING"}
                ],
            },
            "rule_candidates": [
                _plan_summary(plan)
                for plan in rule_candidates
            ],
        }


class LLMCommander:
    def __init__(self, params_path=None, transport=None):
        self._params_path = Path(params_path) if params_path else Path(__file__).with_name("llm_params.json")
        self._params = json.loads(self._params_path.read_text(encoding="utf-8"))
        if self._params.get("provider") != "openai_compatible":
            raise ValueError("llm provider must be openai_compatible")
        self._transport = transport or self._default_transport
        self._summary_builder = TacticalSummaryBuilder()
        self._result_queue = Queue()
        self._lock = threading.Lock()
        self._thread = None
        self._inflight_request_id = None
        self._last_result = self._disabled_result() if not self._enabled() else self._idle_result()
        self._ready_result = None
        self._generation = 0
        self._request_count = 0
        self._response_count = 0
        self._consumed_count = 0
        self._discarded_count = 0
        self._failure_event_count = 0
        self._timeout_event_count = 0
        self._stale_event_count = 0
        self._discard_reasons = {}
        self._status_step_counts = {status.value: 0 for status in LLMStatus}
        self._consecutive_failures = 0
        self._last_failure_step = None
        self._next_retry_step = 0
        self._retry_backoff_steps = int(self._params.get("retry_backoff_initial_steps", 10))

    def reset(self):
        with self._lock:
            self._generation += 1
            self._inflight_request_id = None
            self._ready_result = None
            self._last_result = self._disabled_result() if not self._enabled() else self._idle_result()
            self._drain_queue()
            self._request_count = 0
            self._response_count = 0
            self._consumed_count = 0
            self._discarded_count = 0
            self._failure_event_count = 0
            self._timeout_event_count = 0
            self._stale_event_count = 0
            self._discard_reasons = {}
            self._status_step_counts = {status.value: 0 for status in LLMStatus}
            self._consecutive_failures = 0
            self._last_failure_step = None
            self._next_retry_step = 0
            self._retry_backoff_steps = int(self._params.get("retry_backoff_initial_steps", 10))

    def close(self):
        self.reset()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=0.01)

    def build_summary(
        self,
        observation,
        memory_snapshot,
        situation,
        belief,
        current_plan,
        current_plan_score,
        rule_candidates,
    ):
        return self._summary_builder.build(
            observation,
            memory_snapshot,
            situation,
            belief,
            current_plan,
            current_plan_score,
            rule_candidates,
        )

    def submit(self, summary, request_step):
        if not self._enabled():
            self._last_result = self._disabled_result()
            return None
        with self._lock:
            if self._inflight_request_id is not None:
                return None
            if int(request_step) < self._next_retry_step:
                return None
            summary_hash = _summary_hash(summary)
            request_id = _request_id_from_hash(summary_hash, request_step, self._generation)
            request_plan_id = _summary_plan_id(summary)
            self._inflight_request_id = request_id
            self._request_count += 1
            generation = self._generation
            thread = threading.Thread(
                target=self._run_request,
                args=(request_id, int(request_step), generation, summary, summary_hash, request_plan_id),
                daemon=True,
            )
            self._thread = thread
            self._last_result = LLMGenerationResult(
                request_id=request_id,
                status=LLMStatus.PENDING,
                request_step=request_step,
                summary_hash=summary_hash,
                request_plan_id=request_plan_id,
            )
            log_event(
                "llm_request",
                {
                    "request_id": request_id,
                    "request_step": int(request_step),
                    "generation": generation,
                    "summary_hash": summary_hash,
                    "request_plan_id": request_plan_id,
                    "enabled": True,
                },
            )
            thread.start()
            return request_id

    def poll(self, current_step):
        if not self._enabled():
            self._last_result = self._disabled_result()
            self._remember_status(self._last_result.status)
            return self._last_result
        try:
            worker_result = self._result_queue.get_nowait()
        except Empty:
            if self._inflight_request_id is not None:
                self._last_result = LLMGenerationResult(
                    request_id=self._inflight_request_id,
                    status=LLMStatus.PENDING,
                    request_step=self._last_result.request_step,
                    summary_hash=self._last_result.summary_hash,
                    request_plan_id=self._last_result.request_plan_id,
                )
            self._remember_status(self._last_result.status)
            return self._last_result

        with self._lock:
            if worker_result.generation != self._generation:
                return self._last_result
            self._inflight_request_id = None
            self._response_count += 1
        result = self._convert_worker_result(worker_result, current_step)
        self._last_result = result
        self._ready_result = result if result.status == LLMStatus.READY else None
        self._remember_status(result.status)
        return result

    def has_inflight_request(self):
        return self._inflight_request_id is not None

    def consume_ready_result(self):
        result = self._ready_result
        self._ready_result = None
        if result is not None:
            self._consumed_count += 1
            self._last_result = LLMGenerationResult(
                request_id=result.request_id,
                status=LLMStatus.IDLE,
                request_step=result.request_step,
                completed_step=result.completed_step,
                summary_hash=result.summary_hash,
                request_plan_id=result.request_plan_id,
                response_age_steps=result.response_age_steps,
                candidates=result.candidates,
                candidate_ids=result.candidate_ids,
                rejected_candidates=result.rejected_candidates,
                error=result.error,
                latency_ms=result.latency_ms,
                raw_response_digest=result.raw_response_digest,
                consumed=True,
                stale_reasons=result.stale_reasons,
            )
        return result

    def discard_ready_result(self, stale_reasons=(), reason=None):
        result = self._ready_result
        self._ready_result = None
        if result is not None:
            self._discarded_count += 1
            discard_reason = reason or (stale_reasons[0] if stale_reasons else "discarded")
            self._discard_reasons[discard_reason] = self._discard_reasons.get(discard_reason, 0) + 1
            self._last_result = LLMGenerationResult(
                request_id=result.request_id,
                status=LLMStatus.IDLE,
                request_step=result.request_step,
                completed_step=result.completed_step,
                summary_hash=result.summary_hash,
                request_plan_id=result.request_plan_id,
                response_age_steps=result.response_age_steps,
                candidates=result.candidates,
                candidate_ids=result.candidate_ids,
                rejected_candidates=result.rejected_candidates,
                error=result.error,
                latency_ms=result.latency_ms,
                raw_response_digest=result.raw_response_digest,
                consumed=False,
                stale_reasons=tuple(stale_reasons),
            )
        return result

    def stats(self):
        return {
            "request_count": self._request_count,
            "response_count": self._response_count,
            "consumed_count": self._consumed_count,
            "discarded_count": self._discarded_count,
            "discard_reasons": dict(self._discard_reasons),
            "failure_event_count": self._failure_event_count,
            "timeout_event_count": self._timeout_event_count,
            "stale_event_count": self._stale_event_count,
            "stale_count": self._stale_event_count,
            "status_step_counts": dict(self._status_step_counts),
            "consecutive_failures": self._consecutive_failures,
            "last_failure_step": self._last_failure_step,
            "next_retry_step": self._next_retry_step,
            "retry_backoff_steps": self._retry_backoff_steps,
        }

    def _run_request(self, request_id, request_step, generation, summary, summary_hash, request_plan_id):
        start = time.perf_counter()
        try:
            raw = self._transport(self._build_payload(summary), self._params)
            self._result_queue.put(
                _WorkerResult(
                    request_id=request_id,
                    request_step=request_step,
                    generation=generation,
                    summary_hash=summary_hash,
                    request_plan_id=request_plan_id,
                    raw_response=raw,
                    latency_ms=(time.perf_counter() - start) * 1000.0,
                )
            )
        except TimeoutError as error:
            self._result_queue.put(
                _WorkerResult(
                    request_id=request_id,
                    request_step=request_step,
                    generation=generation,
                    summary_hash=summary_hash,
                    request_plan_id=request_plan_id,
                    error=str(error),
                    timeout=True,
                    latency_ms=(time.perf_counter() - start) * 1000.0,
                )
            )
        except Exception as error:
            self._result_queue.put(
                _WorkerResult(
                    request_id=request_id,
                    request_step=request_step,
                    generation=generation,
                    summary_hash=summary_hash,
                    request_plan_id=request_plan_id,
                    error=str(error),
                    latency_ms=(time.perf_counter() - start) * 1000.0,
                )
            )

    def _convert_worker_result(self, worker_result, current_step):
        response_age_steps = int(current_step) - worker_result.request_step
        if int(current_step) - worker_result.request_step > int(self._params["stale_after_steps"]):
            self._stale_event_count += 1
            self._register_failure(current_step)
            self._log_worker_response(worker_result, current_step, "stale", error="result is stale")
            return LLMGenerationResult(
                request_id=worker_result.request_id,
                status=LLMStatus.STALE,
                request_step=worker_result.request_step,
                completed_step=current_step,
                summary_hash=worker_result.summary_hash,
                request_plan_id=worker_result.request_plan_id,
                response_age_steps=response_age_steps,
                error="result is stale",
                latency_ms=worker_result.latency_ms,
                raw_response_digest=_digest(worker_result.raw_response),
                stale_reasons=("response_age_exceeded",),
            )
        if worker_result.timeout:
            self._timeout_event_count += 1
            self._register_failure(current_step)
            self._log_worker_response(worker_result, current_step, "timeout", error=worker_result.error)
            return LLMGenerationResult(
                request_id=worker_result.request_id,
                status=LLMStatus.TIMEOUT,
                request_step=worker_result.request_step,
                completed_step=current_step,
                summary_hash=worker_result.summary_hash,
                request_plan_id=worker_result.request_plan_id,
                response_age_steps=response_age_steps,
                error=worker_result.error,
                latency_ms=worker_result.latency_ms,
            )
        if worker_result.error is not None:
            self._failure_event_count += 1
            self._register_failure(current_step)
            self._log_worker_response(worker_result, current_step, "transport_error", error=worker_result.error)
            return LLMGenerationResult(
                request_id=worker_result.request_id,
                status=LLMStatus.FAILED,
                request_step=worker_result.request_step,
                completed_step=current_step,
                summary_hash=worker_result.summary_hash,
                request_plan_id=worker_result.request_plan_id,
                response_age_steps=response_age_steps,
                error=worker_result.error,
                latency_ms=worker_result.latency_ms,
            )
        try:
            candidates, rejected = self._parse_candidates(
                worker_result.request_id,
                worker_result.request_step,
                worker_result.raw_response,
            )
        except Exception as error:
            self._failure_event_count += 1
            self._register_failure(current_step)
            self._log_worker_response(worker_result, current_step, "parse_failed", error=str(error))
            return LLMGenerationResult(
                request_id=worker_result.request_id,
                status=LLMStatus.FAILED,
                request_step=worker_result.request_step,
                completed_step=current_step,
                summary_hash=worker_result.summary_hash,
                request_plan_id=worker_result.request_plan_id,
                response_age_steps=response_age_steps,
                error=str(error),
                latency_ms=worker_result.latency_ms,
                raw_response_digest=_digest(worker_result.raw_response),
            )
        self._register_success()
        status = LLMStatus.READY
        candidate_ids = tuple(plan.plan_id for plan in candidates)
        self._log_worker_response(
            worker_result,
            current_step,
            "ready" if candidates else "all_rejected",
            candidates=candidates,
            candidate_ids=candidate_ids,
            rejected_candidates=rejected,
            error=None if candidates else "all LLM candidates rejected",
        )
        return LLMGenerationResult(
            request_id=worker_result.request_id,
            status=status,
            request_step=worker_result.request_step,
            completed_step=current_step,
            summary_hash=worker_result.summary_hash,
            request_plan_id=worker_result.request_plan_id,
            response_age_steps=response_age_steps,
            candidates=tuple(candidates),
            candidate_ids=candidate_ids,
            rejected_candidates=tuple(rejected),
            error=None if candidates else "all LLM candidates rejected",
            latency_ms=worker_result.latency_ms,
            raw_response_digest=_digest(worker_result.raw_response),
        )

    def _log_worker_response(
        self,
        worker_result,
        current_step,
        parse_status,
        candidates=(),
        candidate_ids=(),
        rejected_candidates=(),
        error=None,
    ):
        try:
            extracted_text = _extract_text(worker_result.raw_response) if worker_result.raw_response is not None else None
        except Exception as extract_error:
            extracted_text = None
            error = error or f"extract_text failed: {extract_error}"
        log_event(
            "llm_plan",
            {
                "request_id": worker_result.request_id,
                "request_step": worker_result.request_step,
                "completed_step": int(current_step),
                "generation": worker_result.generation,
                "summary_hash": worker_result.summary_hash,
                "request_plan_id": worker_result.request_plan_id,
                "response_age_steps": int(current_step) - int(worker_result.request_step),
                "latency_ms": round(float(worker_result.latency_ms), 3),
                "raw_response": worker_result.raw_response,
                "raw_response_digest": _digest(worker_result.raw_response),
                "extracted_text": extracted_text,
                "parse_status": parse_status,
                "candidate_ids": list(candidate_ids or ()),
                "candidates": [plan_summary(plan) for plan in candidates or ()],
                "rejected_candidates": list(rejected_candidates or ()),
                "error": error,
            },
        )

    def _remember_status(self, status):
        key = getattr(status, "value", status)
        if key is not None:
            self._status_step_counts[key] = self._status_step_counts.get(key, 0) + 1

    def _register_failure(self, current_step):
        self._consecutive_failures += 1
        self._last_failure_step = int(current_step)
        backoff = max(1, int(round(float(self._retry_backoff_steps))))
        self._next_retry_step = int(current_step) + backoff
        multiplier = float(self._params.get("retry_backoff_multiplier", 2.0))
        maximum = int(self._params.get("retry_backoff_max_steps", 60))
        self._retry_backoff_steps = min(maximum, max(1, int(round(backoff * multiplier))))

    def _register_success(self):
        self._consecutive_failures = 0
        self._last_failure_step = None
        self._next_retry_step = 0
        self._retry_backoff_steps = int(self._params.get("retry_backoff_initial_steps", 10))

    def _parse_candidates(self, request_id, request_step, raw_response):
        text = _extract_text(raw_response)
        if len(text) > int(self._params["max_response_chars"]):
            raise ValueError("LLM response exceeds max_response_chars")
        data = _extract_unique_json_object(text)
        if set(data) != {"candidates"} or not isinstance(data["candidates"], list):
            raise ValueError("LLM response must contain only candidates list")
        if not (2 <= len(data["candidates"]) <= int(self._params["max_candidates"])):
            raise ValueError("LLM candidate count outside allowed range")

        candidates = []
        rejected = []
        context = getattr(self, "_parse_context", None)
        if context is None:
            raise ValueError("missing parse context")
        for index, item in enumerate(data["candidates"]):
            try:
                plan = self._candidate_to_plan(request_id, request_step, index, item, context)
                result = validate_plan(
                    plan,
                    context["observation"],
                    context["memory_snapshot"],
                    context["situation"],
                )
                if result.valid:
                    candidates.append(plan)
                else:
                    rejected.append({"index": index, "errors": tuple(result.errors)})
            except Exception as error:
                rejected.append({"index": index, "errors": (str(error),)})
        return candidates, rejected

    def prepare_parse_context(self, observation, memory_snapshot, situation):
        self._parse_context = {
            "observation": observation,
            "memory_snapshot": memory_snapshot,
            "situation": situation,
        }

    def _candidate_to_plan(self, request_id, request_step, index, item, context):
        allowed_keys = {"mode", "tactic", "primary_target", "roles", "target_assignments", "valid_for_steps", "rationale"}
        if set(item) != allowed_keys:
            raise ValueError("candidate has extra or missing fields")
        mode = StrategyMode(item["mode"])
        tactic = Tactic(item["tactic"])
        primary_target = item["primary_target"]
        valid_for_steps = int(item["valid_for_steps"])
        if not int(self._params["min_valid_for_steps"]) <= valid_for_steps <= int(self._params["max_valid_for_steps"]):
            raise ValueError("valid_for_steps outside configured range")
        platform_ids = set(context["observation"].controlled_platform_ids)
        target_ids = {
            target_id
            for target_id, track in context["memory_snapshot"].tracks.items()
            if track.status in {"OBSERVED", "COASTING"}
        }
        roles = _parse_role_map(item["roles"], platform_ids)
        target_assignments = _parse_target_map(item["target_assignments"], platform_ids, target_ids)
        if primary_target is not None and primary_target not in target_ids:
            raise ValueError(f"unknown primary_target: {primary_target}")
        rationale = item["rationale"]
        if not isinstance(rationale, list) or any(not isinstance(value, str) for value in rationale):
            raise ValueError("rationale must be a list of strings")
        digest = hashlib.sha256(
            json.dumps(item, sort_keys=True, ensure_ascii=True).encode("utf-8")
        ).hexdigest()[:12]
        return TeamPlan(
            plan_id=f"llm_{request_id}_{index}_{digest}",
            created_step=request_step,
            created_sim_time=context["observation"].sim_time,
            mode=mode,
            tactic=tactic,
            roles=roles,
            target_assignments=target_assignments,
            primary_target=primary_target,
            valid_for_steps=valid_for_steps,
            source=PlanSource.LLM,
            rationale=rationale,
            metadata={"request_id": request_id, "candidate_index": index},
        )

    def _build_payload(self, summary):
        return {
            "model": self._params["model"],
            "temperature": self._params["temperature"],
            "max_tokens": self._params["max_tokens"],
            "messages": [
                {
                    "role": "system",
                    "content": _load_system_prompt(),
                },
                {
                    "role": "user",
                    "content": json.dumps(summary, ensure_ascii=False, sort_keys=True),
                },
            ],
        }

    def _default_transport(self, payload, params):
        api_key = params.get("api_key")
        if not api_key:
            raise RuntimeError("missing API key")
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            params["base_url"],
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=float(params["request_timeout_s"])) as response:
                return response.read().decode("utf-8")
        except urllib.error.URLError as error:
            if isinstance(error.reason, TimeoutError):
                raise TimeoutError("LLM request timed out") from error
            raise

    def _enabled(self):
        return bool(self._params.get("enabled", False))

    def _disabled_result(self):
        return LLMGenerationResult(None, LLMStatus.DISABLED, None)

    def _idle_result(self):
        return LLMGenerationResult(None, LLMStatus.IDLE, None)

    def _drain_queue(self):
        while True:
            try:
                self._result_queue.get_nowait()
            except Empty:
                return


def _load_system_prompt():
    path = Path(__file__).with_name("prompt.st")
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        text = ""
    if text:
        return text
    return (
        "Return exactly one JSON object and no other text. "
        "The object must have exactly one key: candidates. "
        "candidates must contain 2 to 4 objects. "
        "Each candidate object must have exactly these keys and no others: "
        "mode, tactic, primary_target, roles, target_assignments, valid_for_steps, rationale. "
        "Use only enum values and ids listed in the user's allowed object. "
        "roles must map every live platform id to one Role value. "
        "target_assignments must map every live platform id to a target id or null. "
        "rationale must be a list of short strings. "
        "Do not output set_flight, fire, co_fire, headings, altitudes, mach, scores, or probabilities."
    )


def _track_summary(target_id, track):
    return {
        "target_id": target_id,
        "target_side": track.target_side,
        "model": track.model,
        "latitude": track.position.latitude,
        "longitude": track.position.longitude,
        "altitude_m": track.position.altitude_m,
        "speed_mps": _speed_mps(track.velocity),
        "heading_deg": track.attitude.heading_deg,
        "detected_by": list(track.detected_by),
        "status": track.status,
        "track_age_s": track.track_age_s,
        "time_since_last_seen_s": track.time_since_last_seen_s,
    }


def _formation_summary(situation):
    own = situation.own_formation
    enemy = situation.enemy_formation
    return {
        "own_spacing_m": own.spacing_m if own else None,
        "own_heading_delta_deg": own.heading_delta_deg if own else None,
        "enemy_spacing_m": enemy.spacing_m if enemy else None,
        "enemy_heading_delta_deg": enemy.heading_delta_deg if enemy else None,
        "centroid_distance_m": situation.centroid_distance_m,
        "centroid_closing_speed_mps": situation.centroid_closing_speed_mps,
        "enemy_depth_delta_m": situation.enemy_depth_delta_m,
    }


def _plan_summary(plan):
    if plan is None:
        return None
    return {
        "plan_id": plan.plan_id,
        "mode": plan.mode.value,
        "tactic": plan.tactic.value,
        "roles": {key: value.value for key, value in sorted(plan.roles.items())},
        "target_assignments": dict(sorted(plan.target_assignments.items())),
        "primary_target": plan.primary_target,
        "valid_for_steps": plan.valid_for_steps,
        "source": plan.source.value,
    }


def _score_summary(score):
    if score is None:
        return None
    return {
        "valid": score.valid,
        "final_score": score.final_score,
        "expected_utility": score.expected_utility,
        "worst_case_utility": score.worst_case_utility,
        "switch_cost": score.switch_cost,
        "summary": score.summary,
    }


def _parse_role_map(values, platform_ids):
    if not isinstance(values, dict):
        raise ValueError("roles must be an object")
    roles = {}
    for platform_id, role_name in values.items():
        if platform_id not in platform_ids:
            raise ValueError(f"unknown platform in roles: {platform_id}")
        roles[platform_id] = Role(role_name)
    return roles


def _parse_target_map(values, platform_ids, target_ids):
    if not isinstance(values, dict):
        raise ValueError("target_assignments must be an object")
    assignments = {}
    for platform_id, target_id in values.items():
        if platform_id not in platform_ids:
            raise ValueError(f"unknown platform in target_assignments: {platform_id}")
        if target_id is not None and target_id not in target_ids:
            raise ValueError(f"unknown target in target_assignments: {target_id}")
        assignments[platform_id] = target_id
    return assignments


def _extract_text(raw_response):
    if isinstance(raw_response, str):
        try:
            data = json.loads(raw_response)
            if isinstance(data, dict) and "choices" in data:
                return data["choices"][0]["message"]["content"]
        except Exception:
            pass
        return raw_response
    if isinstance(raw_response, dict):
        if "choices" in raw_response:
            return raw_response["choices"][0]["message"]["content"]
        return json.dumps(raw_response, ensure_ascii=False)
    return str(raw_response)


def _extract_unique_json_object(text):
    objects = []
    start = None
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == "\"":
                in_string = False
            continue
        if char == "\"":
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start is not None:
                objects.append(text[start:index + 1])
                start = None
            if depth < 0:
                raise ValueError("ambiguous JSON braces")
    if len(objects) != 1:
        raise ValueError("response must contain exactly one JSON object")
    return json.loads(objects[0])


def _request_id(summary, request_step, generation):
    digest = _summary_hash(summary)
    return _request_id_from_hash(digest, request_step, generation)


def _request_id_from_hash(summary_hash, request_step, generation):
    digest = hashlib.sha256(f"{generation}:{request_step}:{summary_hash}".encode("utf-8")).hexdigest()[:12]
    return f"req_{request_step}_{digest}"


def _summary_hash(summary):
    payload = json.dumps(summary, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _summary_plan_id(summary):
    plan = summary.get("current_plan") if isinstance(summary, dict) else None
    if isinstance(plan, dict):
        return plan.get("plan_id")
    return None


def _digest(raw_response):
    if raw_response is None:
        return None
    return hashlib.sha256(str(raw_response).encode("utf-8")).hexdigest()[:16]


def _speed_mps(velocity):
    return (
        velocity.north_mps * velocity.north_mps
        + velocity.east_mps * velocity.east_mps
        + velocity.up_mps * velocity.up_mps
    ) ** 0.5


def _weapon_remaining(unit, weapon_name):
    for weapon in unit.weapons:
        if weapon.name == weapon_name:
            return int(weapon.count)
    return 0
