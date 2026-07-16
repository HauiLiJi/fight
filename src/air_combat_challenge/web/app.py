from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import traceback
import urllib.error
import urllib.request
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..client import EnvClient
from ..env import AFsimEnv
from ..paths import PROJECT_ROOT
from ..protocol import afsim_pb2
from ..competition.agents import load_agent_manifest
from ..competition.environment import CompetitionEnv
from ..competition.runner import CompetitionRunner


DEFAULT_AGENT = "examples/rules/a2a_rule_agent.json"
DEFAULT_AFSIM_SCENARIO = "scenarios/air_to_air/start_up.txt"
KM_PER_LATITUDE_DEGREE = 111.195
# Configure the LLM once here. Keep this file out of source control if it contains a real key.
LLM_API_KEY = "sk-tHCKe3i1g92EQZRKkMR4NHy7v1uvu3mSOnzGxLxWy25j3BeN"
LLM_MODEL = "gpt-5.6-terra"
LLM_API_BASE = "https://www.micuapi.ai/v1"
LLM_SYSTEM_PROMPT = """你是一名空战仿真复盘分析助手。基于用户提供的单局 AFSIM 仿真结构化数据，使用中文输出客观、可追溯的赛后分析。

要求：
1. 先用 2-4 句话概述胜负、终局原因和决定性的时间窗口。
2. 分别分析蓝方与红方的态势、探测/锁定、机动、火力使用和编队协同；只讨论数据实际支持的内容。
3. 每个关键判断尽量引用平台 ID、仿真时间或事件作为证据；信息不足时明确说明“数据不足以判断”，不得臆造未记录的探测、命中或意图。
4. 给出不超过三条面向下一轮仿真配置或 Agent 规则的高层改进建议。建议应是仿真评估层面的，不提供现实世界可直接执行的武器操作步骤或参数。
5. 以 Markdown 输出，使用以下标题：## 总结、## 关键过程、## 双方表现、## 下一轮关注点。

仿真数据是唯一事实来源。事件、动作和遥测均可能不完整；不要把模型推断写成事实。"""


class RunRequest(BaseModel):
    blue_count: int = Field(default=2, ge=1, le=8)
    red_count: int = Field(default=2, ge=1, le=8)
    blue_start_latitude: float = Field(default=0.0, ge=-90.0, le=90.0)
    blue_start_longitude: float = Field(default=1.0, ge=-180.0, le=180.0)
    red_start_latitude: float = Field(default=0.0, ge=-90.0, le=90.0)
    red_start_longitude: float = Field(default=-1.0, ge=-180.0, le=180.0)
    blue_formation_spacing_km: float = Field(default=10.0, ge=0.1, le=200.0)
    red_formation_spacing_km: float = Field(default=10.0, ge=0.1, le=200.0)
    altitude_m: float = Field(default=8000.0, ge=1000.0, le=20000.0)
    speed_mps: float = Field(default=300.0, ge=50.0, le=700.0)
    blue_agent: str = DEFAULT_AGENT
    red_agent: str = DEFAULT_AGENT
    afsim_scenario: str = DEFAULT_AFSIM_SCENARIO
    auto_start_afsim: bool = True
    seed: Optional[int] = None
    max_steps: int = Field(default=600, ge=1, le=10000)
    time_scale: float = Field(default=1.0, ge=0.2, le=3.0)
    afsim_ip: str = "127.0.0.1"
    afsim_port: int = Field(default=19920, ge=1, le=65535)
    global_view: bool = True


class FilePickerRequest(BaseModel):
    kind: str


class AfsimStartRequest(BaseModel):
    afsim_scenario: str = DEFAULT_AFSIM_SCENARIO
    afsim_ip: str = "127.0.0.1"
    afsim_port: int = Field(default=19920, ge=1, le=65535)
    auto_start_afsim: bool = True


def _scenario_from_request(request: RunRequest):
    aircrafts = {}
    for side, count, latitude, longitude, spacing, heading in (
        (
            "blue",
            request.blue_count,
            request.blue_start_latitude,
            request.blue_start_longitude,
            request.blue_formation_spacing_km / KM_PER_LATITUDE_DEGREE,
            270,
        ),
        (
            "red",
            request.red_count,
            request.red_start_latitude,
            request.red_start_longitude,
            request.red_formation_spacing_km / KM_PER_LATITUDE_DEGREE,
            90,
        ),
    ):
        center = (count - 1) / 2
        for index in range(count):
            platform_id = f"{side}_fighter_{index + 1:02d}"
            aircrafts[platform_id] = {
                "plat_id": platform_id,
                "plat_type": "F22",
                "side": side,
                "heading": heading,
                "pitch": 0.0,
                "roll": 0.0,
                "lat": latitude + (index - center) * spacing,
                "lon": longitude,
                "alt": request.altitude_m,
                "speed": request.speed_mps,
            }
    return {"aircrafts": aircrafts}


class ConnectionHub:
    def __init__(self):
        self.connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.connections.add(websocket)

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            self.connections.discard(websocket)

    async def broadcast(self, message: dict):
        async with self._lock:
            connections = list(self.connections)
        disconnected = []
        for websocket in connections:
            try:
                await websocket.send_json(message)
            except (RuntimeError, WebSocketDisconnect):
                disconnected.append(websocket)
        if disconnected:
            async with self._lock:
                for websocket in disconnected:
                    self.connections.discard(websocket)


class SimulationManager:
    def __init__(self, hub: ConnectionHub, client_factory=EnvClient):
        self.hub = hub
        self.client_factory = client_factory
        self.lock = threading.Lock()
        self.running = False
        self.run_id: Optional[str] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.last_summary = _read_latest_episode_summary()
        self.last_error: Optional[str] = None
        self.stop_event: Optional[threading.Event] = None
        self.active_endpoint = None
        self.afsim_process = None

    def status(self):
        return {
            "running": self.running,
            "run_id": self.run_id,
            "last_summary": self.last_summary,
            "last_error": self.last_error,
        }

    def start(self, request: RunRequest, loop: asyncio.AbstractEventLoop):
        with self.lock:
            if self.running:
                raise RuntimeError("A simulation is already running")
            self.running = True
            self.run_id = str(uuid.uuid4())
            self.last_error = None
            self.loop = loop
            run_id = self.run_id
            self.stop_event = threading.Event()
            stop_event = self.stop_event
            self.active_endpoint = (request.afsim_ip, request.afsim_port)
        thread = threading.Thread(target=self._run, args=(run_id, request, stop_event), daemon=True)
        thread.start()
        return run_id

    def stop(self):
        with self.lock:
            if not self.running or self.stop_event is None:
                return {"accepted": False}
            self.stop_event.set()
            endpoint = self.active_endpoint
        if endpoint is None:
            return {"accepted": True, "afsim_stopped": False}

        client = self.client_factory(afsim_ip=endpoint[0], afsim_port=endpoint[1], rpc_timeout=3.0)
        try:
            client.connect_server()
            client.stop()
            return {"accepted": True, "afsim_stopped": True}
        except Exception as error:
            return {"accepted": True, "afsim_stopped": False, "error": str(error)}
        finally:
            client.close()

    def _publish(self, message_type: str, payload):
        if self.loop is None:
            return
        message = {"type": message_type, "run_id": self.run_id, "payload": payload}
        asyncio.run_coroutine_threadsafe(self.hub.broadcast(message), self.loop)

    def _run(self, run_id: str, request: RunRequest, stop_event: threading.Event):
        try:
            self.ensure_afsim(
                request.afsim_ip,
                request.afsim_port,
                request.afsim_scenario,
                request.auto_start_afsim,
            )
            scenario_path = self._write_scenario(run_id, request)
            blue_agent = load_agent_manifest(self._resolve_agent_path(request.blue_agent))
            red_agent = load_agent_manifest(self._resolve_agent_path(request.red_agent))
            raw_env = AFsimEnv(request.afsim_ip, request.afsim_port, scenario_path)
            env = CompetitionEnv(raw_env, max_steps=request.max_steps, global_view=request.global_view)
            runner = CompetitionRunner(
                env,
                blue_agent=blue_agent,
                red_agent=red_agent,
                output_dir=PROJECT_ROOT / "runs",
                max_steps=request.max_steps,
                time_scale=request.time_scale,
                should_stop=stop_event.is_set,
                on_episode_start=lambda episode: self._publish("episode_start", episode.model_dump(mode="json")),
                on_step=lambda result: self._publish("step", result.model_dump(mode="json")),
                on_episode_complete=lambda summary: self._publish("complete", summary),
            )
            summary = runner.run_episode(seed=request.seed)
            self.last_summary = summary
        except Exception as error:
            self.last_error = str(error)
            self._publish(
                "error",
                {"message": str(error), "traceback": traceback.format_exc(limit=4)},
            )
        finally:
            with self.lock:
                self.running = False
                self.stop_event = None
                self.active_endpoint = None
                if self.afsim_process is not None and self.afsim_process.poll() is not None:
                    self.afsim_process = None
            self._publish("status", self.status())

    def ensure_afsim(self, ip, port, scenario, auto_start):
        scenario_path = self._resolve_scenario_path(scenario)
        with self.lock:
            if self._is_afsim_active(ip, port):
                return {"ready": True, "started": False, "endpoint": f"{ip}:{port}"}
            if not auto_start:
                raise RuntimeError("AFSIM is not running and automatic startup is disabled")
            self._launch_afsim(ip, port, scenario_path)
        return {"ready": True, "started": True, "endpoint": f"{ip}:{port}"}

    def _launch_afsim(self, ip, port, scenario_path: Path):
        executable = self._find_mission_executable()
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.afsim_process = subprocess.Popen(
            [str(executable), "-es", str(scenario_path)],
            cwd=str(executable.parent),
            creationflags=creation_flags,
        )
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if self.afsim_process.poll() is not None:
                raise RuntimeError("mission.exe exited before the AFSIM gRPC server became ready")
            if self._is_afsim_active(ip, port):
                return
            time.sleep(0.25)
        raise TimeoutError("AFSIM did not become active within 30 seconds")

    def _is_afsim_active(self, ip, port):
        client = self.client_factory(afsim_ip=ip, afsim_port=port, rpc_timeout=1.0)
        try:
            client.connect_server()
            return client.get_server_state() == afsim_pb2.active
        except Exception:
            return False
        finally:
            client.close()

    @staticmethod
    def _find_mission_executable():
        configured_home = os.environ.get("AFSIM_HOME")
        candidate_roots = [Path(configured_home)] if configured_home else []
        candidate_roots.extend(sorted(PROJECT_ROOT.parent.glob("afsim-*-bin-*")))
        for root in candidate_roots:
            executable = root / "bin" / "mission.exe"
            if executable.is_file():
                return executable
        raise FileNotFoundError(
            "Cannot find mission.exe. Set AFSIM_HOME to the AFSIM installation directory."
        )

    @staticmethod
    def _resolve_agent_path(agent_path: str):
        candidate = Path(agent_path).expanduser()
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        candidate = candidate.resolve()
        if candidate.suffix.lower() != ".json" or not candidate.is_file():
            raise ValueError("Agent manifest must be an existing JSON file")
        return candidate

    @staticmethod
    def _resolve_scenario_path(scenario_path: str):
        candidate = Path(scenario_path).expanduser()
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        candidate = candidate.resolve()
        if candidate.name != "start_up.txt" or not candidate.is_file():
            raise ValueError("AFSIM scenario must be an existing start_up.txt file")
        return candidate

    @staticmethod
    def _write_scenario(run_id: str, request: RunRequest):
        scenario_dir = PROJECT_ROOT / "runs" / "web_scenarios"
        scenario_dir.mkdir(parents=True, exist_ok=True)
        path = scenario_dir / f"{run_id}.json"
        path.write_text(json.dumps(_scenario_from_request(request), indent=2), encoding="utf-8")
        return path


def _choose_local_file(kind: str):
    """Open a native file picker because browsers cannot disclose local paths."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as error:
        raise RuntimeError("Python Tk is required to open the local file picker") from error

    initial_dir = PROJECT_ROOT / ("scenarios" if kind == "scenario" else "examples")
    filetypes = (
        [("AFSIM scenario entry", "start_up.txt"), ("Text files", "*.txt")]
        if kind == "scenario"
        else [("Agent manifest", "*.json"), ("JSON files", "*.json")]
    )
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        selected = filedialog.askopenfilename(
            parent=root,
            initialdir=str(initial_dir),
            title="Select AFSIM scenario" if kind == "scenario" else "Select agent manifest",
            filetypes=filetypes,
        )
        return selected or None
    finally:
        root.destroy()


def _read_replay_telemetry(episode_id: str):
    try:
        episode_id = str(uuid.UUID(episode_id))
    except ValueError as error:
        raise ValueError("Invalid episode id") from error
    replay_path = PROJECT_ROOT / "runs" / f"{episode_id}.jsonl"
    if not replay_path.is_file():
        raise FileNotFoundError("Replay file does not exist")

    trajectories = {}
    events = []
    fire_actions = []
    locks = []

    def add_observations(observations):
        for observation in observations.values():
            for unit in observation.get("own_units", []):
                platform_id = unit.get("platform_id")
                if not platform_id:
                    continue
                trajectory = trajectories.setdefault(
                    platform_id,
                    {"platform_id": platform_id, "side": unit.get("side"), "samples": []},
                )
                trajectory["samples"].append(
                    {
                        "step_index": observation.get("step_index", 0),
                        "sim_time": observation.get("sim_time", 0.0),
                        "position": unit.get("position", {}),
                        "velocity": unit.get("velocity", {}),
                        "attitude": unit.get("attitude", {}),
                        "sensor": unit.get("sensor"),
                        "weapons": unit.get("weapons", []),
                    }
                )
            for track in observation.get("tracks", []):
                for detector_id in track.get("detected_by", []):
                    locks.append(
                        {
                            "platform_id": detector_id,
                            "target_id": track.get("target_id"),
                            "step_index": observation.get("step_index", 0),
                            "sim_time": observation.get("sim_time", 0.0),
                        }
                    )

    with replay_path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            payload = record.get("payload", {})
            if record.get("record_type") == "episode_start":
                add_observations(payload.get("episode", {}).get("observations", {}))
            elif record.get("record_type") == "step":
                result = payload.get("result", {})
                add_observations(result.get("observations", {}))
                events.extend(result.get("events", []))
                for side_batches in payload.get("submitted_actions", {}).values():
                    for batch in side_batches:
                        for action in batch.get("batch", {}).get("actions", []):
                            if action.get("type") in {"fire", "co_fire"}:
                                fire_actions.append(
                                    {
                                        "platform_id": action.get("platform_id"),
                                        "target_id": action.get("target_id"),
                                        "weapon_name": action.get("weapon_name"),
                                        "action_type": action.get("type"),
                                        "step_index": result.get("step_index", 0),
                                        "sim_time": result.get("sim_time", 0.0),
                                    }
                                )
    events.sort(key=lambda event: (event.get("sim_time", 0.0), event.get("event_id", -1)))
    fired_by_pair = {}
    for event in events:
        if event.get("event_type") == "WeaponFired":
            fired_by_pair.setdefault(
                (event.get("shooter"), event.get("target")),
                [],
            ).append(event.get("sim_time"))
    for action in fire_actions:
        fired_times = fired_by_pair.get((action.get("platform_id"), action.get("target_id")), [])
        later_fires = [time for time in fired_times if time >= action["sim_time"] - 1.0]
        if later_fires:
            action["submitted_sim_time"] = action["sim_time"]
            action["sim_time"] = later_fires[0]

    return {
        "episode_id": episode_id,
        "trajectories": list(trajectories.values()),
        "events": events,
        "fire_actions": fire_actions,
        "locks": locks,
    }


def _read_episode_summary(episode_id: str):
    summary_path = PROJECT_ROOT / "runs" / f"{episode_id}.summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError("Episode summary does not exist")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _read_latest_episode_summary():
    summaries = sorted(
        (PROJECT_ROOT / "runs").glob("*.summary.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not summaries:
        return None
    try:
        return json.loads(summaries[0].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _round_number(value, digits=2):
    return round(value, digits) if isinstance(value, (int, float)) else None


def _engagement_time_consistency(telemetry: dict):
    """Build traceable launch-to-damage chains from the events AFSIM exposes."""
    events = telemetry.get("events", [])
    terminal_events = [
        event
        for event in events
        if event.get("event_type") in {"WeaponHit", "WeaponMissed"}
    ]
    used_terminal_indexes = set()
    chains = []
    issues = []

    for launch in (event for event in events if event.get("event_type") == "WeaponFired"):
        launch_time = launch.get("sim_time")
        if not isinstance(launch_time, (int, float)):
            issues.append({"kind": "invalid_launch_time", "launch": launch})
            continue
        terminal_index = next(
            (
                index
                for index, terminal in enumerate(terminal_events)
                if index not in used_terminal_indexes
                and terminal.get("shooter") == launch.get("shooter")
                and terminal.get("target") == launch.get("target")
                and terminal.get("sim_time", float("inf")) >= launch_time
            ),
            None,
        )
        terminal = terminal_events[terminal_index] if terminal_index is not None else None
        if terminal_index is not None:
            used_terminal_indexes.add(terminal_index)

        damage_event = None
        if terminal and terminal.get("event_type") == "WeaponHit":
            damage_event = next(
                (
                    event
                    for event in events
                    if event.get("event_type") == "PlatformBroken"
                    and event.get("platform") == launch.get("target")
                    and event.get("sim_time", float("inf")) >= terminal.get("sim_time", 0.0)
                ),
                None,
            )
        chain = {
            "shooter": launch.get("shooter"),
            "target": launch.get("target"),
            "weapon": launch.get("weapon"),
            "launch_time": launch_time,
            "terminal_event": terminal.get("event_type") if terminal else None,
            "terminal_time": terminal.get("sim_time") if terminal else None,
            "damage_time": damage_event.get("sim_time") if damage_event else None,
        }
        chains.append(chain)
        if terminal and terminal.get("sim_time", launch_time) < launch_time:
            issues.append({"kind": "terminal_before_launch", "chain": chain})
        if (
            damage_event
            and damage_event.get("sim_time", terminal.get("sim_time", 0.0))
            < terminal.get("sim_time", 0.0)
        ):
            issues.append({"kind": "damage_before_hit", "chain": chain})

    launches = [event for event in events if event.get("event_type") == "WeaponFired"]
    for terminal in terminal_events:
        if not any(
            launch.get("shooter") == terminal.get("shooter")
            and launch.get("target") == terminal.get("target")
            and launch.get("sim_time", float("inf")) <= terminal.get("sim_time", float("-inf"))
            for launch in launches
        ):
            issues.append({"kind": "terminal_without_preceding_launch", "event": terminal})

    return {"chain_count": len(chains), "issues": issues, "chains": chains}


def _build_llm_brief(summary: dict, telemetry: dict):
    """Reduce the replay to factual, bounded data suitable for an LLM request."""
    platforms = []
    for trajectory in telemetry["trajectories"]:
        samples = trajectory.get("samples", [])
        if not samples:
            continue
        first, last = samples[0], samples[-1]
        altitudes = [
            altitude
            for sample in samples
            if isinstance((altitude := sample.get("position", {}).get("altitude_m")), (int, float))
        ]
        speeds = []
        for sample in samples:
            velocity = sample.get("velocity", {})
            components = [velocity.get(axis, 0.0) or 0.0 for axis in ("north_mps", "east_mps", "up_mps")]
            speeds.append(sum(component * component for component in components) ** 0.5)
        platforms.append(
            {
                "platform_id": trajectory.get("platform_id"),
                "side": trajectory.get("side"),
                "sample_count": len(samples),
                "first_sample": first,
                "last_sample": last,
                "altitude_m": {
                    "min": _round_number(min(altitudes)) if altitudes else None,
                    "max": _round_number(max(altitudes)) if altitudes else None,
                },
                "speed_mps": {"min": _round_number(min(speeds)), "max": _round_number(max(speeds))},
            }
        )
    events = telemetry.get("events", [])
    event_counts = {}
    for event in events:
        event_type = event.get("event_type", "unknown")
        event_counts[event_type] = event_counts.get(event_type, 0) + 1
    return {
        "episode_summary": summary,
        "replay_data": {
            "platforms": platforms,
            "event_counts": event_counts,
            "events": events,
            "fire_actions": telemetry.get("fire_actions", []),
            "locks": telemetry.get("locks", []),
            "engagement_time_consistency": _engagement_time_consistency(telemetry),
        },
        "data_notes": "平台遥测已保留首末样本及高度/速度范围；事件、开火动作与锁定记录按回放原样提供。",
    }


def _request_llm_analysis(episode_id: str):
    if not LLM_API_KEY or LLM_API_KEY == "PASTE_YOUR_OPENAI_API_KEY_HERE":
        raise RuntimeError("LLM is not configured. Set LLM_API_KEY in web/app.py before requesting analysis.")
    api_base = LLM_API_BASE.rstrip("/")
    model = LLM_MODEL
    telemetry = _read_replay_telemetry(episode_id)
    summary = _read_episode_summary(episode_id)
    brief = _build_llm_brief(summary, telemetry)
    request_body = json.dumps(
        {
            "model": model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": "请对以下单局空战仿真数据作赛后复盘：\n" + json.dumps(brief, ensure_ascii=False)},
            ],
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{api_base}/chat/completions",
        data=request_body,
        headers={
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json",
            "User-Agent": "air-combat-challenge/0.1",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"LLM request failed ({error.code}): {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"LLM request could not be completed: {error.reason}") from error
    try:
        analysis = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError("LLM response did not contain an analysis message") from error
    return {"episode_id": episode_id, "model": model, "analysis": analysis}


def create_app():
    app = FastAPI(title="Air Combat Control Console", docs_url=None, redoc_url=None)
    hub = ConnectionHub()
    manager = SimulationManager(hub)
    static_dir = Path(__file__).with_name("static")

    @app.get("/")
    async def index():
        return FileResponse(static_dir / "index.html")

    @app.get("/app.js")
    async def app_script():
        return FileResponse(static_dir / "app.js", media_type="text/javascript")

    @app.get("/app.css")
    async def app_style():
        return FileResponse(static_dir / "app.css", media_type="text/css")

    @app.get("/baiyang-logo.png")
    async def baiyang_logo():
        return FileResponse(static_dir / "baiyang-logo.png", media_type="image/png")

    @app.get("/api/status")
    async def status():
        return manager.status()

    @app.get("/api/replays/{episode_id}/telemetry")
    async def replay_telemetry(episode_id: str):
        try:
            return await asyncio.to_thread(_read_replay_telemetry, episode_id)
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (ValueError, json.JSONDecodeError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/api/replays/{episode_id}/analysis")
    async def replay_analysis(episode_id: str):
        try:
            uuid.UUID(episode_id)
            return await asyncio.to_thread(_request_llm_analysis, episode_id)
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (ValueError, json.JSONDecodeError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except RuntimeError as error:
            status_code = 503 if "not configured" in str(error) else 502
            raise HTTPException(status_code=status_code, detail=str(error)) from error

    @app.get("/api/file-options")
    async def file_options():
        scenario_root = PROJECT_ROOT / "scenarios"
        scenarios = [
            str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
            for path in sorted(scenario_root.rglob("start_up.txt"))
        ]
        agents = []
        for path in sorted(PROJECT_ROOT.rglob("*.json")):
            if "runs" in path.relative_to(PROJECT_ROOT).parts:
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if {"api_version", "topology", "entrypoint"}.issubset(data):
                agents.append(str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"))
        return {"scenarios": scenarios, "agents": agents}

    @app.post("/api/pick-file")
    async def pick_file(request: FilePickerRequest):
        if request.kind not in {"scenario", "agent"}:
            raise HTTPException(status_code=422, detail="Unsupported file picker kind")
        try:
            path = await asyncio.to_thread(_choose_local_file, request.kind)
        except Exception as error:
            raise HTTPException(status_code=500, detail=str(error)) from error
        return {"path": path}

    @app.get("/api/afsim-status")
    async def afsim_status(ip: str = "127.0.0.1", port: int = 19920):
        client = EnvClient(afsim_ip=ip, afsim_port=port, rpc_timeout=2.0)
        try:
            client.connect_server()
            state = client.get_server_state()
            return {
                "ready": state == afsim_pb2.active,
                "state_code": state,
                "endpoint": f"{ip}:{port}",
            }
        except Exception as error:
            return {"ready": False, "endpoint": f"{ip}:{port}", "error": str(error)}
        finally:
            client.close()

    @app.post("/api/afsim/start")
    async def start_afsim(request: AfsimStartRequest):
        try:
            return await asyncio.to_thread(
                manager.ensure_afsim,
                request.afsim_ip,
                request.afsim_port,
                request.afsim_scenario,
                request.auto_start_afsim,
            )
        except Exception as error:
            raise HTTPException(status_code=500, detail=str(error)) from error

    @app.post("/api/runs")
    async def start_run(request: RunRequest):
        try:
            run_id = manager.start(request, asyncio.get_running_loop())
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {"run_id": run_id, "status": "starting"}

    @app.post("/api/runs/stop")
    async def stop_run():
        result = await asyncio.to_thread(manager.stop)
        if not result["accepted"]:
            raise HTTPException(status_code=409, detail="No simulation is running")
        return {"status": "stopping", **result}

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await hub.connect(websocket)
        await websocket.send_json({"type": "status", "payload": manager.status()})
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            await hub.disconnect(websocket)

    return app
