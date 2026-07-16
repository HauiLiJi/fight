import multiprocessing
from queue import Empty
import traceback
import uuid

from .agents import load_agent_manifest
from .models import ActionBatchV1, EpisodeContextV1, ObservationV1


class AgentWorkerError(RuntimeError):
    pass


class AgentTimeoutError(AgentWorkerError):
    pass


class AgentExecutionError(AgentWorkerError):
    pass


def _worker_main(manifest_path, expected_source_hash, request_queue, response_queue):
    try:
        loaded = load_agent_manifest(manifest_path)
        if loaded.source_hash != expected_source_hash:
            raise RuntimeError("agent source tree changed after validation")
        agent = loaded.agent_class()
        startup_error = None
    except BaseException as error:
        agent = None
        startup_error = f"{type(error).__name__}: {error}"
    while True:
        request = request_queue.get()
        request_id = request["request_id"]
        operation = request["operation"]
        try:
            if startup_error is not None:
                raise RuntimeError(startup_error)
            if operation == "close":
                agent.close()
                response_queue.put(
                    {"request_id": request_id, "ok": True, "result": None}
                )
                return
            if operation == "reset":
                context = EpisodeContextV1.model_validate(request["payload"])
                agent.reset(context)
                result = None
            elif operation == "act":
                observation = ObservationV1.model_validate(request["payload"])
                result = agent.act(observation)
                if not isinstance(result, ActionBatchV1):
                    result = ActionBatchV1.model_validate(result)
                result = result.model_dump(mode="json")
            else:
                raise ValueError(f"unknown worker operation: {operation}")
            response_queue.put({"request_id": request_id, "ok": True, "result": result})
        except BaseException as error:
            response_queue.put(
                {
                    "request_id": request_id,
                    "ok": False,
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
            )


class AgentWorker:
    def __init__(self, loaded_agent):
        self.loaded_agent = loaded_agent
        self.timeout_s = loaded_agent.manifest.step_timeout_s
        self._context = multiprocessing.get_context("spawn")
        self._process = None
        self._request_queue = None
        self._response_queue = None

    def start(self):
        if self.is_alive():
            return
        self._request_queue = self._context.Queue()
        self._response_queue = self._context.Queue()
        self._process = self._context.Process(
            target=_worker_main,
            args=(
                str(self.loaded_agent.manifest_path),
                self.loaded_agent.source_hash,
                self._request_queue,
                self._response_queue,
            ),
            daemon=True,
        )
        self._process.start()

    def reset(self, context):
        self._request("reset", context.model_dump(mode="json"))

    def act(self, observation):
        result = self._request("act", observation.model_dump(mode="json"))
        return ActionBatchV1.model_validate(result)

    def restart(self):
        self._terminate()
        self.start()

    def is_alive(self):
        return self._process is not None and self._process.is_alive()

    def close(self):
        if self.is_alive():
            try:
                self._request("close", None, timeout_s=1.0)
            except AgentWorkerError:
                pass
        self._terminate()

    def _request(self, operation, payload, timeout_s=None):
        if not self.is_alive():
            if self._process is not None:
                raise AgentExecutionError("agent worker is not running")
            self.start()
        request_id = str(uuid.uuid4())
        self._request_queue.put(
            {"request_id": request_id, "operation": operation, "payload": payload}
        )
        try:
            response = self._response_queue.get(timeout=timeout_s or self.timeout_s)
        except Empty as error:
            self._terminate()
            raise AgentTimeoutError(
                f"agent operation {operation} exceeded {timeout_s or self.timeout_s:.3f}s"
            ) from error
        if response.get("request_id") != request_id:
            self._terminate()
            raise AgentExecutionError("agent worker response id mismatch")
        if not response.get("ok"):
            raise AgentExecutionError(response.get("error", "agent execution failed"))
        return response.get("result")

    def _terminate(self):
        if self._process is not None and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2.0)
        self._process = None
        if self._request_queue is not None:
            self._request_queue.close()
        if self._response_queue is not None:
            self._response_queue.close()
        self._request_queue = None
        self._response_queue = None
