from typing import Any, Dict, List

from .protocol import afsim_pb2, afsim_pb2_grpc
from .utils.logging import get_muted_logger
import time
import grpc
import json

logger = get_muted_logger()

empty = afsim_pb2.google_dot_protobuf_dot_empty__pb2.Empty()


class AFSIMConnectionError(ConnectionError):
    """Raised when the configured AFSIM gRPC endpoint is unavailable."""


class EnvClient(object):
    def __init__(
        self,
        afsim_ip="127.0.0.1",
        afsim_port=19920,
        rpc_timeout=5.0,
        restart_timeout=120.0,
        retry_delay=1.0,
        reset_settle_time=1.0,
        step_timeout_margin=10.0,
    ):
        self._ip = afsim_ip
        self._port = afsim_port
        self.rpc_timeout = float(rpc_timeout)
        self.restart_timeout = float(restart_timeout)
        self.retry_delay = float(retry_delay)
        self.reset_settle_time = float(reset_settle_time)
        self.step_timeout_margin = float(step_timeout_margin)

    def connect_server(self):
        target = self._ip + ":" + str(self._port)
        self.channel = grpc.insecure_channel(
            target,
            options=[('grpc.max_receive_message_length', 10 * 1024 * 1024)])
        #设置消息大小限制为10M

        self.server = afsim_pb2_grpc.SimulationServiceStub(self.channel)
        try:
            grpc.channel_ready_future(self.channel).result(timeout=self.rpc_timeout)
        except grpc.FutureTimeoutError as error:
            self.channel.close()
            raise AFSIMConnectionError(
                f"Cannot connect to AFSIM gRPC server at {target}. "
                "Start scenarios/air_to_air/start_up.txt with mission.exe -es "
                "and keep that process running."
            ) from error

    def get_server_state(self):
        response = self.server.getSimServerState(empty, timeout=self.rpc_timeout)
        return response.state

    def reset(self):
        try:
            self.server.reset(empty, timeout=self.restart_timeout)
        except grpc.RpcError as error:
            if error.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                raise TimeoutError(
                    "AFSIM reset did not complete before restart_timeout"
                ) from error
            raise

        if self.reset_settle_time > 0:
            time.sleep(self.reset_settle_time)

    def stop(self):
        """Stop the currently active AFSIM simulation without closing Warlock."""
        return self.server.stop(empty, timeout=self.rpc_timeout)

    def restart(self):
        self.reset()
        deadline = time.monotonic() + self.restart_timeout
        last_error = None
        while time.monotonic() < deadline:
            try:
                state = self.server.getSimServerState(empty, timeout=self.rpc_timeout)
                logger.info("GetSimulationState(): {}".format(state.state))
                if state.state == afsim_pb2.active:
                    return
            except grpc.RpcError as e:
                last_error = e
                if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                    logger.warning("调用超时")
                else:
                    logger.warning("调用发生异常(可能是AfSim引擎在重启中): {}".format(e.details()))
            time.sleep(self.retry_delay)
        raise TimeoutError("AFSIM did not become active before restart_timeout") from last_error

    def send_command(self,
                     func_name: str,
                     params_list: List[Dict[str, Any]],
                     ):
        send_command_msg = afsim_pb2.SendCommand()
        send_command_msg.funcName = func_name

        # 添加参数
        for param in params_list:
            param_msg = send_command_msg.params.add()
            param_msg.name = param['name']
            param_msg.value = param['value']
            param_msg.type = param['type']

        # 调用sendCommand方法
        response = self.server.sendCommand(send_command_msg, timeout=self.rpc_timeout)
        return response

    def step_n_times(self, n_time):
        timeout = max(
            self.rpc_timeout,
            float(n_time) + self.step_timeout_margin,
        )
        return self.server.advanceStepTo(
            afsim_pb2.AdvanceStepTo(time=n_time),
            timeout=timeout,
        )

    def get_sim_data(self, label: str) -> Dict[str, Any]:
        data = self.server.getSimData(
            afsim_pb2.GetSimData(dataType=label),
            timeout=self.rpc_timeout,
        )
        if getattr(data, "code", 0) not in (None, 0):
            raise RuntimeError(f"AFSIM getSimData({label}) failed: {data.message}")
        return json.loads(data.message)

    def close(self):
        channel = getattr(self, "channel", None)
        if channel is not None:
            channel.close()
