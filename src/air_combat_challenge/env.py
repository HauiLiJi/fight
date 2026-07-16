from .client import EnvClient
from .commands.cmd_executors.base_cmd_executor import BaseCmdExecutor
from .commands.command_dispatcher import CommandDispatcher
from .scenario_generator import ScenarioGenerator
from .utils.logging import get_muted_logger

import copy
import json
import random

logger = get_muted_logger()


class AFsimEnv:

    def __init__(self, ip, port, scenario_path, use_random_scenario=False, random_scenario_config=None,
                 mission_config=None, debug_print=False):
        self._env_client = EnvClient(afsim_ip=ip, afsim_port=port)
        self._n_time = 1
        self._debug_print = bool(debug_print)
        self.scenario_path = scenario_path
        self.use_random_scenario = use_random_scenario
        self.random_scenario_config = random_scenario_config or {}
        self._base_mission_config = copy.deepcopy(mission_config or {})
        self._current_mission_config = copy.deepcopy(self._base_mission_config)
        self._command_dispatcher = CommandDispatcher()
        self._register()
        # self._events_manager = EventsManager(summary_prefix={}.get("summary_prefix", ""))
        # todo acmi_recoder

    def _register(self):
        # 注册cmd executor
        self._command_dispatcher.register_executor("base_cmd",
                                                   BaseCmdExecutor(self._env_client))

    def _load_scenario(self, scenario_dict):
        #加载想定数据
        cmds = [
            {
                "cmd_type": "base_cmd",
                "sub_cmd_type": "create_platform",
                "params": {"scenario_dict": scenario_dict}
            }
        ]
        self._command_dispatcher.execute(cmds)

    def _select_episode_mission_config(self):
        """每局只随机一次任务类型，保证所有智能体看到同一个任务意图。"""
        mission_config = copy.deepcopy(self._base_mission_config)
        mission_profiles = mission_config.get("mission_profiles", {})
        candidates = mission_config.get("mission_candidates") or list(mission_profiles.keys())

        if mission_config.get("mission_randomize", False) and candidates:
            weights = self._mission_weights_for_candidates(candidates, mission_config.get("mission_weights"))
            mission_type = random.choices(candidates, weights=weights, k=1)[0]
        else:
            mission_type = mission_config.get("mission_type")

        if mission_type is not None:
            mission_config["mission_type"] = mission_type
            mission_config["mission_profile"] = copy.deepcopy(mission_profiles.get(mission_type, {}))
        return mission_config

    @staticmethod
    def _mission_weights_for_candidates(candidates, mission_weights):
        if isinstance(mission_weights, dict):
            return [float(mission_weights.get(candidate, 1.0)) for candidate in candidates]
        if isinstance(mission_weights, (list, tuple)) and len(mission_weights) == len(candidates):
            return [float(weight) for weight in mission_weights]
        return None

    def _make_extra_info(self):
        return {
            "mission_type": self._current_mission_config.get("mission_type"),
            "mission_config": copy.deepcopy(self._current_mission_config),
        }

    def reset(self):
        """

        :return: raw_obs, extra_info
        """

        self._env_client.connect_server()
        self._env_client.restart()
        self._current_mission_config = self._select_episode_mission_config()
        message = "[AFSIM_ENV] episode_mission_type={}".format(self._current_mission_config.get("mission_type"))
        logger.info(message)
        if self._debug_print:
            print(message)

        # 加载想定数据
        if self.use_random_scenario:
            # 生成随机想定
            generator = ScenarioGenerator(self.random_scenario_config)
            scenario_dict = generator.generate()
        else:
            # 从文件读取
            with open(self.scenario_path, 'r', encoding='utf-8') as f:
                scenario_dict = json.load(f)
        logger.info(
            "[AFSIM_ENV] load_scenario path={} aircraft_count={}".format(
                self.scenario_path,
                len(scenario_dict.get("aircrafts", {})),
            )
        )

        self._load_scenario(scenario_dict)
        # 六自由度模型执行默认动作序列
        self._env_client.step_n_times(30)
        self._sim_data = self._env_client.get_sim_data("SituationData")

        return self._sim_data, self._make_extra_info()

    def step(self, command_dict):
        """
        :param command_dict
        :return: raw_obs, extra_info
        """
        command_results = {"blue": [], "red": []}
        for side, cmds in command_dict.items():
            if cmds:
                logger.info("[AFSIM_ENV] side={} command_count={}".format(side, len(cmds)))
                self._print_attack_cmds(side, cmds)
                responses = self._command_dispatcher.execute(cmds)
                command_results[side] = [self._serialize_response(response) for response in responses]

        self._env_client.step_n_times(self._n_time)
        self._sim_data = self._env_client.get_sim_data("SituationData")

        extra_info = self._make_extra_info()
        extra_info["command_results"] = command_results
        return self._sim_data, extra_info

    def get_events(self):
        return self._env_client.get_sim_data("events")

    @staticmethod
    def _serialize_response(response):
        return {
            "code": getattr(response, "code", 0),
            "sim_time": getattr(response, "simTime", None),
            "message": getattr(response, "message", ""),
        }

    def _print_attack_cmds(self, side, cmds):
        for cmd in cmds:
            if not isinstance(cmd, dict):
                continue
            if cmd.get("cmd_type") != "base_cmd":
                continue
            if cmd.get("sub_cmd_type") not in {"attack_target", "co_attack_target"}:
                continue
            params = cmd.get("params", {})
            message = "[AFSIM_ENV] side={} cmd={} platform={} weapon={} target={}".format(
                side,
                cmd.get("sub_cmd_type"),
                params.get("platform_name"),
                params.get("weapon_name"),
                params.get("target_name"),
            )
            logger.info(message)
            if self._debug_print:
                print(message)
