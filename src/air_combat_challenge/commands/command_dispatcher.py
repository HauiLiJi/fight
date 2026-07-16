
class CommandDispatcher:

    def __init__(self):
        self._executor_registry = {}

    def register_executor(self, cmd_type: str, executor):
        if cmd_type not in self._executor_registry:
            self._executor_registry[cmd_type] = executor
        else:
            raise ValueError(f"Duplicate Registration: {cmd_type}!")

    def execute(self, cmds):
        """
        接受智能体指令，调用宏指令接口（如机动动作）或仿真指令接口，产生仿真能运行的指令
        :param cmds: 智能体产生的指令（和仿真指令区分开）
            形式为：[{"cmd_type": "cmd_type_name", "sub_cmd_type": "sub_cmd_type_name", "params": {"xxx": 123}}]
            其中cmd_type对应cmd executor对象的注册名，sub_cmd_type为executor中函数名，params为函数参数
        :return: 仿真指令
        """
        responses = []
        for cmd_dict in cmds:
            if (cmd_dict is None) or (cmd_dict == {}):
                continue
            if cmd_dict["cmd_type"] in self._executor_registry:
                func = getattr(self._executor_registry[cmd_dict["cmd_type"]], cmd_dict["sub_cmd_type"])
                if func is not None:
                    responses.append(func(**cmd_dict["params"]))
                else:
                    raise ValueError(f"{cmd_dict['cmd_type']} object has no attribute {cmd_dict['sub_cmd_type']}")
            else:
                raise ValueError(f"Executor {cmd_dict['cmd_type']} is not registered!")
        return responses
