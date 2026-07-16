from ...utils.logging import get_muted_logger
from ...protocol import afsim_pb2

logger = get_muted_logger()


class BaseCmdExecutor:
    # 飞机机动动作指令
    def __init__(self, client):
        self._env_client = client

    def fly_heading_speed_altitude(self, platform_name: str, altitude: float, heading: float, mach: float, **kwargs):
        """姿态控制"""
        cmds = [
            {"name": "aPlatformName", "value": platform_name, "type": afsim_pb2.string, },
            {"name": "fAltitude", "value": str(altitude), "type": afsim_pb2.double, },
            {"name": "fHeading", "value": str(heading), "type": afsim_pb2.double, },
            {"name": "fMach", "value": str(mach), "type": afsim_pb2.double, },
        ]
        return self._env_client.send_command("FlyHeadingSpeedAltitude", cmds)

    def fly_heading(self, platform_name: str, heading: float, **kwargs):
        """航向控制"""
        cmds = [
            {"name": "aPlatformName", "value": platform_name, "type": afsim_pb2.string, },
            {"name": "fHeading", "value": str(heading), "type": afsim_pb2.double, },
        ]
        return self._env_client.send_command("FlyHeading", cmds)

    def fly_altitude(self, platform_name: str, altitude: float, **kwargs):
        """
        高度控制 m
        """
        cmds = [
            {"name": "aPlatformName", "value": platform_name, "type": afsim_pb2.string, },
            {"name": "fAltitude", "value": str(altitude), "type": afsim_pb2.double, },
        ]
        return self._env_client.send_command("FlyAltitude", cmds)

    def fly_speed(self, platform_name: str, speed: float, **kwargs):
        """
        速度控制 m/s
        """

        cmds = [
            {"name": "aPlatformName", "value": platform_name, "type": afsim_pb2.string, },
            {"name": "fSpeed", "value": str(speed), "type": afsim_pb2.double, },
        ]
        return self._env_client.send_command("FlySpeed", cmds)

    def create_platform(self, scenario_dict: dict = None, **kwargs):
        """
        创建基础想定
        """
        if scenario_dict:
            # 处理aircrafts配置
            aircrafts = scenario_dict.get('aircrafts', {})
            responses = []
            for aircraft_name, aircraft_info in aircrafts.items():
                # 提取参数
                cmds = [
                    {"name":"aPlatformName","value": aircraft_info.get('plat_id', aircraft_name), "type":afsim_pb2.string,},
                    {"name":"aPlatformType","value": aircraft_info.get('plat_type'),"type":afsim_pb2.string,},
                    {"name":"aSide","value": aircraft_info.get('side'),"type":afsim_pb2.string,},
                    {"name":"fHeading","value": str(aircraft_info.get('heading', 0.0)),"type":afsim_pb2.double,},
                    {"name":"fPitch","value": str(aircraft_info.get('pitch', 0.0)),"type":afsim_pb2.double,},
                    {"name":"fRoll","value": str(aircraft_info.get('roll', 0.0)),"type":afsim_pb2.double,},
                    {"name":"fLatitude","value": str(aircraft_info.get('lat', 0.0)),"type":afsim_pb2.double,},
                    {"name":"fLongitude","value": str(aircraft_info.get('lon', 0.0)),"type":afsim_pb2.double,},
                    {"name":"fAltitude","value": str(aircraft_info.get('alt', 0.0)),"type":afsim_pb2.double,},
                    {"name":"fSpeed","value": str(aircraft_info.get('speed', 0.0)),"type":afsim_pb2.double,},
                ]
                # 调用CreatePlatform函数
                responses.append(self._env_client.send_command("CreatePlatform", cmds))
            return responses
        return []

    def attack_target(self, platform_name: str,weapon_name: str, target_name: str, **kwargs):
        """
        攻击指定目标

        空空弹名称 ： mrm
        地空弹名称 ： gam
        空地弹名称 ： agm
        """
        cmds = [
            {"name": "aPlatformName", "value": platform_name, "type": afsim_pb2.string, },
            {"name": "aWeaponName", "value": weapon_name, "type": afsim_pb2.string, },
            {"name": "aTargetName", "value": target_name, "type": afsim_pb2.string, },
        ]
        response = self._env_client.send_command("AttackTarget", cmds)
        # 本地推演保留该回执日志，用于区分“智能体生成了 AttackTarget”和“AFSIM 接收/执行了 AttackTarget”。
        message = "[AFSIM_CMD] AttackTarget platform={} weapon={} target={} code={} sim_time={} message={}".format(
            platform_name,
            weapon_name,
            target_name,
            getattr(response, "code", None),
            getattr(response, "simTime", None),
            getattr(response, "message", ""),
        )
        logger.info(message)
        return response

    def co_attack_target(self, platform_name: str,weapon_name: str, target_name: str, guider_name: str, **kwargs):
        """
        基于探测者探测到的目标进行攻击
        platform: 攻击者
        weapon_name: 武器名称
        target_name: 目标
        guider_name: 探测者

        空空弹名称 ： mrm
        地空弹名称 ： gam
        空地弹名称 ： agm
        """
        cmds = [
            {"name": "aPlatformName", "value": platform_name, "type": afsim_pb2.string, },
            {"name": "aWeaponName", "value": weapon_name, "type": afsim_pb2.string, },
            {"name": "aTargetName", "value": target_name, "type": afsim_pb2.string, },
            {"name": "aGuiderName", "value": guider_name, "type": afsim_pb2.string, },
        ]
        response = self._env_client.send_command("CoAttackTarget", cmds)
        message = "[AFSIM_CMD] CoAttackTarget platform={} weapon={} target={} guider={} code={} sim_time={} message={}".format(
            platform_name,
            weapon_name,
            target_name,
            guider_name,
            getattr(response, "code", None),
            getattr(response, "simTime", None),
            getattr(response, "message", ""),
        )
        logger.info(message)
        return response

    def fly_in_path(self,platform_name: str,route: str, is_cycle: bool):
        """
        路径飞行
        route格式：纬度，经度，高度，速度
                  严格按照（北东天）坐标系输入点位，点位间用';'分割，数据以','分割。
                  角度单位（°），高度单位（m），速度单位（m/s）
        例：
        "40.0,120.0,8000,150;45.5,120.0,8500,150"

        is_cycle:是否为循环路径
        """
        cmds = [
            {"name": "aPlatformName", "value": platform_name, "type": afsim_pb2.string, },
            {"name": "aRoute", "value": route, "type": afsim_pb2.string, },
            {"name": "bIsCycle", "value": str(is_cycle), "type": afsim_pb2.bool, },
        ]
        # 调用CreatePlatform函数
        return self._env_client.send_command("FlyInPath", cmds)

    def delete_platform(self,platform_name:str):
        cmds = [
            {"name": "aPlatformName", "value": platform_name, "type": afsim_pb2.string, },
        ]
        return self._env_client.send_command("DeletePlatform", cmds)
