"""Translate the internal tactical summary into a Chinese LLM-facing payload."""


PHASE_LABELS = {
    "approach": "接近阶段",
    "attack": "攻击阶段",
    "disengage": "脱离阶段",
    "regroup": "重组阶段",
    "endgame": "残局阶段",
}
RISK_LABELS = {"low": "低", "medium": "中", "high": "高"}
STATUS_LABELS = {"alive": "存活", "destroyed": "已损毁"}


def build_chinese_state_summary(state_summary):
    return {
        "仿真时间秒": state_summary.get("sim_time"),
        "红方编队状态": _force_status(state_summary.get("red_force_status", {})),
        "红方飞机": [_red_unit(unit) for unit in state_summary.get("red_units", [])],
        "红方空对空导弹库存": _air_to_air_inventory(
            state_summary.get("red_air_to_air_inventory", {})
        ),
        "来弹威胁": _missile_threats(
            state_summary.get("incoming_missile_threats", {})
        ),
        "远距空战包线": state_summary.get("bvr_engagement", {}),
        "蓝方敌情记忆": _contact_status(state_summary.get("blue_contact_status", {})),
        "蓝方目标航迹": [_blue_track(track) for track in state_summary.get("blue_tracks", [])],
        "双机几何关系": _pair_geometry(state_summary.get("pair_geometry", {})),
        "威胁评估": _threat_eval(state_summary.get("threat_eval", {})),
        "交战阶段": PHASE_LABELS.get(
            state_summary.get("engagement_phase"),
            state_summary.get("engagement_phase"),
        ),
    }


def _contact_status(contact_status):
    return {
        "已知存活蓝机": contact_status.get("known_alive_ids", []),
        "当前可见蓝机": contact_status.get("visible_ids", []),
        "失联蓝机": contact_status.get("lost_contact_ids", []),
        "已确认损毁蓝机": contact_status.get("destroyed_ids", []),
        "最近失联时长秒": contact_status.get("last_contact_age_s"),
        "失联详情": [
            {
                "目标编号": item.get("target_id"),
                "最后观测仿真时间秒": item.get("last_seen_sim_time"),
                "失联时长秒": item.get("age_s"),
                "预测已达时限": item.get("prediction_capped"),
            }
            for item in contact_status.get("lost_contacts", [])
        ],
    }


def _force_status(force_status):
    return {
        "初始飞机数量": force_status.get("initial_count"),
        "存活飞机数量": force_status.get("alive_count"),
        "损毁飞机数量": force_status.get("destroyed_count"),
        "本步新损毁飞机": force_status.get("newly_destroyed_ids"),
        "飞机状态": [
            {
                "飞机编号": unit.get("platform_id"),
                "状态": STATUS_LABELS.get(unit.get("status"), unit.get("status")),
                "损毁仿真时间秒": unit.get("destroyed_at_sim_time"),
            }
            for unit in force_status.get("units", [])
        ],
    }


def _red_unit(unit):
    return {
        "飞机编号": unit.get("platform_id"),
        "存活": unit.get("alive"),
        "高度米": unit.get("altitude_m"),
        "速度米每秒": unit.get("speed_mps"),
        "中距导弹剩余": unit.get("medium_missiles"),
        "近距导弹剩余": unit.get("short_missiles"),
        "全部武器库存": [
            {
                "武器名称": weapon.get("name"),
                "武器类型": weapon.get("weapon_type"),
                "已启用": weapon.get("enabled"),
                "剩余数量": weapon.get("count"),
                "距上次发射秒": weapon.get("time_since_last_fired_s"),
                "空对空武器": weapon.get("is_air_to_air"),
            }
            for weapon in unit.get("weapons", [])
        ],
        "最近敌机距离米": unit.get("nearest_enemy_distance"),
        "最近敌机编号": unit.get("nearest_enemy"),
        "危险等级": RISK_LABELS.get(unit.get("danger"), unit.get("danger")),
        "风险分数": unit.get("risk_score"),
        "来弹数量": unit.get("incoming_missile_count"),
    }


def _air_to_air_inventory(inventory):
    return {
        "空对空导弹总数": inventory.get("total_count"),
        "可攻击红机": inventory.get("attack_capable_unit_ids"),
        "按类型汇总": [
            {
                "武器类型": item.get("weapon_type"),
                "总剩余数量": item.get("total_count"),
                "携带平台": [
                    {
                        "飞机编号": carrier.get("platform_id"),
                        "武器名称": carrier.get("weapon_name"),
                        "已启用": carrier.get("enabled"),
                        "剩余数量": carrier.get("count"),
                    }
                    for carrier in item.get("carriers", [])
                ],
            }
            for item in inventory.get("by_type", [])
        ],
    }


def _blue_track(track):
    return {
        "目标编号": track.get("target_id"),
        "距最近红机米": track.get("distance_to_nearest_red_m"),
        "可见红机": track.get("visible_to"),
        "高度米": track.get("altitude_m"),
        "可攻击": track.get("is_attackable"),
        "当前BVR发射包线米": track.get("bvr_launch_range_m"),
        "相对最近红机高度差米": track.get("relative_altitude_to_nearest_red_m"),
        "最近红机方位角度": track.get("bearing_from_nearest_red_deg"),
    }


def _pair_geometry(pair_geometry):
    return {
        "双机间距米": pair_geometry.get("pair_distance_m"),
        "分散过大": pair_geometry.get("too_spread"),
        "具备互援": pair_geometry.get("mutual_support"),
    }


def _threat_eval(threat_eval):
    return {
        "最受威胁红机": threat_eval.get("most_threatened_red"),
        "处于高风险": threat_eval.get("high_risk"),
        "遭受双机压力": threat_eval.get("double_pressure"),
        "来弹总数": threat_eval.get("incoming_missile_count"),
    }


def _missile_threats(threats):
    return {
        "活动威胁": threats.get("active_threats", []),
        "受威胁红机": threats.get("threatened_platform_ids", []),
        "可见敌方导弹": threats.get("visible_enemy_missile_ids", []),
    }
