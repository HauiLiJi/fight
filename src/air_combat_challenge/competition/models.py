from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


Side = Literal["blue", "red"]
Winner = Literal["blue", "red", "draw"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class PositionV1(StrictModel):
    longitude: float
    latitude: float
    altitude_m: float


class VelocityV1(StrictModel):
    north_mps: float = 0.0
    east_mps: float = 0.0
    up_mps: float = 0.0


class AttitudeV1(StrictModel):
    heading_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0


class SensorStateV1(StrictModel):
    enabled: bool
    cue_azimuth: float = 0.0


class WeaponStateV1(StrictModel):
    name: str
    weapon_type: str
    enabled: bool
    count: int = Field(ge=0)
    time_since_last_fired_s: float = 0.0


class UnitStateV1(StrictModel):
    platform_id: str
    side: Side
    entity_type: str
    position: PositionV1
    velocity: VelocityV1
    attitude: AttitudeV1
    sensor: Optional[SensorStateV1] = None
    weapons: List[WeaponStateV1] = Field(default_factory=list)


class TrackStateV1(StrictModel):
    target_id: str
    target_side: str
    model: str
    position: PositionV1
    velocity: VelocityV1
    attitude: AttitudeV1
    detected_by: List[str] = Field(default_factory=list)


class EventV1(StrictModel):
    event_id: int = Field(ge=0)
    sim_time: float
    event_type: str
    platform: Optional[str] = None
    shooter: Optional[str] = None
    weapon: Optional[str] = None
    target: Optional[str] = None
    sensor: Optional[str] = None
    event_time_kind: Optional[str] = None


class ObservationV1(StrictModel):
    api_version: Literal["1.0"] = "1.0"
    episode_id: str
    step_index: int = Field(ge=0)
    sim_time: float
    side: Side
    controlled_platform_ids: List[str]
    own_units: List[UnitStateV1]
    tracks: List[TrackStateV1]
    events: List[EventV1] = Field(default_factory=list)
    mission: Dict[str, Any] = Field(default_factory=dict)


class ActionBase(StrictModel):
    action_id: Optional[str] = None
    platform_id: str


class SetFlightActionV1(ActionBase):
    type: Literal["set_flight"]
    heading_deg: float = Field(ge=0.0, lt=360.0)
    altitude_m: float = Field(ge=1000.0, le=20000.0)
    mach: float = Field(ge=0.2, le=2.0)


class SetHeadingActionV1(ActionBase):
    type: Literal["set_heading"]
    heading_deg: float = Field(ge=0.0, lt=360.0)


class SetAltitudeActionV1(ActionBase):
    type: Literal["set_altitude"]
    altitude_m: float = Field(ge=1000.0, le=20000.0)


class SetSpeedActionV1(ActionBase):
    type: Literal["set_speed"]
    speed_mps: float = Field(ge=50.0, le=700.0)


class WaypointV1(StrictModel):
    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)
    altitude_m: float = Field(ge=1000.0, le=20000.0)
    speed_mps: float = Field(ge=50.0, le=700.0)


class FlyPathActionV1(ActionBase):
    type: Literal["fly_path"]
    waypoints: List[WaypointV1] = Field(min_length=1, max_length=16)
    cycle: bool = False


class FireActionV1(ActionBase):
    type: Literal["fire"]
    weapon_name: str
    target_id: str


class CoFireActionV1(FireActionV1):
    type: Literal["co_fire"]
    guider_id: str


ActionV1 = Annotated[
    Union[
        SetFlightActionV1,
        SetHeadingActionV1,
        SetAltitudeActionV1,
        SetSpeedActionV1,
        FlyPathActionV1,
        FireActionV1,
        CoFireActionV1,
    ],
    Field(discriminator="type"),
]


class ActionBatchV1(StrictModel):
    api_version: Literal["1.0"] = "1.0"
    actions: List[ActionV1] = Field(default_factory=list, max_length=64)


class ActionReportV1(StrictModel):
    action_index: int = Field(ge=0)
    action_id: Optional[str] = None
    action_type: str
    platform_id: str
    status: Literal["accepted", "rejected", "execution_error"]
    reason_code: str
    message: str = ""


class EpisodeContextV1(StrictModel):
    api_version: Literal["1.0"] = "1.0"
    episode_id: str
    side: Side
    controlled_platform_ids: List[str]
    seed: Optional[int] = None
    mission: Dict[str, Any] = Field(default_factory=dict)


class EpisodeStartV1(StrictModel):
    api_version: Literal["1.0"] = "1.0"
    episode_id: str
    observations: Dict[str, ObservationV1]


class ScoreOutcomeV1(StrictModel):
    rewards: Dict[str, float]
    terminated: bool
    truncated: bool
    winner: Optional[Winner] = None
    reason: Optional[str] = None


class StepResultV1(StrictModel):
    api_version: Literal["1.0"] = "1.0"
    episode_id: str
    step_index: int = Field(ge=0)
    sim_time: float
    observations: Dict[str, ObservationV1]
    rewards: Dict[str, float]
    terminated: bool
    truncated: bool
    winner: Optional[Winner] = None
    reason: Optional[str] = None
    events: List[EventV1] = Field(default_factory=list)
    action_reports: Dict[str, List[ActionReportV1]] = Field(default_factory=dict)
    diagnostics: Dict[str, Any] = Field(default_factory=dict)


class AgentManifestV1(StrictModel):
    api_version: Literal["1.0"] = "1.0"
    topology: Literal["team", "per_aircraft"]
    entrypoint: str
    step_timeout_s: float = Field(default=5.0, gt=0.0, le=60.0)
