import json
import os
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import mujoco
import mujoco.viewer
import numpy as np


CONFIG_PATH = os.getenv("HUMANOID_CONFIG", "config.json")
KEYFRAME_NAME = os.getenv("KEYFRAME_NAME", "stand_bi")
PIN_ROOT_ENV = os.getenv("PIN_ROOT")
CUSTOM_STAND_NAME = "stand_bi"


def _env_bool(val: str | None):
    if val is None:
        return None
    return val.lower() not in ("0", "false", "no")


PIN_ROOT_OVERRIDE = _env_bool(PIN_ROOT_ENV)


@dataclass
class ActuatorInfo:
    index: int
    name: str
    joint_id: int
    qpos_adr: int
    qvel_adr: int
    ctrl_range: Tuple[float, float]
    gear: float
    kp: float
    kd: float
    ki: float


DEFAULT_CONFIG = {
    "pin_root": True,
    "keyframe_name": KEYFRAME_NAME,
    "friction": [4.5, 0.1, 0.01],
    "gains": {
        "hip": {"kp": 80.0, "kd": 35.0, "ki": 0.0},
        "knee": {"kp": 80.0, "kd": 35.0, "ki": 0.0},
        "ankle": {"kp": 75.0, "kd": 30.0, "ki": 0.0},
        "abdomen": {"kp": 35.0, "kd": 15.0, "ki": 0.0},
        "shoulder": {"kp": 25.0, "kd": 12.0, "ki": 0.0},
        "elbow": {"kp": 25.0, "kd": 12.0, "ki": 0.0},
    },
    "pose": {
        "root_z": 1.05,
        "abdomen": [0.0, 0.0, 0.0],
        "right_leg": [-0.10, 0.03, 0.04, -0.14, -0.01, 0.02],
        "left_leg": [-0.10, -0.03, -0.04, -0.14, -0.01, -0.02],
        "right_arm": [0.02, -0.18, -0.28],
        "left_arm": [-0.02, -0.18, -0.28],
    },
    "smoothing": 0.4,
    "i_clamp": 0.2,
}


def load_config(path: str) -> Dict:
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        return dict(DEFAULT_CONFIG)
    try:
        with open(path, "r") as f:
            cfg = json.load(f)
    except Exception:
        cfg = dict(DEFAULT_CONFIG)
    # merge defaults
    merged = json.loads(json.dumps(DEFAULT_CONFIG))
    def deep_merge(dst, src):
        for k, v in src.items():
            if isinstance(v, dict) and k in dst and isinstance(dst[k], dict):
                deep_merge(dst[k], v)
            else:
                dst[k] = v
    deep_merge(merged, cfg)
    return merged


def gain_for_actuator(name: str, cfg: Dict) -> Tuple[float, float, float]:
    lname = name.lower()
    g = cfg["gains"]
    if "hip" in lname:
        return g["hip"]["kp"], g["hip"]["kd"], g["hip"].get("ki", 0.0)
    if "knee" in lname:
        return g["knee"]["kp"], g["knee"]["kd"], g["knee"].get("ki", 0.0)
    if "ankle" in lname:
        return g["ankle"]["kp"], g["ankle"]["kd"], g["ankle"].get("ki", 0.0)
    if "shoulder" in lname:
        return g["shoulder"]["kp"], g["shoulder"]["kd"], g["shoulder"].get("ki", 0.0)
    if "elbow" in lname:
        return g["elbow"]["kp"], g["elbow"]["kd"], g["elbow"].get("ki", 0.0)
    if "abdomen" in lname:
        return g["abdomen"]["kp"], g["abdomen"]["kd"], g["abdomen"].get("ki", 0.0)
    return 25.0, 12.0, 0.0


def build_actuator_table(model: mujoco.MjModel, cfg: Dict) -> Tuple[np.ndarray, list[ActuatorInfo]]:
    infos = []
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"act{i}"
        joint_id = int(model.actuator_trnid[i][0])
        qpos_adr = int(model.jnt_qposadr[joint_id])
        qvel_adr = int(model.jnt_dofadr[joint_id])
        limited = bool(model.actuator_ctrllimited[i])
        crange = tuple(model.actuator_ctrlrange[i]) if limited else (-200.0, 200.0)
        kp, kd, ki = gain_for_actuator(name, cfg)
        gear = float(model.actuator_gear[i][0]) if model.actuator_gear.shape[1] > 0 else 1.0
        infos.append(
            ActuatorInfo(
                index=i,
                name=name,
                joint_id=joint_id,
                qpos_adr=qpos_adr,
                qvel_adr=qvel_adr,
                ctrl_range=crange,
                gear=gear,
                kp=kp,
                kd=kd,
                ki=ki,
            )
        )
    qpos_ref = np.zeros(model.nu)
    return qpos_ref, infos


def reset_to_keyframe(model: mujoco.MjModel, data: mujoco.MjData, key_name: str) -> None:
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name)
    if key_id == -1:
        raise ValueError(f"Keyframe '{key_name}' not found in model")
    mujoco.mj_resetDataKeyframe(model, data, key_id)


def custom_stand_qpos(model: mujoco.MjModel, cfg: Dict) -> np.ndarray:
    p = cfg["pose"]
    qpos = np.array([
        0.0, 0.0, p.get("root_z", 1.05),
        1.0, 0.0, 0.0, 0.0,
        *p.get("abdomen", [0.0, 0.0, 0.0]),
        *p.get("right_leg", [-0.10, 0.03, 0.04, -0.14, -0.01, 0.02]),
        *p.get("left_leg", [-0.10, -0.03, -0.04, -0.14, -0.01, -0.02]),
        *p.get("right_arm", [0.02, -0.18, -0.28]),
        *p.get("left_arm", [-0.02, -0.18, -0.28]),
    ], dtype=float)
    if qpos.shape[0] != model.nq:
        raise ValueError(f"Custom stand qpos length {qpos.shape[0]} != model.nq {model.nq}")
    return qpos


def tweak_friction(model: mujoco.MjModel, cfg: Dict) -> None:
    names = [
        "floor",
        "foot1_right",
        "foot2_right",
        "foot1_left",
        "foot2_left",
    ]
    target = np.array(cfg.get("friction", [3.5, 0.1, 0.01]))
    for name in names:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid != -1:
            model.geom_friction[gid] = target


def apply_config(model: mujoco.MjModel, data: mujoco.MjData, cfg: Dict):
    tweak_friction(model, cfg)
    qpos_ref, actuators = build_actuator_table(model, cfg)
    integral = np.zeros(model.nu)
    prev_ctrl = np.zeros(model.nu)

    keyframe_name = cfg.get("keyframe_name", KEYFRAME_NAME)
    if keyframe_name == CUSTOM_STAND_NAME:
        data.qpos[:] = custom_stand_qpos(model, cfg)
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
    else:
        reset_to_keyframe(model, data, keyframe_name)
        mujoco.mj_forward(model, data)

    root_qpos = data.qpos[:7].copy()
    for info in actuators:
        qpos_ref[info.index] = data.qpos[info.qpos_adr]
    return qpos_ref, actuators, integral, prev_ctrl, root_qpos


def main():
    model = mujoco.MjModel.from_xml_path("humanoid.xml")
    data = mujoco.MjData(model)

    cfg = load_config(CONFIG_PATH)
    cfg_mtime = os.path.getmtime(CONFIG_PATH) if os.path.exists(CONFIG_PATH) else None
    qpos_ref, actuators, integral, prev_ctrl, root_qpos = apply_config(model, data, cfg)

    dt = model.opt.timestep

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step_count = 0
        while viewer.is_running():
            step_count += 1
            # hot reload config every ~10 frames
            if step_count % 10 == 0 and os.path.exists(CONFIG_PATH):
                new_mtime = os.path.getmtime(CONFIG_PATH)
                if cfg_mtime is None or new_mtime > cfg_mtime:
                    cfg = load_config(CONFIG_PATH)
                    cfg_mtime = new_mtime
                    qpos_ref, actuators, integral, prev_ctrl, root_qpos = apply_config(model, data, cfg)

            pin_root_cfg = cfg.get("pin_root", True)
            pin_root = pin_root_cfg if PIN_ROOT_OVERRIDE is None else PIN_ROOT_OVERRIDE
            i_clamp = cfg.get("i_clamp", 0.2)
            smooth = cfg.get("smoothing", 0.4)

            if pin_root:
                data.qpos[:7] = root_qpos
                data.qvel[:6] = 0.0

            for info in actuators:
                q = data.qpos[info.qpos_adr]
                qd = data.qvel[info.qvel_adr]
                err = qpos_ref[info.index] - q
                derr = -qd

                integral[info.index] = np.clip(integral[info.index] + err * dt, -i_clamp, i_clamp)

                torque_cmd = info.kp * err + info.kd * derr + info.ki * integral[info.index]

                bias = data.qfrc_bias[info.qvel_adr]
                ctrl = (torque_cmd - bias) / max(info.gear, 1e-6)

                ctrl = np.clip(ctrl, info.ctrl_range[0], info.ctrl_range[1])
                smooth_ctrl = smooth * prev_ctrl[info.index] + (1 - smooth) * ctrl
                prev_ctrl[info.index] = smooth_ctrl
                data.ctrl[info.index] = smooth_ctrl

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(dt)


if __name__ == "__main__":
    main()
