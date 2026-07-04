import os
import time
from dataclasses import dataclass
from typing import Tuple

import mujoco
import mujoco.viewer
import numpy as np


KEYFRAME_NAME = os.getenv("KEYFRAME_NAME", "stand_on_left_leg")
PIN_ROOT = os.getenv("PIN_ROOT", "1").lower() not in ("0", "false", "no")
CUSTOM_STAND_NAME = "stand_bi"


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


def gain_for_actuator(name: str) -> Tuple[float, float, float]:
    lname = name.lower()
    if "hip" in lname or "knee" in lname:
        return 95.0, 42.0, 0.0
    if "ankle" in lname:
        return 100.0, 36.0, 1.5
    if "shoulder" in lname:
        return 25.0, 12.0, 0.0
    if "elbow" in lname:
        return 25.0, 12.0, 0.0
    if "abdomen" in lname:
        return 35.0, 18.0, 0.0
    return 25.0, 12.0, 0.0


def build_actuator_table(model: mujoco.MjModel) -> Tuple[np.ndarray, list[ActuatorInfo]]:
    infos = []
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"act{i}"
        joint_id = int(model.actuator_trnid[i][0])
        qpos_adr = int(model.jnt_qposadr[joint_id])
        qvel_adr = int(model.jnt_dofadr[joint_id])
        limited = bool(model.actuator_ctrllimited[i])
        crange = tuple(model.actuator_ctrlrange[i]) if limited else (-200.0, 200.0)
        kp, kd, ki = gain_for_actuator(name)
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


def custom_stand_qpos(model: mujoco.MjModel) -> np.ndarray:
    qpos = np.array([
        0.0, 0.0, 1.08,  # root pos
        1.0, 0.0, 0.0, 0.0,  # root quat
        0.0, 0.0, 0.0,  # abdomen z,y,x
        -0.08, 0.06, -0.02, -0.12, 0.04, 0.05,  # right leg hipx, hipz (abduction), hipy, knee, ankle_y, ankle_x (eversion)
        -0.08, -0.06, 0.02, -0.12, 0.04, -0.05,  # left leg
        0.02, -0.20, -0.30,  # right shoulder1, shoulder2, elbow
        -0.02, -0.20, -0.30,  # left shoulder1, shoulder2, elbow
    ], dtype=float)
    if qpos.shape[0] != model.nq:
        raise ValueError(f"Custom stand qpos length {qpos.shape[0]} != model.nq {model.nq}")
    return qpos


def tweak_friction(model: mujoco.MjModel) -> None:
    names = [
        "floor",
        "foot1_right",
        "foot2_right",
        "foot1_left",
        "foot2_left",
    ]
    target = np.array([5.5, 0.1, 0.01])
    for name in names:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid != -1:
            model.geom_friction[gid] = target


def main():
    model = mujoco.MjModel.from_xml_path("humanoid.xml")
    data = mujoco.MjData(model)

    tweak_friction(model)

    qpos_ref, actuators = build_actuator_table(model)
    integral = np.zeros(model.nu)
    prev_ctrl = np.zeros(model.nu)

    if KEYFRAME_NAME == CUSTOM_STAND_NAME:
        data.qpos[:] = custom_stand_qpos(model)
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
    else:
        reset_to_keyframe(model, data, KEYFRAME_NAME)
        mujoco.mj_forward(model, data)

    root_qpos = data.qpos[:7].copy()

    for info in actuators:
        qpos_ref[info.index] = data.qpos[info.qpos_adr]

    dt = model.opt.timestep
    i_clamp = 0.2
    smooth = 0.3

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            if PIN_ROOT:
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
