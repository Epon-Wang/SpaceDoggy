#!/usr/bin/env python3
import time
import math
from dataclasses import dataclass
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_ as WirelessController_type
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmd_type
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowState_type
from unitree_sdk2py.idl.default import unitree_go_msg_dds__WirelessController_ as WirelessController_msg
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_ as LowCmd_msg
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_ as LowState_msg

TOPIC_WIRELESS = "rt/wirelesscontroller"
TOPIC_LOWCMD = "rt/lowcmd"
TOPIC_LOWSTATE = "rt/lowstate"
DOMAIN_ID = 1
IFACE = "lo"

CTRL_HZ = 400.0
DEADZONE = 0.08
SMOOTH = 0.2
KP_DEFAULT = 35.0
KD_DEFAULT = 1.2
FREQ_BASE = 0.8
STEP_H = 0.05
HIP_AMP = 0.28
KNEE_BIAS = 0.8

JOINT_MAP = {
    "FR": {"HAB": 0, "HIP": 1, "KNEE": 2},
    "FL": {"HAB": 3, "HIP": 4, "KNEE": 5},
    "RR": {"HAB": 6, "HIP": 7, "KNEE": 8},
    "RL": {"HAB": 9, "HIP": 10, "KNEE": 11},
}
LEGS = ["FR", "FL", "RR", "RL"]
PHASE = {"FR": 0.0, "FL": math.pi, "RR": math.pi, "RL": 0.0}

def deadzone(x, dz):
    return 0.0 if abs(x) < dz else x

@dataclass
class JoyCmd:
    lx: float = 0.0
    ly: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    keys: int = 0

class TeleopBridge:
    def __init__(self):
        ChannelFactoryInitialize(DOMAIN_ID, IFACE)
        self.joy = JoyCmd()
        self.q_cur = []
        self.num_motor = None
        self.t = 0.0
        self.dt = 1.0 / CTRL_HZ
        self._lx_f = 0.0
        self._ly_f = 0.0
        self._rx_f = 0.0
        self._ry_f = 0.0
        self.pub_lowcmd = ChannelPublisher(TOPIC_LOWCMD, LowCmd_type)
        self.pub_lowcmd.Init()
        self.sub_joy = ChannelSubscriber(TOPIC_WIRELESS, WirelessController_type)
        self.sub_joy.Init(self._joy_cb, 10)
        self.sub_state = ChannelSubscriber(TOPIC_LOWSTATE, LowState_type)
        self.sub_state.Init(self._state_cb, 10)
        self.ctrl = RecurrentThread(interval=self.dt, target=self._step, name="teleop_bridge")
        self.ctrl.Start()

    def _joy_cb(self, msg: WirelessController_msg):
        self.joy.lx = deadzone(float(msg.lx), DEADZONE)
        self.joy.ly = deadzone(float(msg.ly), DEADZONE)
        self.joy.rx = deadzone(float(msg.rx), DEADZONE)
        self.joy.ry = deadzone(float(msg.ry), DEADZONE)
        self.joy.keys = int(msg.keys)

    def _state_cb(self, msg: LowState_msg):
        if hasattr(msg, "motor_state"):
            n = len(msg.motor_state)
            if not self.q_cur or len(self.q_cur) != n:
                self.q_cur = [0.0] * n
            for i in range(n):
                self.q_cur[i] = float(msg.motor_state[i].q)

    def _step(self):
        a = SMOOTH
        self._lx_f = (1 - a) * self._lx_f + a * self.joy.lx
        self._ly_f = (1 - a) * self._ly_f + a * self.joy.ly
        self._rx_f = (1 - a) * self._rx_f + a * self.joy.rx
        self._ry_f = (1 - a) * self._ry_f + a * self.joy.ry
        self.t += self.dt
        freq = max(0.2, FREQ_BASE + 1.0 * self._ly_f)
        cmd = LowCmd_msg()
        nm = len(cmd.motor_cmd)
        if self.num_motor is None:
            self.num_motor = nm
        for leg in LEGS:
            phase = 2 * math.pi * freq * self.t + PHASE[leg]
            s = 0.5 * (1.0 + math.sin(phase))
            side = +1.0 if leg in ("FR", "RR") else -1.0
            q_hab = 0.10 * self._lx_f + 0.06 * self._rx_f * side
            q_hip = HIP_AMP * math.sin(phase)
            q_knee = KNEE_BIAS - 0.25 * math.cos(phase) + (0.8 * STEP_H if s > 0.6 else 0.0)
            j_hab = JOINT_MAP[leg]["HAB"]
            j_hip = JOINT_MAP[leg]["HIP"]
            j_knee = JOINT_MAP[leg]["KNEE"]
            for j, qdes in [(j_hab, q_hab), (j_hip, q_hip), (j_knee, q_knee)]:
                m = cmd.motor_cmd[j]
                m.q = qdes
                m.dq = 0.0
                m.kp = KP_DEFAULT
                m.kd = KD_DEFAULT
                m.tau = 0.0
        used = {
            JOINT_MAP["FR"]["HAB"], JOINT_MAP["FR"]["HIP"], JOINT_MAP["FR"]["KNEE"],
            JOINT_MAP["FL"]["HAB"], JOINT_MAP["FL"]["HIP"], JOINT_MAP["FL"]["KNEE"],
            JOINT_MAP["RR"]["HAB"], JOINT_MAP["RR"]["HIP"], JOINT_MAP["RR"]["KNEE"],
            JOINT_MAP["RL"]["HAB"], JOINT_MAP["RL"]["HIP"], JOINT_MAP["RL"]["KNEE"],
        }
        for j in range(len(cmd.motor_cmd)):
            if j in used:
                continue
            m = cmd.motor_cmd[j]
            m.q = 0.0
            m.dq = 0.0
            m.kp = 0.0
            m.kd = 0.0
            m.tau = 0.0
        self.pub_lowcmd.Write(cmd)

    def Close(self):
        self.ctrl.Stop()
        self.ctrl.Join()

if __name__ == "__main__":
    br = TeleopBridge()
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        br.Close()