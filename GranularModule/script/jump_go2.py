import time
import sys
import numpy as np
import math

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.utils.crc import CRC


stand_joint_pos = np.array([
    0.00571868, 0.608813, -1.21763, -0.00571868, 0.608813, -1.21763,    # front legs
    0.00571868, 0.608813, -1.21763, -0.00571868, 0.608813, -1.21763     # back legs
], dtype=float)


crouch_joint_pos = np.array([
    0.05, 1.5, -2.5, -0.05, 1.5, -2.5,  # front legs
    0.10, 1.6, -2.7, -0.10, 1.6, -2.7   # back legs
], dtype=float)


jump_joint_pos = np.array([
    0.0, 0.3, -0.5, 0.0, 0.3, -0.5,   # front legs
    0.0, 0.2, -0.4, 0.0, 0.2, -0.4    # back legs
], dtype=float)


land_joint_pos = np.array([
    0.05, 1.0, -2.0, -0.05, 1.0, -2.0,  # front legs
    0.05, 1.0, -2.0, -0.05, 1.0, -2.0   # back legs
], dtype=float)



dt = 0.002
running_time = 0.0
crc = CRC()

# LPF Parameters
filter_alpha = 0.2              # Filter coefficient, smaller = smoother = slower responses
prev_joint_pos = np.zeros(12)   # Previous joint positions
prev_joint_vel = np.zeros(12)   # Previous joint velocities

# Timing Settings
tCrouch = 0.5
tJump = 0.2
tFlight = 0.3
tLand = 0.3
tRecover = 0.5
tPause = 1.0
tTotal = tCrouch + tJump + tFlight + tLand + tRecover + tPause

input("Press enter to start")

if __name__ == '__main__':

    if len(sys.argv) < 2:
        ChannelFactoryInitialize(1, "lo")
    else:
        ChannelFactoryInitialize(0, sys.argv[1])

    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()

    cmd = unitree_go_msg_dds__LowCmd_()
    cmd.head[0] = 0xFE
    cmd.head[1] = 0xEF
    cmd.level_flag = 0xFF
    cmd.gpio = 0
    
    for i in range(20):
        cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
        cmd.motor_cmd[i].q = 0.0
        cmd.motor_cmd[i].kp = 0.0
        cmd.motor_cmd[i].dq = 0.0
        cmd.motor_cmd[i].kd = 0.0
        cmd.motor_cmd[i].tau = 0.0

    prev_joint_pos = stand_joint_pos.copy()
    prev_joint_vel = np.zeros(12)


    while True:
        step_start = time.perf_counter()
        running_time += dt
        
        cycle_time = running_time % tTotal
    
        if cycle_time < tCrouch:
            # Phase 1: Crouch
            phase = cycle_time / tCrouch
            smooth_phase = 0.5 * (1 - math.cos(phase * math.pi))  # Smooth transition
            
            for i in range(12):
                target_pos = (1 - smooth_phase) * stand_joint_pos[i] + smooth_phase * crouch_joint_pos[i]
                pos_derivative = (crouch_joint_pos[i] - stand_joint_pos[i]) * 0.5 * math.sin(phase * math.pi) * math.pi / tCrouch
                
                cmd.motor_cmd[i].q = target_pos
                cmd.motor_cmd[i].dq = pos_derivative

                if i < 6:
                    # front legs
                    cmd.motor_cmd[i].kp = 42.0  # stiffness
                    cmd.motor_cmd[i].kd = 2.2   # damping
                else:
                    # back legs
                    cmd.motor_cmd[i].kp = 46.0  # stiffness
                    cmd.motor_cmd[i].kd = 2.4   # damping

                cmd.motor_cmd[i].tau = 0.0
                
        elif cycle_time < tCrouch + tJump:
            # Phase 2: Jump
            phase = (cycle_time - tCrouch) / tJump
            smooth_phase = phase * phase
            
            for i in range(12):
                target_pos = (1 - smooth_phase) * crouch_joint_pos[i] + smooth_phase * jump_joint_pos[i]
                pos_derivative = (jump_joint_pos[i] - crouch_joint_pos[i]) * 2 * phase / tJump
                
                cmd.motor_cmd[i].q = target_pos
                cmd.motor_cmd[i].dq = pos_derivative
                
                if i < 6:
                    # front legs
                    cmd.motor_cmd[i].kp = 65.0  # stiffness
                    cmd.motor_cmd[i].kd = 3.2   # damping
                else:
                    # back legs
                    cmd.motor_cmd[i].kp = 72.0  # stiffness
                    cmd.motor_cmd[i].kd = 3.6   # damping
                
                cmd.motor_cmd[i].tau = 0.0
                
        elif cycle_time < tCrouch + tJump + tFlight:
            # Phase 3: Flight
            for i in range(12):
                cmd.motor_cmd[i].q = jump_joint_pos[i]
                cmd.motor_cmd[i].dq = 0.0
                
                # both legs
                cmd.motor_cmd[i].kp = 30.0  # stiffness
                cmd.motor_cmd[i].kd = 2.0   # damping
                cmd.motor_cmd[i].tau = 0.0
                
        elif cycle_time < tCrouch + tJump + tFlight + tLand:
            # Phase 4: Landing
            phase = (cycle_time - tCrouch - tJump - tFlight) / tLand
            smooth_phase = 0.5 * (1 - math.cos(phase * math.pi))
            
            for i in range(12):
                target_pos = (1 - smooth_phase) * jump_joint_pos[i] + smooth_phase * land_joint_pos[i]
                pos_derivative = (land_joint_pos[i] - jump_joint_pos[i]) * 0.5 * math.sin(phase * math.pi) * math.pi / tLand
                
                cmd.motor_cmd[i].q = target_pos
                cmd.motor_cmd[i].dq = pos_derivative
                
                if i < 6:
                    # front legs
                    cmd.motor_cmd[i].kp = 48.0   # stiffness
                    cmd.motor_cmd[i].kd = 6.2    # damping
                else:
                    # back legs
                    cmd.motor_cmd[i].kp = 52.0   # stiffness
                    cmd.motor_cmd[i].kd = 6.5    # damping
                
                cmd.motor_cmd[i].tau = 0.0
                
        elif cycle_time < tCrouch + tJump + tFlight + tLand + tRecover:
            # Phase 5: Standing Recovery
            phase = (cycle_time - tCrouch - tJump - tFlight - tLand) / tRecover
            smooth_phase = 0.5 * (1 - math.cos(phase * math.pi))
            
            for i in range(12):
                target_pos = (1 - smooth_phase) * land_joint_pos[i] + smooth_phase * stand_joint_pos[i]
                pos_derivative = (stand_joint_pos[i] - land_joint_pos[i]) * 0.5 * math.sin(phase * math.pi) * math.pi / tRecover
                
                cmd.motor_cmd[i].q = target_pos
                cmd.motor_cmd[i].dq = pos_derivative
                
                # both legs
                cmd.motor_cmd[i].kp = 35.0   # stiffness
                cmd.motor_cmd[i].kd = 2.5    # damping
                cmd.motor_cmd[i].tau = 0.0
                
        else:
            # Phase 6: Pause
            for i in range(12):
                cmd.motor_cmd[i].q = stand_joint_pos[i]
                cmd.motor_cmd[i].dq = 0.0

                # both legs
                cmd.motor_cmd[i].kp = 50.0   # stiffness
                cmd.motor_cmd[i].kd = 2.0    # damping
                cmd.motor_cmd[i].tau = 0.0

        # LPF control signal smoothing
        for i in range(12):
            filtered_pos = filter_alpha * cmd.motor_cmd[i].q + (1 - filter_alpha) * prev_joint_pos[i]
            filtered_vel = filter_alpha * cmd.motor_cmd[i].dq + (1 - filter_alpha) * prev_joint_vel[i]
            
            cmd.motor_cmd[i].q = filtered_pos
            cmd.motor_cmd[i].dq = filtered_vel
            
            prev_joint_pos[i] = filtered_pos
            prev_joint_vel[i] = filtered_vel

        cmd.crc = crc.Crc(cmd)
        pub.Write(cmd)

        time_until_next_step = dt - (time.perf_counter() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        