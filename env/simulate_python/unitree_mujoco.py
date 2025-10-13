import time
import numpy as np
import mujoco
import mujoco.viewer
from threading import Thread
import threading

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py_bridge import UnitreeSdk2Bridge, ElasticBand
from granular_module import GranularModules

import config


locker = threading.Lock()

mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
mj_data = mujoco.MjData(mj_model)


REF_PLANE_NAME = "ref_plane"
FOOT_GEOMS_NAMES = ["FL", "FR", "RL", "RR"]  # four feet geom names from go2.xml

# Get geom ids for the ground
plane_id = mj_model.geom(REF_PLANE_NAME).id
foot_ids = {name: mj_model.geom(name).id for name in FOOT_GEOMS_NAMES}

print("Plane ID:", plane_id)
print("Foot Geom IDs:", foot_ids)

granular_modules = GranularModules(planeID=plane_id, footIDs=foot_ids)

# Setup contact force visualization options
vis_options = None
if config.ENABLE_CONTACT_FORCE_VISUALIZATION:
    print("Contact force visualization is enabled.")
    vis_options = mujoco.MjvOption()
    mujoco.mjv_defaultOption(vis_options)
    vis_options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    vis_options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    vis_options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False  # Set to True to make bodies transparent
    
    # Adjust scales for better contact visualization
    mj_model.vis.scale.contactwidth = 0.3    # Contact point width
    mj_model.vis.scale.contactheight = 0.02   # Contact point height
    mj_model.vis.scale.forcewidth = 0.05      # Force arrow width (thickness)
    mj_model.vis.map.force = 0.03              # Force arrow length scale (MAIN PARAMETER for arrow length)


if config.ENABLE_ELASTIC_BAND:
    elastic_band = ElasticBand()
    if config.ROBOT == "h1" or config.ROBOT == "g1":
        band_attached_link = mj_model.body("torso_link").id
    else:
        band_attached_link = mj_model.body("base_link").id
    viewer = mujoco.viewer.launch_passive(
        mj_model, mj_data, key_callback=elastic_band.MujuocoKeyCallback
    )
else:
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

mj_model.opt.timestep = config.SIMULATE_DT
num_motor_ = mj_model.nu
dim_motor_sensor_ = 3 * num_motor_

time.sleep(0.2)


def SimulationThread():
    global mj_data, mj_model

    ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)
    unitree = UnitreeSdk2Bridge(mj_model, mj_data)

    if config.USE_JOYSTICK:
        unitree.SetupJoystick(device_id=0, js_type=config.JOYSTICK_TYPE)
    if config.PRINT_SCENE_INFORMATION:
        unitree.PrintSceneInformation()

    while viewer.is_running():
        step_start = time.perf_counter()

        locker.acquire()

        if config.ENABLE_ELASTIC_BAND:
            if elastic_band.enable:
                mj_data.xfrc_applied[band_attached_link, :3] = elastic_band.Advance(
                    mj_data.qpos[:3], mj_data.qvel[:3]
                )
        mujoco.mj_step(mj_model, mj_data)

        locker.release()

        time_until_next_step = mj_model.opt.timestep - (
            time.perf_counter() - step_start
        )
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)


def PhysicsViewerThread():
    while viewer.is_running():
        locker.acquire()
        # Apply options for contact force visualization 
        if vis_options is not None:
            with viewer.lock():
                viewer.opt.flags[:] = vis_options.flags[:]

        # Monitor Foot Status: z, z_dot, z_ddot
        # 1. `monitor = False` if you felt it too verbose
        # 2. please refer to granular_module.py for the meanings of these variables
        granular_modules.distPlane2Foot(mj_data, monitor=True)
        granular_modules.velAccPlane2Foot(mj_model, mj_data, monitor=True)

        viewer.sync()
        locker.release()
        time.sleep(config.VIEWER_DT)


if __name__ == "__main__":
    viewer_thread = Thread(target=PhysicsViewerThread)
    sim_thread = Thread(target=SimulationThread)

    viewer_thread.start()
    sim_thread.start()
