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

# Parameters
REF_PLANE_NAME = "ref_plane"
GND_PLANE_NAME = "floor"
FOOT_GEOMS_NAMES = ["FL", "FR", "RL", "RR"]  # four feet geom names from go2.xml
RECORD_DURATION = 20.0  # in seconds, for data recording

# Data recording setup
rec_init_t =    None
isRecording =   False

# Get geom ids for the ground
ref_plane_id =  mj_model.geom(REF_PLANE_NAME).id
gnd_plane_id =  mj_model.geom(GND_PLANE_NAME).id
foot_ids =      {name: mj_model.geom(name).id for name in FOOT_GEOMS_NAMES}

# Granular media module initialization
granular_modules = GranularModules(
    refPlaneID= ref_plane_id, 
    gndPlaneID= gnd_plane_id, 
    footIDs=    foot_ids, 
    footNames=  FOOT_GEOMS_NAMES
)

# Foot Data Storage:
#  - Distance, Velocity, Acceleration to Reference Plane
footData = {
    'time':                 [],
    'z':                    {name: [] for name in FOOT_GEOMS_NAMES},            # {foot_name: [distances]}
    'z_dot':                {name: [] for name in FOOT_GEOMS_NAMES},            # {foot_name: [velocities]}
    'z_ddot':               {name: [] for name in FOOT_GEOMS_NAMES},            # {foot_name: [accelerations]}
    'contact_force':        {name: [] for name in FOOT_GEOMS_NAMES},            # {foot_name: [force magnitudes]}
    'contact_force_vec':    {name: [] for name in FOOT_GEOMS_NAMES}             # {foot_name: [force vectors]}
}

# Foot Data Storage:
#  - F_GM normal to ground plane
current_contact_forces = {name: np.zeros(3) for name in FOOT_GEOMS_NAMES}

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
    mj_model.vis.scale.contactheight = 0.02  # Contact point height
    mj_model.vis.scale.forcewidth = 0.05     # Force arrow width (thickness)
    mj_model.vis.map.force = 0.03            # Force arrow length scale (MAIN PARAMETER for arrow length)


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
    global mj_data, mj_model, current_contact_forces

    ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)
    unitree = UnitreeSdk2Bridge(mj_model, mj_data)

    if config.USE_JOYSTICK:
        unitree.SetupJoystick(device_id=0, js_type=config.JOYSTICK_TYPE)
    if config.PRINT_SCENE_INFORMATION:
        unitree.PrintSceneInformation()

    while viewer.is_running():
        step_start = time.perf_counter()

        locker.acquire()

        # Apply granular media forces to each foot
        # =================================================================================
        # params_FL = granular_modules.get_GM_ParamsFromModel(mj_model, foot_ids["FL"])
        # params_all = {name: params_FL for name in foot_ids.keys()}

        # forces = granular_modules.compute_GM_AllFoot(
        #     mj_model, mj_data, paramsPerFoot=params_all, monitor=False
        # )

        # # F_GM Plotting
        # mj_data.xfrc_applied[:] = 0.0
        # for foot_name, gid in foot_ids.items():
        #     body_id = mj_model.geom_bodyid[gid]
        #     f_world = forces[foot_name]
        #     mj_data.xfrc_applied[body_id, :3] += f_world

        #     # Store current contact forces for recording
        #     current_contact_forces[foot_name] = f_world.copy()
        # =================================================================================

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
    global rec_init_t, isRecording

    while viewer.is_running():
        locker.acquire()
        # Apply options for contact force visualization 
        if vis_options is not None:
            with viewer.lock():
                viewer.opt.flags[:] = vis_options.flags[:]
        
        # =================================================================================
        # Monitor Foot-Ground Contact Status
        # contact_info = []
        # for foot_name, foot_id in foot_ids.items():
        #     hasContact, _ = granular_modules.isFootFloorContact(mj_data, foot_id)
        #     contact_info.append(f"{foot_name}: {'Contact' if hasContact else 'No Contact'}")
        # print("Foot Contacts: " + ", ".join(contact_info))

        # =================================================================================
        # Monitor Foot Normal Data Status: z, z_dot, z_ddot
        # 1. `monitor = False` if you felt it too verbose
        # 2. please refer to granular_module.py for the meanings of these variables
        dist = granular_modules.distPlane2Foot(mj_data, monitor=True)
        velAcc = granular_modules.velAccPlane2Foot(mj_model, mj_data, monitor=False)

        # --------------------------------------------------------------------------------
        # Foot Normal Data recording logic
        # curr_t = time.time()
        
        # if not isRecording:
        #     # Start recording
        #     rec_init_t = curr_t
        #     isRecording = True
        #     print("Started recording foot data for 10 seconds...")
        
        # if isRecording:
        #     elapsed_t = curr_t - rec_init_t

        #     # Data Recording
        #     if elapsed_t <= RECORD_DURATION:
        #         footData['time'].append(elapsed_t)
                
        #         # Record z
        #         for foot_name, distData in dist.items():
        #             footData['z'][foot_name].append(distData)
                
        #         # Record z_dot, z_ddot
        #         for foot_name, velAccData in velAcc.items():
        #             footData['z_dot'][foot_name].append(velAccData['z_dot'])
        #             footData['z_ddot'][foot_name].append(velAccData['z_ddot'])
                
        #         # Record contact forces
        #         for foot_name in FOOT_GEOMS_NAMES:
        #             force_vec = current_contact_forces[foot_name]
        #             force_mag = np.linalg.norm(force_vec)
        #             footData['contact_force'][foot_name].append(force_mag)
        #             footData['contact_force_vec'][foot_name].append(force_vec.copy())

        #     # Data Visualization
        #     else:
        #         isRecording = False
        #         granular_modules.plotDataPlane2Foot(footData)
        #         granular_modules.plot_GM_OnFoot(footData)
        
        # =================================================================================



        viewer.sync()
        locker.release()
        time.sleep(config.VIEWER_DT)


if __name__ == "__main__":
    viewer_thread = Thread(target=PhysicsViewerThread)
    sim_thread = Thread(target=SimulationThread)

    viewer_thread.start()
    sim_thread.start()
