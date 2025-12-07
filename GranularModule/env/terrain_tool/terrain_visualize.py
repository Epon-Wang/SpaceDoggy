# Viewer for terrain XML file
import mujoco
import mujoco.viewer

xml_path = "/home/epon/SpaceDoggy/env/scene_terrain.xml"  

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Press ESC to quit.")
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
