# Granular Terrain Model

An simplified implementation of Granular Media Model proposed in *S. Choi et al., “[Learning quadrupedal locomotion on deformable terrain](https://www.science.org/doi/10.1126/scirobotics.ade2256)”, Science Robotics, 2023*

> **[WARNING]** The folder `/GranularModule` is ONLY for testing, any modification made within this folder WILL NOT affect the granular module used in policy training

## Setup

```bash
conda create -n Granular python=3.10
conda activate Granular
```
Install [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python) and other Dependencies

```bash
pip3 install mujoco noise opencv-python numpy pygame
```

## Usage

You can simulate this module with [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)

### Simulator
```bash
cd ./env/simulate_python
python3 ./unitree_mujoco.py
```

### Quadruped Spawn
In `config.py`, we can set the initial position and Euler angles of the robot:

```python
# Initial position (meters)
START_X = 0.0        # Initial X
START_Y = 0.0        # Initial Y
START_HEIGHT = 2.0   # Initial Z (height)

# Euler angles (radians)
START_ROLL  = 0.0    # roll  (about x-axis)
START_PITCH = 0.0    # pitch (about y-axis)
START_YAW   = 0.0    # yaw   (about z-axis)

```
### Terrain Generation
Navigate to `./env/terrain_tool` and run the following command:
```bash
python3 height_field_generator.py
```
This will give you a height field image, to gain the image type you want, please refer to each generation type method and uncomment them if needed. Then run the following command:
```bash
python3 terrain_generator.py
```
This will give you the terrain MJCF file, which the dog will step on. 

### Keyboard Teleop
Navigate to `./env/simulate_python` and start three terminals, run each command in its terminal in this sequence:
```bash
python3 teleop_keyboard.py
python3 teleop_bridge.py
python3 unitree_mujoco.py
```
Now you can see the dog walking on the terrain. You can make the dog move faster unsing the keyboard. For the detailed operation please refer to `teleop_keyboard.py`.

### Testing Script
First, navigate to `./script` and start the testing script
```bash
python3 name_of_example.py
```
Then, navigate to `./env/simulate_python` to start the simulator
```bash
python3 unitree_mujoco.py
```

## Citation

The content in this folder is built upon [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)