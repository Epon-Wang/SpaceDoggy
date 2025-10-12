# SpaceDoggy
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)


Quadruped Locomotion Under Low-Gravity Environment

Course Project for ROB-GY 7863A Planning, Learning, and Control for Autonomous Space Robots

### Team Member
- [Zihan Liu](https://github.com/GuoZheXinDeGuang)

- [Yipeng Wang](https://github.com/Epon-Wang)
### Preparation
Before setting up the environment, please do the following to install the unitree_sdk2_python:
```bash
cd ~
sudo apt install python3-pip
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
pip3 install -e .
```
### Environment Setup
1. Conda Environment
    ```bash
    conda create -n Doggy python=3.10
    conda activate Doggy
    ```
2. Install [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python)

2. Other Dependencies
    ```bash
    pip3 install mujoco noise opencv-python numpy pygame
    ```


### Test Simulator
```bash
cd ./env/simulate_python
python3 ./unitree_mujoco.py
```
### Parameters
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

### Teleop
Navigate to `./env/simulate_python` and start three terminals, run each command in its terminal in this sequence:
```bash
python3 teleop_keyboard.py
```
```bash
python3 teleop_bridge.py
```
```bash
python3 unitree_mujoco.py
```
Now you can see the dog walking on the terrain. You can make the dog move faster unsing the keyboard. For the detailed operation please refer to `teleop_keyboard.py`.