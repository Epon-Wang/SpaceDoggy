# SpaceDoggy
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)


Quadruped Locomotion Under Low-Gravity Environment

Course Project for ROB-GY 7863A Planning, Learning, and Control for Autonomous Space Robots

### Team Member
- [Zihan Liu](https://github.com/GuoZheXinDeGuang)

- [Yipeng Wang](https://github.com/Epon-Wang)

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
