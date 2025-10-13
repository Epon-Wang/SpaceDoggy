import mujoco
import numpy as np


class GranularModules:
    def __init__(
            self, 
            planeID: int,   # plane geom ID
            footIDs: dict   # dictionary of foot geom Names & IDs
        ) -> None:
        self.planeID = planeID
        self.footIDs = footIDs

        self.prev_zdot = {name: 0.0 for name in footIDs.keys()}
        self.prev_time = None
    


    def planeCenterNormal(
            self,
            mjData: mujoco.MjData
        ) -> tuple:
        """
        Input:
            - mjData: mujoco.MjData object
        Output:
            - normal vector & center of the plane
        """
        # Rotation Matrix of the reference plane
        R = np.array(mjData.geom_xmat[self.planeID]).reshape(3, 3)

        # 3rd column of R = z-axis vector = normal vector
        n = R[:, 2]
        
        # Center of the plane
        p0 = mjData.geom_xpos[self.planeID].copy()

        return n, p0
    


    def distPlane2Foot(
            self,
            mjData:     mujoco.MjData, 
            monitor:    bool = False
        ) -> dict:
        """
        Input:
            - mjData:   mujoco.MjData object
            - monitor:  print the distances for monitoring purposes
        Output:
            - normal distances from the feet of Go2 to the plane
        Note:
            - Output = z
        """

        n, p0 = self.planeCenterNormal(mjData)

        distances = {}
        # Compute normal distance to reference plane for each foot
        for foot_name, foot_id in self.footIDs.items():
            # Get foot position
            foot_pos = mjData.geom_xpos[foot_id].copy()

            distance = (foot_pos - p0) @ n
            distances[foot_name] = distance

        if monitor:
            print("==================================")
            print(f"Foot z:      FL={distances['FL']:.4f}, FR={distances['FR']:.4f}, RL={distances['RL']:.4f}, RR={distances['RR']:.4f}")
        
        return distances
    


    def velAccPlane2Foot(
            self,
            mjModel:    mujoco.MjModel, 
            mjData:     mujoco.MjData,
            monitor:    bool = False
        ) -> dict:
        """
        Input:
            - mjModel:  mujoco.MjModel object
            - mjData:   mujoco.MjData object
            - monitor:  print the velocities & accelerations for monitoring purposes
        Output:
            - normal velocities & accelerations from the feet in global frame
        Note:
            - Output = z_dot, z_ddot
        """

        n, _ = self.planeCenterNormal(mjData)

        # compute dt
        dt = None
        t = float(mjData.time)
        if self.prev_time is not None:
            dt = t - self.prev_time
        else:
            dt = None

        velAcc = {}
        for foot_name, foot_id in self.footIDs.items():
            # end-effector(foot) jacobian
            jacp = np.zeros((3, mjModel.nv))
            _ = np.zeros((3, mjModel.nv)) # we do not need angular jacobian
            mujoco.mj_jacGeom(mjModel, mjData, jacp, _, foot_id)

            # linear velocity under global frame
            lin_v = jacp @ mjData.qvel
            # project to normal linear velocity
            z_dot = float(lin_v @ n)

            if dt is None or dt <= 0.0:
                z_ddot = 0.0
            else:
                z_ddot = (z_dot - self.prev_zdot[foot_name]) / dt

            velAcc[foot_name] = {"z_dot": z_dot, "z_ddot": z_ddot}
            self.prev_zdot[foot_name] = z_dot

        self.prev_time = t

        if monitor:
            print("----------------------------------")
            print("Foot z_dot:  " + ", ".join([f"{k}={v['z_dot']:.4f}" for k, v in velAcc.items()]))
            print("Foot z_ddot: " + ", ".join([f"{k}={v['z_ddot']:.4f}" for k, v in velAcc.items()]))

        return velAcc