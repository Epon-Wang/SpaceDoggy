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
        self.counter = 0  # Initialize counter as instance variable
    


    def distPlane2Foot(
            self,
            mjData: mujoco.MjData, 
        ) -> dict:
        """
        Input:
            - mjData: mujoco.MjData object
        Output:
            - normal distances from the feet of Go2 to the plane
        """

        def planeCenterNormal(
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

        n, p0 = planeCenterNormal(mjData)

        distances = {}
        for foot_name, foot_id in self.footIDs.items():
            # Get foot position
            foot_pos = mjData.geom_xpos[foot_id].copy()
            
            # Calculate normal distance to plane
            distance = np.dot(foot_pos - p0, n)
            distances[foot_name] = distance
        
        # Print distances every few frames to avoid spam
        self.counter += 1

        if self.counter % 1 == 0:  # Print every frame
            print(f"Foot distances to plane: FL={distances['FL']:.4f}, FR={distances['FR']:.4f}, RL={distances['RL']:.4f}, RR={distances['RR']:.4f}")
        
        return distances