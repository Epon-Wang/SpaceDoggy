import mujoco
import numpy as np
import math


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
    
    # Read the parameters from the model
    def _num_scalar(self, mjModel: mujoco.MjModel, name: str) -> float:
        """读 <custom><numeric name=... data=...> 的标量；缺就抛错。"""
        idx = mujoco.mj_name2id(mjModel, mujoco.mjtObj.mjOBJ_NUMERIC, name)
        if idx < 0:
            raise KeyError(f"Missing <numeric name='{name}'> in model.")
        adr = mjModel.numeric_adr[idx]
        sz  = mjModel.numeric_size[idx]
        if sz != 1:
            raise ValueError(f"<numeric name='{name}'> must be scalar, got size={sz}.")
        return float(mjModel.numeric_data[adr])

    def _build_gm_params_from_model(self, mjModel: mujoco.MjModel, foot_geom_id: int) -> dict:
        """从模型读取所有 GM 参数 + 圆脚半径换算 A,P。"""
        # 圆脚半径（MuJoCo: geom_size[0] 就是半径）
        R = float(mjModel.geom_size[foot_geom_id, 0])
        if R <= 0:
            raise ValueError(f"Invalid foot radius: {R}")
        A = math.pi * R * R
        P = 2.0 * math.pi * R

        # 其余来自 <numeric>
        return dict(
            A=A, P=P,
            theta=self._num_scalar(mjModel, "gm_theta"),
            nu=   self._num_scalar(mjModel, "gm_nu"),
            z0=   self._num_scalar(mjModel, "gm_z0"),
            phi=  self._num_scalar(mjModel, "gm_phi"),
            rho=  self._num_scalar(mjModel, "gm_rho"),
            c_g=  self._num_scalar(mjModel, "gm_cg"),
            c_d=  self._num_scalar(mjModel, "gm_cd"),
            sigma_flat=self._num_scalar(mjModel, "gm_sigma_flat"),
            epsilon_f=self._num_scalar(mjModel, "gm_eps_f"),
            sigma_cone=(
                self._num_scalar(mjModel, "gm_sigma_cone")
                if mujoco.mj_name2id(mjModel, mujoco.mjtObj.mjOBJ_NUMERIC, "gm_sigma_cone") >= 0
                else 0.0
            ),
            ema_alpha_base=0.8,   # 平滑（可按需改）
            ema_rate_c_r=0.0,
        )

        
    def compute_gm_force(
    self,
    foot_name: str,
    z: float,
    z_dot: float,
    z_ddot: float,
    params: dict,
    use_ema: bool = True,
    update_state: bool = True,
) -> float:
        """
        返回沿参考平面法向的标量 F_GM（单位 N）。
        依赖输入：z, z_dot, z_ddot（你已有），params（第1步构造）。
        """
        # --- 持久状态（每只脚的 z_max、EMA） ---
        if not hasattr(self, "gm_state"):
            self.gm_state = {
                name: {"z_max": -1e9, "F_ema": 0.0, "tau_r": 0.0}
                for name in self.footIDs.keys()
            }
        st = self.gm_state[foot_name]

        # --- 读参数 ---
        A = float(params["A"]);       P = float(params["P"])
        theta = float(params["theta"]); nu = float(params["nu"]); z0 = float(params.get("z0", 0.0))
        phi = float(params["phi"]);   rho = float(params["rho"]); c_g = float(params["c_g"])
        c_d = float(params["c_d"]);   sigma_flat = float(params["sigma_flat"])
        sigma_cone = float(params.get("sigma_cone", 0.0))
        eps_f = float(params.get("epsilon_f", 1e-4))
        ema_alpha_base = float(params.get("ema_alpha_base", 0.8))
        ema_rate_c_r = float(params.get("ema_rate_c_r", 0.0))

        # --- 几何：S11 平面面积 + 原函数 ---
        rh = 2.0 * A / P           # 圆脚：= R
        alpha = nu / math.tan(theta)
        dz = z - z0
        core = max(rh - alpha * dz, 0.0)
        A_flat = math.pi * core * core
        I_flat = math.pi * (
            (rh * rh) * z
            - rh * alpha * (dz ** 2)
            + (alpha * alpha) * (dz ** 3) / 3.0
        )
        # 侧锥项（若暂不启用，可保持 0）
        A_cone = 0.0
        I_cone = 0.0

        # --- 附加质量与导数 ---
        m_a = -c_g * phi * rho * nu * I_flat
        dm_a_dz = -c_g * phi * rho * nu * A_flat
        m_a_dot = dm_a_dz * z_dot

        # --- 势力 + 惯性/阻尼 ---
        F_p = sigma_flat * I_flat + sigma_cone * I_cone
        F_raw = F_p - c_d * m_a_dot * z_dot - m_a * z_ddot

        if z <= 0.0:
            F_act = 0.0
        else:
            F_act = F_raw

        # --- EMA 平滑（可选） ---
        if use_ema:
            if ema_rate_c_r > 0.0 and update_state:
                st["tau_r"] = min(1.0, st["tau_r"] + ema_rate_c_r)
            alpha_ema = ema_alpha_base
            F_out = (1.0 - alpha_ema) * st["F_ema"] + alpha_ema * F_act
            if update_state:
                st["F_ema"] = F_out
        else:
            F_out = F_act

        return float(F_out)

    def compute_gm_forces_for_all_feet(
    self,
    mjModel: mujoco.MjModel,
    mjData: mujoco.MjData,
    params_per_foot: dict | None = None,  # 可给每只脚不同 params
    use_ema: bool = True,
    monitor: bool = False,
) -> dict:
        """
        返回 {foot_name: np.array([Fx,Fy,Fz])}，世界系。
        params_per_foot: 若为 None，则默认用 FL 的半径构造一份 params 给所有脚；
                        若为 dict，形如 {'FL': params_FL, ...}
        """
        # 1) 法向
        n, _ = self.planeCenterNormal(mjData)
        n = np.asarray(n, float)

        # 2) z、z_dot、z_ddot
        z_dict = self.distPlane2Foot(mjData, monitor=False)
        va_dict = self.velAccPlane2Foot(mjModel, mjData, monitor=False)

        # 3) 若未提供每脚参数，则用 FL 的几何构一份通用
        if params_per_foot is None:
            if "FL" not in self.footIDs:
                any_name = next(iter(self.footIDs.keys()))
                base_gid = self.footIDs[any_name]
            else:
                base_gid = self.footIDs["FL"]
            base_params = self._build_gm_params_from_model(mjModel, base_gid)
            params_per_foot = {name: base_params for name in self.footIDs.keys()}

        # 4) 逐脚计算
        forces = {}
        for name, gid in self.footIDs.items():
            z_signed   = float(z_dict[name])                 
            zd_signed  = float(va_dict[name]["z_dot"])
            zdd_signed = float(va_dict[name]["z_ddot"])

            z_pen  = max(0.0, -z_signed)
            z_dot  = -zd_signed
            z_ddot = -zdd_signed
            Fn = self.compute_gm_force(
                name, z_pen, z_dot, z_ddot,
                params=params_per_foot[name],
                use_ema=False,           
                update_state=True,
            )
            forces[name] = Fn * n

        if monitor:
            line = ", ".join([f"{k}={np.dot(v,n):.2f}" for k,v in forces.items()])
            print(f"[GM] Fn (along n): {line}")

        return forces