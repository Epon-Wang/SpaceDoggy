import xml.etree.ElementTree as xml_et
import numpy as np
import cv2
import noise

from terrain_para_config import terrain_dict

INPUT_SCENE_PATH = "env/terrain_tool/scene.xml"
OUTPUT_SCENE_PATH = "env/scene_terrain.xml"
GO2_ROBOT_PATH = "env/go2.xml"


# zyx euler angle to quaternion
def euler_to_quat(roll, pitch, yaw):
    cx = np.cos(roll / 2)
    sx = np.sin(roll / 2)
    cy = np.cos(pitch / 2)
    sy = np.sin(pitch / 2)
    cz = np.cos(yaw / 2)
    sz = np.sin(yaw / 2)

    return np.array(
        [
            cx * cy * cz + sx * sy * sz,
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz,
        ],
        dtype=np.float64,
    )


# zyx euler angle to rotation matrix
def euler_to_rot(roll, pitch, yaw):
    rot_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)],
        ],
        dtype=np.float64,
    )

    rot_y = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ],
        dtype=np.float64,
    )
    rot_z = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    return rot_z @ rot_y @ rot_x


# 2d rotate
def rot2d(x, y, yaw):
    nx = x * np.cos(yaw) - y * np.sin(yaw)
    ny = x * np.sin(yaw) + y * np.cos(yaw)
    return nx, ny


# 3d rotate
def rot3d(pos, euler):
    R = euler_to_rot(euler[0], euler[1], euler[2])
    return R @ pos


def list_to_str(vec):
    return " ".join(str(s) for s in vec)


class TerrainGenerator:

    def __init__(self) -> None:
        self.scene = xml_et.parse(INPUT_SCENE_PATH)
        self.robot = xml_et.parse(GO2_ROBOT_PATH)
        self.root = self.scene.getroot()
        self.robotRoot = self.robot.getroot()
        self.worldbody = self.root.find("worldbody")
        self.robotWorldbody = self.robotRoot.find("worldbody")
        self.asset = self.root.find("asset")
    
    def SetGravity(self, gravity=[0.0, 0.0, -9.81]):
        option = xml_et.SubElement(self.root, "option")
        option.attrib["gravity"] = list_to_str(gravity)
        option.tail = "\n"
    
    def SetRobotSpawnPosition(self, position=[0.0, 0.0, 0.445]):
        # Check if base_link already exists in worldbody
        base_link = None
        for body in self.robotWorldbody.findall("body"):
            if body.get("name") == "base_link":
                base_link = body
                break
        
        if base_link is not None:
            base_link.attrib["pos"] = list_to_str(position)
        else:
            raise ValueError("base_link body not found in robot XML.")
        base_link.tail = "\n"

    def _get_or_create_custom(self):
        custom = self.root.find("custom")
        if custom is None:
            custom = xml_et.SubElement(self.root, "custom")
        return custom

    def _set_numeric(self, custom_node, name: str, value):
        """在 <custom> 下写/更新 <numeric name=... data=...>（若存在则覆盖）"""
        node = None
        for n in custom_node.findall("numeric"):
            if n.get("name") == name:
                node = n
                break
        if node is None:
            node = xml_et.SubElement(custom_node, "numeric", name=name)
        node.set("data", str(value))
        node.tail = "\n"

    def AddLunarGMParams(self, overrides: dict | None = None):
        """
        写入“月球近似”的 GM 参数（可用 overrides 覆盖）。
        这些参数与 z, z_dot, z_ddot 结合即可计算 F_GM。
        """
        lunar_defaults = dict(
            gm_theta=0.5236,       # θ≈30° (rad)
            gm_nu=0.40,            # 招募率
            gm_z0=0.0,             # 参考深度
            gm_phi=0.60,           # 体积分数（中等致密）
            gm_rho=3000.0,         # 颗粒密度 kg/m^3（岩屑/玻璃）
            gm_cg=1.0,             # 附加质量系数
            gm_cd=1.2,             # 惯性阻力缩放
            gm_sigma_flat=2.0e5,   # 平面项等效刚度（可按沉陷调参）
            gm_eps_f=1e-4,         # 塑性容差
            gm_sigma_cone=1.0e5,   # 侧锥项等效刚度（可先留低一些）
        )
        if overrides:
            lunar_defaults.update(overrides)

        custom = self._get_or_create_custom()
        for k, v in lunar_defaults.items():
            self._set_numeric(custom, k, v)


    # Add Box to scene
    def AddBox(self,
               position=[1.0, 0.0, 0.0],
               euler=[0.0, 0.0, 0.0], 
               size=[0.1, 0.1, 0.1]):
        geo = xml_et.SubElement(self.worldbody, "geom")
        geo.attrib["pos"] = list_to_str(position)
        geo.attrib["type"] = "box"
        geo.attrib["size"] = list_to_str(
            0.5 * np.array(size))  # half size of box for mujoco
        quat = euler_to_quat(euler[0], euler[1], euler[2])
        geo.attrib["quat"] = list_to_str(quat)
    
    def AddGeometry(self,
               position=[1.0, 0.0, 0.0],
               euler=[0.0, 0.0, 0.0], 
               size=[0.1, 0.1],geo_type="box"):
        
        # geo_type supports "plane", "sphere", "capsule", "ellipsoid", "cylinder", "box"
        geo = xml_et.SubElement(self.worldbody, "geom")
        geo.attrib["pos"] = list_to_str(position)
        geo.attrib["type"] = geo_type
        geo.attrib["size"] = list_to_str(
            0.5 * np.array(size))  # half size of box for mujoco
        quat = euler_to_quat(euler[0], euler[1], euler[2])
        geo.attrib["quat"] = list_to_str(quat)

    def AddReferencePlane(self,
                         position=[0.0, 0.0, 0.05],
                         size=[10, 10, 0.001],
                         rgba=[0, 1, 0, 0.15]):

        geo = xml_et.SubElement(self.worldbody, "geom")
        geo.attrib["name"] = "ref_plane"
        geo.attrib["type"] = "plane"
        geo.attrib["pos"] = list_to_str(position)
        geo.attrib["quat"] = list_to_str([1, 0, 0, 0])
        geo.attrib["size"] = list_to_str(size)
        geo.attrib["rgba"] = list_to_str(rgba)
        geo.attrib["contype"] = str(0)
        geo.attrib["conaffinity"] = str(0)
        geo.attrib["group"] = str(1)

    def AddStairs(self,
                  init_pos=[1.0, 0.0, 0.0],
                  yaw=0.0,
                  width=0.2,
                  height=0.15,
                  length=1.5,
                  stair_nums=10):

        local_pos = [0.0, 0.0, -0.5 * height]
        for i in range(stair_nums):
            local_pos[0] += width
            local_pos[2] += height
            x, y = rot2d(local_pos[0], local_pos[1], yaw)
            self.AddBox([x + init_pos[0], y + init_pos[1], local_pos[2]],
                        [0.0, 0.0, yaw], [width, length, height])

    def AddSuspendStairs(self,
                         init_pos=[1.0, 0.0, 0.0],
                         yaw=1.0,
                         width=0.2,
                         height=0.15,
                         length=1.5,
                         gap=0.1,
                         stair_nums=10):

        local_pos = [0.0, 0.0, -0.5 * height]
        for i in range(stair_nums):
            local_pos[0] += width
            local_pos[2] += height
            x, y = rot2d(local_pos[0], local_pos[1], yaw)
            self.AddBox([x + init_pos[0], y + init_pos[1], local_pos[2]],
                        [0.0, 0.0, yaw],
                        [width, length, abs(height - gap)])

    def AddRoughGround(self,
                       init_pos=[1.0, 0.0, 0.0],
                       euler=[0.0, -0.0, 0.0],
                       nums=[10, 10],
                       box_size=[0.5, 0.5, 0.5],
                       box_euler=[0.0, 0.0, 0.0],
                       separation=[0.2, 0.2],
                       box_size_rand=[0.05, 0.05, 0.05],
                       box_euler_rand=[0.2, 0.2, 0.2],
                       separation_rand=[0.05, 0.05]):

        local_pos = [0.0, 0.0, -0.5 * box_size[2]]
        new_separation = np.array(separation) + np.array(
            separation_rand) * np.random.uniform(-1.0, 1.0, 2)
        for i in range(nums[0]):
            local_pos[0] += new_separation[0]
            local_pos[1] = 0.0
            for j in range(nums[1]):
                new_box_size = np.array(box_size) + np.array(
                    box_size_rand) * np.random.uniform(-1.0, 1.0, 3)
                new_box_euler = np.array(box_euler) + np.array(
                    box_euler_rand) * np.random.uniform(-1.0, 1.0, 3)
                new_separation = np.array(separation) + np.array(
                    separation_rand) * np.random.uniform(-1.0, 1.0, 2)

                local_pos[1] += new_separation[1]
                pos = rot3d(local_pos, euler) + np.array(init_pos)
                self.AddBox(pos, new_box_euler, new_box_size)

    def AddPerlinHeighField(
            self,
            position=[1.0, 0.0, 0.0],  # position
            euler=[0.0, -0.0, 0.0],  # attitude
            size=[1.0, 1.0],  # width and length
            height_scale=1.0,  # max height
            negative_height=0.2,  # height in the negative direction of z axis
            image_width=128,  # height field image size
            img_height=128,
            smooth=100.0,  # smooth scale
            perlin_octaves=6,  # perlin noise parameter
            perlin_persistence=0.5,
            perlin_lacunarity=2.0,
            output_hfield_image="height_field.png"):

        # Generating height field based on perlin noise
        terrain_image = np.zeros((img_height, image_width), dtype=np.uint8)
        for y in range(image_width):
            for x in range(image_width):
                # Perlin noise
                noise_value = noise.pnoise2(x / smooth,
                                            y / smooth,
                                            octaves=perlin_octaves,
                                            persistence=perlin_persistence,
                                            lacunarity=perlin_lacunarity)
                terrain_image[y, x] = int((noise_value + 1) / 2 * 255)

        cv2.imwrite("../" + output_hfield_image,
                    terrain_image)

        hfield = xml_et.SubElement(self.asset, "hfield")
        hfield.attrib["name"] = "perlin_hfield"
        hfield.attrib["size"] = list_to_str(
            [size[0] / 2.0, size[1] / 2.0, height_scale, negative_height])
        hfield.attrib["file"] = "../" + output_hfield_image

        geo = xml_et.SubElement(self.worldbody, "geom")
        geo.attrib["type"] = "hfield"
        geo.attrib["hfield"] = "perlin_hfield"
        geo.attrib["pos"] = list_to_str(position)
        quat = euler_to_quat(euler[0], euler[1], euler[2])
        geo.attrib["quat"] = list_to_str(quat)

        # # Add Dynamics to the height field
        # geo.attrib["margin"]   = "0.03"              # softer “skin” around surface
        # geo.attrib["condim"]   = "3"
        # geo.attrib["friction"] = "1.2 0.06 0.55"
        # # spring-damper contact model: softer & underdamped = bouncy
        # geo.attrib["solref"]   = "0.05 0.35"         # (timeconst=0.05, damping ratio=0.35)
        # geo.attrib["solimp"]   = "0.70 0.97 0.06 0.5 2.2"  # softer start, thicker cushion

        # geo.attrib["margin"]   = "0.03"                 # modest soft skin
        # geo.attrib["condim"]   = "3"
        # geo.attrib["friction"] = "5.0 0.5 2.0"       # more stick (slide/spin/roll)

        # # Overdamped viscoelastic contact → sticky, little to no rebound
        # geo.attrib["solref"]   = "0.10 1.40"             # timeconst=0.08s, damping ratio>1 (overdamped)
        # geo.attrib["solimp"]   = "0.75 0.95 0.05 0.55 2.0"  # softer onset + thicker cushion

    def AddHeighFieldFromImage(
            self,
            position=[0.0, 0.0, 0.0],  # position
            euler=[0.0, -0.0, 0.0],  # attitude
            size=[10.0, 10.0],  # width and length
            height_scale=1.0,  # max height
            negative_height=0.1,  # height in the negative direction of z axis
            input_img='/home/epon/SpaceDoggy/env/height_field.png',
            output_hfield_image="height_field.png",
            image_scale=[1.0, 1.0],  # reduce image resolution
            invert_gray=False):

        input_image = cv2.imread(input_img)  # change to your image path

        width = int(input_image.shape[1] * image_scale[0])
        height = int(input_image.shape[0] * image_scale[1])
        resized_image = cv2.resize(input_image, (width, height),
                                   interpolation=cv2.INTER_AREA)
        terrain_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        if invert_gray:
            terrain_image = 255 - position
        cv2.imwrite("../" + output_hfield_image,
                    terrain_image)

        hfield = xml_et.SubElement(self.asset, "hfield")
        hfield.attrib["name"] = "image_hfield"
        hfield.attrib["size"] = list_to_str(
            [size[0] / 2.0, size[1] / 2.0, height_scale, negative_height])
        hfield.attrib["file"] = "../" + output_hfield_image

        geo = xml_et.SubElement(self.worldbody, "geom")
        geo.attrib["type"] = "hfield"
        geo.attrib["hfield"] = "image_hfield"
        geo.attrib["pos"] = list_to_str(position)
        quat = euler_to_quat(euler[0], euler[1], euler[2])
        geo.attrib["quat"] = list_to_str(quat)

        # # Add Dynamics to the height field
        # geo.attrib["margin"]   = "0.03"              # softer “skin” around surface
        # geo.attrib["condim"]   = "3"
        # geo.attrib["friction"] = "1.2 0.06 0.55"
        # # spring-damper contact model: softer & underdamped = bouncy
        # geo.attrib["solref"]   = "0.05 0.35"         # (timeconst=0.05, damping ratio=0.35)
        # geo.attrib["solimp"]   = "0.70 0.97 0.06 0.5 2.2"  # softer start, thicker cushion

        # geo.attrib["margin"]   = "0.03"                 # modest soft skin
        # geo.attrib["condim"]   = "3"
        # geo.attrib["friction"] = "5.0 0.5 2.0"       # more stick (slide/spin/roll)

        # # Overdamped viscoelastic contact → sticky, little to no rebound
        # geo.attrib["solref"]   = "0.10 1.40"             # timeconst=0.08s, damping ratio>1 (overdamped)
        # geo.attrib["solimp"]   = "0.75 0.95 0.05 0.55 2.0"  # softer onset + thicker cushion

    def Save(self):
        self.scene.write(OUTPUT_SCENE_PATH)
        self.robot.write(GO2_ROBOT_PATH)

    def DisableFloorCollision(self):
        """测试 GM 外力时，关闭 'floor' 的接触，避免与接触力叠加。"""
        for g in self.worldbody.findall("geom"):
            if g.get("name") == "floor":
                g.set("contype", "0")
                g.set("conaffinity", "0")
                g.tail = "\n"

if __name__ == "__main__":
    tg = TerrainGenerator()

    tg.SetGravity()
    
    # Set robot spawn position
    tg.SetRobotSpawnPosition(position=[10.0, 10.0, 0.445])
    
    # Add reference plane (green, semi-transparent)
    tg.AddReferencePlane(
        position=[0, 0, 0.7],
        size=[10, 10, 0.001],
        rgba=[0, 1, 0, 0.15])
    
    # Add lunar GM params
    # tg.DisableFloorCollision()
    tg.AddLunarGMParams() # use defaults

    # Add terrains based on terrain_dict
    # tg.AddLunarGMParams(overrides=terrain_dict) 

    # Perlin height field
    # tg.AddPerlinHeighField(position=[0.0, 0.0, 0.0], size=[10.0, 10.0])
    # tg.AddHeighFieldFromImage(
    #     position=[0.0, 0.0, 0.0],
    #     size=[10.0,10.0])

    tg.AddHeighFieldFromImage()

    tg.Save()