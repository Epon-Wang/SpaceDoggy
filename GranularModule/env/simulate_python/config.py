ROBOT = "go2" # Robot name, "go2", "b2", "b2w", "h1", "go2w", "g1" 
# Please change to your directory
ROBOT_SCENE = "env/scene_terrain.xml" # Robot scene
DOMAIN_ID = 1 # Domain id
INTERFACE = "lo" # Interface 

USE_JOYSTICK = 0 # Whether to use Joystick or not
JOYSTICK_TYPE = "xbox" # support "xbox" and "switch" gamepad layout
JOYSTICK_DEVICE = 0 # Joystick number

PRINT_SCENE_INFORMATION = True # Print link, joint and sensors information of robot
ENABLE_ELASTIC_BAND = False # Virtual spring band, used for lifting h1
ENABLE_CONTACT_FORCE_VISUALIZATION = True # Enable contact force visualization


SIMULATE_DT = 0.005  # Need to be larger than the runtime of viewer.sync()
VIEWER_DT = 0.02  # 50 fps for viewer

# Initial Position
START_X = 0.0       # Initial X
START_Y = 0.0       # Initial Y
START_HEIGHT = 1.1  # Initial Z

# Euler Angle (Radian)
START_ROLL = 0.0
START_PITCH = 0.0
START_YAW = 0.0