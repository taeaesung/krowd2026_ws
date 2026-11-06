# Atomic trajectory contract from planning to control.
uint8 STOP_AT_END=0
uint8 PASS_THROUGH=1
uint8 FOLLOW_UNTIL_REPLACED=2

string trajectory_id
builtin_interfaces/Time generation_time
builtin_interfaces/Time valid_until
uint8 goal_mode
nav_msgs/Path path

# One value per path pose. Pose orientation supplies yaw.
# Curvature units: 1/m. Target speed units: m/s.
float64[] curvatures
float64[] target_speeds
