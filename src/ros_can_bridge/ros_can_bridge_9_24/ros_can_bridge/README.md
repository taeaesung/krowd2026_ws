# ros_can_bridge

Bidirectional ROS2 and SocketCAN bridge.

## Default topics

- `/brake`: `std_msgs/msg/Bool`
- `/target_speed`: `std_msgs/msg/Float32`, km/h
- `/target_angle`: `std_msgs/msg/Float32`, radian
- `/wheel_sensor`: `ros_can_bridge_msgs/msg/WheelSpeedStamped`, m/s
- `/encoder`: `ros_can_bridge_msgs/msg/EncoderAngleStamped`, radian

## Default CAN frames

- `0x100`: brake, `data[0] = 0 or 1`, `data[1:8] = 0`
- `0x110`: target speed, `int16 little endian`, `km/h * 100`
- `0x120`: target angle, `int16 little endian`, `rad * 100`
- `0x200`: average rear wheel speed, `int16 little endian`, `km/h * 100`; the bridge divides by 360 for ROS m/s
- `0x201`: steering encoder angle, `int16 little endian`, `rad * 100`

Both directions run at 50 Hz by default. The Drive and Steer Teensy programs
publish feedback frames at 50 Hz. Bytes 2 through 7 are reserved and currently
zero. `can_to_ros_node` uses the Linux SocketCAN receive timestamp for the ROS
message header, so the Teensy payload does not contain a timestamp.

The speed command has a 0.2 second watchdog. If speed messages stop, only the
speed command is forced to zero. Applying the brake immediately sends speed zero
and clears the previous speed command. After releasing the brake, a new speed
message is required before acceleration resumes.

## CAN setup

```bash
sudo ip link set can0 type can bitrate 500000
sudo ip link set can0 up
```

If `can0` is already configured:

```bash
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 500000
sudo ip link set can0 up
```

## Build and run

```bash
cd ~/krowd_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select ros_can_bridge_msgs ros_can_bridge
source install/setup.bash
ros2 run ros_can_bridge ros_to_can_node
```

Run CAN feedback reception in another terminal:

```bash
cd ~/krowd_ws
source install/setup.bash
ros2 run ros_can_bridge can_to_ros_node
```

## Test topics

```bash
ros2 topic pub /target_angle std_msgs/msg/Float32 "{data: 0.5}" -1
ros2 topic pub /target_speed std_msgs/msg/Float32 "{data: 2.3}" -1
ros2 topic pub /brake std_msgs/msg/Bool "{data: true}" -1
ros2 topic echo /wheel_sensor
ros2 topic echo /encoder
```

## Parameters

ROS to CAN example:

```bash
ros2 run ros_can_bridge ros_to_can_node --ros-args \
  -p can_interface:=can0 \
  -p send_rate_hz:=50.0 \
  -p command_timeout_sec:=0.2 \
  -p angle_topic:=/target_angle \
  -p speed_topic:=/target_speed \
  -p brake_topic:=/brake \
  -p angle_scale:=100.0 \
  -p speed_scale:=100.0
```

CAN to ROS example:

```bash
ros2 run ros_can_bridge can_to_ros_node --ros-args \
  -p can_interface:=can0 \
  -p poll_rate_hz:=50.0 \
  -p wheel_sensor_topic:=/wheel_sensor \
  -p encoder_topic:=/encoder \
  -p wheel_sensor_can_id:=0x200 \
  -p encoder_can_id:=0x201 \
  -p wheel_sensor_scale:=360.0 \
  -p encoder_scale:=100.0 \
  -p max_frames_per_poll:=100
```
