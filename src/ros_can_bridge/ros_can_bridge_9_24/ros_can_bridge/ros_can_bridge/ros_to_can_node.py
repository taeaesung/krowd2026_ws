import math
import struct
import time

import can
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from ros_can_bridge.can_link import (
    DEAD_SOCKET_ERRNOS,
    DEVICE_CHECK_INTERVAL_NS,
    RECONNECT_INTERVAL_NS,
    error_errno,
    read_ifindex,
)
from std_msgs.msg import Bool, Float32


INT16_MIN = -32768
INT16_MAX = 32767
ERROR_LOG_INTERVAL_NS = 5_000_000_000


class RosToCanNode(Node):
    def __init__(self):
        """파라미터와 subscription을 생성하되 아직 SocketCAN은 열지 않는다."""
        super().__init__("ros_to_can_node")
        self.bus = None
        self.bus_ifindex = None
        self.timer = None
        self._send_error_state = {}
        self._last_device_check_ns = None
        self._last_reconnect_ns = None
        self._reconnect_error_active = False
        self._last_reconnect_error_log_ns = None

        self.declare_parameter("can_interface", "can0")
        self.declare_parameter("send_rate_hz", 50.0)
        self.declare_parameter("command_timeout_sec", 0.2)
        self.declare_parameter("angle_topic", "/target_angle")
        # /target_speed 단위는 km/h. 여기서 단위 변환 없이 x speed_scale(100) 해서
        # CAN 0x110 으로 보내고, 구동 MCU 가 km/h x 100 으로 읽는다.
        # 발행하는 쪽(MPC can_adapter, krowd_path_tracking)이 m/s -> km/h 변환을 한다.
        self.declare_parameter("speed_topic", "/target_speed")
        self.declare_parameter("brake_topic", "/brake")
        self.declare_parameter("steering_center_topic", "/steering_center")
        self.declare_parameter("angle_can_id", 0x120)
        self.declare_parameter("speed_can_id", 0x110)
        self.declare_parameter("brake_can_id", 0x100)
        self.declare_parameter("steering_center_can_id", 0x130)
        self.declare_parameter("steering_center_duration_sec", 3.0)
        self.declare_parameter("angle_scale", 100.0)
        self.declare_parameter("speed_scale", 100.0)

        self.can_interface = self.get_parameter("can_interface").value
        self.angle_can_id = self.get_parameter("angle_can_id").value
        self.speed_can_id = self.get_parameter("speed_can_id").value
        self.brake_can_id = self.get_parameter("brake_can_id").value
        self.steering_center_can_id = self.get_parameter(
            "steering_center_can_id"
        ).value
        self.steering_center_duration_sec = self.get_parameter(
            "steering_center_duration_sec"
        ).value
        self.angle_scale = self.get_parameter("angle_scale").value
        self.speed_scale = self.get_parameter("speed_scale").value
        self.command_timeout_sec = self.get_parameter("command_timeout_sec").value

        self.target_angle = 0.0
        self.target_speed = 0.0
        self.brake = False
        self.last_speed_time = None
        self.speed_command_ready = False
        self.steering_center_end_ns = None

        qos = QoSProfile(depth=1)
        self.create_subscription(
            Float32,
            self.get_parameter("angle_topic").value,
            self._angle_callback,
            qos,
        )
        self.create_subscription(
            Float32,
            self.get_parameter("speed_topic").value,
            self._speed_callback,
            qos,
        )
        self.create_subscription(
            Bool,
            self.get_parameter("brake_topic").value,
            self._brake_callback,
            qos,
        )
        self.create_subscription(
            Bool,
            self.get_parameter("steering_center_topic").value,
            self._steering_center_callback,
            qos,
        )

        self.send_rate_hz = self.get_parameter("send_rate_hz").value

    def start(self):
        """SocketCAN을 열고 50 Hz 명령 송신 timer를 시작한다."""
        self._open_bus()
        try:
            self.timer = self.create_timer(
                1.0 / self.send_rate_hz,
                self._timer_callback,
            )
        except Exception:
            self._shutdown_bus_safely(log_error=False)
            raise
        self.get_logger().info(
            "ROS to CAN bridge started: "
            f"interface={self.can_interface}, rate={self.send_rate_hz} Hz, "
            f"speed_timeout={self.command_timeout_sec} s"
        )

    def _open_bus(self):
        """SocketCAN을 열고 연결된 장치의 ifindex를 함께 기록한다."""
        bus = can.interface.Bus(
            channel=self.can_interface,
            interface="socketcan",
        )
        self.bus = bus
        self.bus_ifindex = read_ifindex(self.can_interface)
        # 장치 확인과 재연결 주기는 sim time 에 영향받지 않아야 하므로
        # ROS clock 이 아니라 monotonic clock 을 쓴다.
        self._last_device_check_ns = time.monotonic_ns()

    def _angle_callback(self, msg):
        """유한한 목표 조향각을 저장하고 비정상 값이면 기존 각도를 유지한다."""
        if not math.isfinite(msg.data):
            self.get_logger().error(
                f"Ignoring non-finite target_angle: {msg.data}"
            )
            return
        self.target_angle = msg.data

    def _speed_callback(self, msg):
        """브레이크가 해제된 상태에서 유한한 목표 속도를 저장한다."""
        if not math.isfinite(msg.data):
            self.get_logger().error(
                f"Non-finite target_speed={msg.data}; forcing speed to zero"
            )
            self.target_speed = 0.0
            self.speed_command_ready = False
            self.last_speed_time = None
            return

        if self.brake:
            return

        self.target_speed = msg.data
        self.speed_command_ready = True
        self.last_speed_time = self.get_clock().now()

    def _brake_callback(self, msg):
        """브레이크 상태를 적용하고 체결 시 이전 속도 명령을 초기화한다."""
        previous_brake = self.brake
        self.brake = msg.data

        if self.brake:
            self.target_speed = 0.0
            self.speed_command_ready = False
            self.last_speed_time = None
            self._send_speed()
            self._send_brake()
        elif self.brake != previous_brake:
            self._send_brake()

    def _steering_center_callback(self, msg):
        """요청을 받으면 0x120을 멈추고 3초간 영점 프레임을 송신한다."""
        if not msg.data:
            return

        now_ns = time.monotonic_ns()
        if (
            self.steering_center_end_ns is not None
            and now_ns < self.steering_center_end_ns
        ):
            self.get_logger().warn(
                "Steering center request ignored: calibration already active"
            )
            return

        duration_sec = max(0.0, float(self.steering_center_duration_sec))
        self.steering_center_end_ns = now_ns + int(duration_sec * 1_000_000_000)
        self.get_logger().info(
            "Steering center calibration started: "
            f"suppressing 0x{self.angle_can_id:03X} and sending "
            f"0x{self.steering_center_can_id:03X} for {duration_sec:.1f} s"
        )
        self._send_steering_center()

    def _timer_callback(self):
        """장치 상태와 속도 watchdog을 확인한 뒤 CAN 명령을 송신한다."""
        self._check_can_device()
        if self.bus is None:
            self._try_reopen_bus()
            if self.bus is None:
                return

        self._apply_speed_watchdog()
        self._send_brake()
        self._send_speed()

        if self._steering_center_is_active():
            self._send_steering_center()
        else:
            self._send_angle()

    def _check_can_device(self):
        """can0가 교체되었는지 주기적으로 확인하고 죽은 socket을 버린다."""
        if self.bus is None:
            return

        now_ns = time.monotonic_ns()
        if (
            self._last_device_check_ns is not None
            and now_ns - self._last_device_check_ns < DEVICE_CHECK_INTERVAL_NS
        ):
            return
        self._last_device_check_ns = now_ns

        current_ifindex = read_ifindex(self.can_interface)
        if current_ifindex == self.bus_ifindex:
            return

        self.get_logger().error(
            f"{self.can_interface} was replaced "
            f"(ifindex {self.bus_ifindex} -> {current_ifindex}); "
            "the CAN socket can no longer reach the bus and is being reopened"
        )
        self._invalidate_bus()

    def _invalidate_bus(self):
        """더 이상 쓸 수 없는 socket을 닫고 재연결 대상으로 표시한다."""
        self._shutdown_bus_safely(log_error=False)
        self.bus_ifindex = None
        # 끊긴 동안의 속도 명령은 신뢰할 수 없으므로 새 명령이 올 때까지 0을
        # 유지한다. brake 와 조향각은 마지막 값을 그대로 둔다.
        self._reset_speed_command()

    def _reset_speed_command(self):
        """새 속도 명령을 받을 때까지 0을 유지하도록 상태를 되돌린다."""
        self.target_speed = 0.0
        self.speed_command_ready = False
        self.last_speed_time = None

    def _try_reopen_bus(self):
        """끊어진 SocketCAN 연결을 정해진 간격으로 다시 시도한다."""
        now_ns = time.monotonic_ns()
        if (
            self._last_reconnect_ns is not None
            and now_ns - self._last_reconnect_ns < RECONNECT_INTERVAL_NS
        ):
            return
        self._last_reconnect_ns = now_ns

        try:
            self._open_bus()
        except Exception as error:
            self._log_reconnect_error(error)
            return

        self._send_error_state.clear()
        self._reconnect_error_active = False
        self._last_reconnect_error_log_ns = None
        self.get_logger().info(
            f"CAN socket reopened on {self.can_interface} "
            f"(ifindex {self.bus_ifindex}); speed stays 0 until the next "
            "target_speed message"
        )

    def _log_reconnect_error(self, error):
        """재연결 실패를 최초 즉시, 이후 5초 간격으로 기록한다."""
        now_ns = time.monotonic_ns()
        if (
            not self._reconnect_error_active
            or self._last_reconnect_error_log_ns is None
            or now_ns - self._last_reconnect_error_log_ns >= ERROR_LOG_INTERVAL_NS
        ):
            self.get_logger().error(
                f"CAN reopen on {self.can_interface} failed: {error}"
            )
            self._last_reconnect_error_log_ns = now_ns
        self._reconnect_error_active = True

    def _steering_center_is_active(self):
        """영점 송신 시간이 남았는지 확인하고 종료 시 한 번 기록한다."""
        if self.steering_center_end_ns is None:
            return False

        if time.monotonic_ns() < self.steering_center_end_ns:
            return True

        self.steering_center_end_ns = None
        self.get_logger().info(
            "Steering center calibration finished; resuming angle commands"
        )
        return False

    def _apply_speed_watchdog(self):
        """새 속도 명령이 timeout되면 목표 속도를 0으로 만든다."""
        if not self.speed_command_ready or self.last_speed_time is None:
            self.target_speed = 0.0
            return

        elapsed = self.get_clock().now() - self.last_speed_time
        if elapsed.nanoseconds / 1_000_000_000 > self.command_timeout_sec:
            self.target_speed = 0.0
            self.speed_command_ready = False
            self.last_speed_time = None
            self.get_logger().warn("Speed command timed out; forcing speed to zero")

    def _send_brake(self):
        """현재 브레이크 상태를 인코딩해 CAN으로 송신한다."""
        data = bytes([1 if self.brake else 0]) + bytes(7)
        self._send_frame(self.brake_can_id, data, "brake")

    def _send_speed(self):
        """현재 목표 속도를 인코딩해 CAN으로 송신한다."""
        value = self._scaled_int16(self.target_speed, self.speed_scale, "target_speed")
        data = struct.pack("<h", value) + bytes(6)
        self._send_frame(self.speed_can_id, data, "speed")

    def _send_angle(self):
        """현재 목표 조향각을 인코딩해 CAN으로 송신한다."""
        value = self._scaled_int16(self.target_angle, self.angle_scale, "target_angle")
        data = struct.pack("<h", value) + bytes(6)
        self._send_frame(self.angle_can_id, data, "angle")

    def _send_steering_center(self):
        """현재 엔코더 위치를 영점으로 저장하라는 1바이트 요청을 송신한다."""
        self._send_frame(
            self.steering_center_can_id,
            bytes([1]),
            "steering_center",
        )

    def _scaled_int16(self, value, scale, name):
        """물리값에 scale을 적용하고 signed int16 범위로 제한한다."""
        scaled = int(round(value * scale))
        clamped = max(INT16_MIN, min(INT16_MAX, scaled))

        if clamped != scaled:
            self.get_logger().warn(
                f"{name}={value} scaled to {scaled}, clamped to {clamped}"
            )

        return clamped

    def _send_frame(self, can_id, data, frame_name):
        """표준 CAN 프레임 하나를 송신하고 오류 상태를 갱신한다."""
        if self.bus is None:
            return

        frame = can.Message(
            arbitration_id=can_id,
            data=data,
            is_extended_id=False,
        )

        try:
            self.bus.send(frame)
        except can.CanError as error:
            self._log_send_error(frame_name, error)
            if error_errno(error) in DEAD_SOCKET_ERRNOS:
                self.get_logger().error(
                    f"CAN socket on {self.can_interface} is gone; "
                    "closing it and reconnecting"
                )
                self._invalidate_bus()
            return

        self._log_send_recovery(frame_name)

    def _log_send_error(self, frame_name, error):
        """송신 오류를 최초 즉시, 이후 5초 간격으로 기록한다."""
        now_ns = self.get_clock().now().nanoseconds
        active, last_log_ns = self._send_error_state.get(
            frame_name,
            (False, None),
        )
        if (
            not active
            or last_log_ns is None
            or now_ns - last_log_ns >= ERROR_LOG_INTERVAL_NS
        ):
            self.get_logger().error(
                f"CAN {frame_name} frame send failed: {error}"
            )
            last_log_ns = now_ns
        self._send_error_state[frame_name] = (True, last_log_ns)

    def _log_send_recovery(self, frame_name):
        """오류가 발생했던 프레임의 송신이 복구되면 한 번 기록한다."""
        active, _ = self._send_error_state.get(frame_name, (False, None))
        if active:
            self.get_logger().info(f"CAN {frame_name} frame send recovered")
            self._send_error_state[frame_name] = (False, None)

    def _shutdown_bus_safely(self, log_error=True):
        """정리 예외를 전파하지 않고 SocketCAN을 한 번만 종료한다."""
        bus = self.bus
        self.bus = None
        if bus is None:
            return

        try:
            bus.shutdown()
        except Exception as error:
            if log_error:
                self.get_logger().error(f"CAN shutdown failed: {error}")

    def destroy_node(self):
        """송신 timer를 취소하고 CAN과 ROS 리소스를 정리한다."""
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None
        try:
            self._shutdown_bus_safely()
        finally:
            super().destroy_node()


def main(args=None):
    """ROS를 초기화하고 ROS-to-CAN node를 실행한 뒤 항상 정리한다."""
    rclpy.init(args=args)
    node = None

    try:
        node = RosToCanNode()
        node.start()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if node is not None:
                node.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()
