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
from ros_can_bridge_msgs.msg import EncoderAngleStamped, WheelSpeedStamped


ERROR_LOG_INTERVAL_NS = 5_000_000_000


class CanToRosNode(Node):
    def __init__(self):
        """파라미터를 선언하고 CAN 연결 전에 ROS publisher를 생성한다."""
        super().__init__("can_to_ros_node")
        self.bus = None
        self.bus_ifindex = None
        self.timer = None
        self._receive_error_active = False
        self._last_receive_error_log_ns = None
        self._last_device_check_ns = None
        self._last_reconnect_ns = None
        self._reconnect_error_active = False
        self._last_reconnect_error_log_ns = None

        self.declare_parameter("can_interface", "can0")
        self.declare_parameter("poll_rate_hz", 50.0)
        self.declare_parameter("wheel_sensor_topic", "/wheel_sensor")
        self.declare_parameter("encoder_topic", "/encoder")
        self.declare_parameter("wheel_sensor_can_id", 0x200)
        self.declare_parameter("encoder_can_id", 0x201)
        # Drive MCU 0x200 payload is true km/h * 100, while WheelSpeedStamped
        # is m/s. raw / 360 performs both the fixed-point and km/h-to-m/s
        # conversion: (km/h * 100) / (100 * 3.6).
        self.declare_parameter("wheel_sensor_scale", 360.0)
        self.declare_parameter("encoder_scale", 100.0)
        self.declare_parameter("extended_id", False)
        self.declare_parameter("receive_own_messages", False)
        self.declare_parameter("max_frames_per_poll", 100)

        self.can_interface = self.get_parameter("can_interface").value
        self.wheel_sensor_can_id = self.get_parameter("wheel_sensor_can_id").value
        self.encoder_can_id = self.get_parameter("encoder_can_id").value
        self.wheel_sensor_scale = self._positive_scale("wheel_sensor_scale")
        self.encoder_scale = self._positive_scale("encoder_scale")
        self.extended_id = self.get_parameter("extended_id").value
        self.max_frames_per_poll = self.get_parameter("max_frames_per_poll").value
        if self.max_frames_per_poll <= 0:
            raise ValueError("max_frames_per_poll must be greater than 0")

        poll_rate_hz = self.get_parameter("poll_rate_hz").value
        if poll_rate_hz <= 0.0:
            raise ValueError("poll_rate_hz must be greater than 0")

        qos = QoSProfile(depth=1)
        self.wheel_sensor_pub = self.create_publisher(
            WheelSpeedStamped,
            self.get_parameter("wheel_sensor_topic").value,
            qos,
        )
        self.encoder_pub = self.create_publisher(
            EncoderAngleStamped,
            self.get_parameter("encoder_topic").value,
            qos,
        )

        self.poll_rate_hz = poll_rate_hz

    def start(self):
        """SocketCAN을 열고 주기적인 수신 timer를 시작한다."""
        self._open_bus()
        try:
            self.timer = self.create_timer(
                1.0 / self.poll_rate_hz,
                self._poll_can,
            )
        except Exception:
            self._shutdown_bus_safely(log_error=False)
            raise
        self.get_logger().info(
            "CAN to ROS bridge started: "
            f"interface={self.can_interface}, poll_rate={self.poll_rate_hz} Hz, "
            f"wheel_sensor_id=0x{self.wheel_sensor_can_id:X}, "
            f"encoder_id=0x{self.encoder_can_id:X}"
        )

    def _open_bus(self):
        """SocketCAN을 열고 연결된 장치의 ifindex를 함께 기록한다."""
        bus = can.interface.Bus(
            channel=self.can_interface,
            interface="socketcan",
            receive_own_messages=self.get_parameter(
                "receive_own_messages"
            ).value,
        )
        self.bus = bus
        self.bus_ifindex = read_ifindex(self.can_interface)
        # 장치 확인과 재연결 주기는 sim time 에 영향받지 않아야 하므로
        # ROS clock 이 아니라 monotonic clock 을 쓴다.
        self._last_device_check_ns = time.monotonic_ns()

    def _positive_scale(self, parameter_name):
        """스케일 파라미터를 읽고 0 이하의 값을 거부한다."""
        value = self.get_parameter(parameter_name).value
        if value <= 0.0:
            raise ValueError(f"{parameter_name} must be greater than 0")
        return value

    def _poll_can(self):
        """장치 상태를 확인한 뒤 수신 큐의 CAN 프레임을 처리한다."""
        self._check_can_device()
        if self.bus is None:
            self._try_reopen_bus()
            if self.bus is None:
                return

        for _ in range(self.max_frames_per_poll):
            try:
                frame = self.bus.recv(timeout=0.0)
            except can.CanError as error:
                self._log_receive_error(error)
                if error_errno(error) in DEAD_SOCKET_ERRNOS:
                    self.get_logger().error(
                        f"CAN socket on {self.can_interface} is gone; "
                        "closing it and reconnecting"
                    )
                    self._invalidate_bus()
                return

            self._log_receive_recovery()
            if frame is None:
                return

            self._handle_frame(frame)

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

        self._receive_error_active = False
        self._last_receive_error_log_ns = None
        self._reconnect_error_active = False
        self._last_reconnect_error_log_ns = None
        self.get_logger().info(
            f"CAN socket reopened on {self.can_interface} "
            f"(ifindex {self.bus_ifindex})"
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

    def _log_receive_error(self, error):
        """수신 오류를 최초 즉시, 이후 5초 간격으로 기록한다."""
        now_ns = self.get_clock().now().nanoseconds
        should_log = (
            not self._receive_error_active
            or self._last_receive_error_log_ns is None
            or now_ns - self._last_receive_error_log_ns >= ERROR_LOG_INTERVAL_NS
        )
        self._receive_error_active = True

        if should_log:
            self.get_logger().error(f"CAN receive failed: {error}")
            self._last_receive_error_log_ns = now_ns

    def _log_receive_recovery(self):
        """CAN 수신 오류가 복구되면 한 번만 기록한다."""
        if self._receive_error_active:
            self.get_logger().info("CAN receive recovered")
            self._receive_error_active = False
            self._last_receive_error_log_ns = None

    def _handle_frame(self, frame):
        """휠 속도와 엔코더 프레임을 해당 ROS publisher로 전달한다."""
        if frame.is_extended_id != self.extended_id:
            return

        if frame.arbitration_id == self.wheel_sensor_can_id:
            value = self._decode_scaled_float(frame, self.wheel_sensor_scale)
            if value is None:
                return
            message = WheelSpeedStamped()
            message.header.stamp = self._can_timestamp_to_ros(frame.timestamp)
            message.header.frame_id = "wheel_sensor"
            message.speed_mps = value
            self.wheel_sensor_pub.publish(message)
        elif frame.arbitration_id == self.encoder_can_id:
            value = self._decode_scaled_float(frame, self.encoder_scale)
            if value is None:
                return
            message = EncoderAngleStamped()
            message.header.stamp = self._can_timestamp_to_ros(frame.timestamp)
            message.header.frame_id = "encoder"
            message.angle_rad = value
            self.encoder_pub.publish(message)

    def _decode_scaled_float(self, frame, scale):
        """부호 있는 int16 little-endian 값을 읽고 스케일을 적용한다."""
        if len(frame.data) < 2:
            self.get_logger().warn(
                f"Ignoring CAN ID 0x{frame.arbitration_id:X}: "
                "expected at least 2 data bytes"
            )
            return None

        raw_value = struct.unpack("<h", bytes(frame.data[:2]))[0]
        return float(raw_value) / scale

    @staticmethod
    def _can_timestamp_to_ros(timestamp):
        """python-can의 Unix timestamp를 ROS Time 메시지로 변환한다."""
        from builtin_interfaces.msg import Time

        seconds = int(timestamp)
        nanoseconds = int((timestamp - seconds) * 1_000_000_000)
        return Time(sec=seconds, nanosec=nanoseconds)

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
        """polling을 취소하고 CAN과 ROS node 리소스를 정리한다."""
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None
        try:
            self._shutdown_bus_safely()
        finally:
            super().destroy_node()


def main(args=None):
    """ROS를 초기화하고 CAN-to-ROS node를 실행한 뒤 항상 정리한다."""
    rclpy.init(args=args)
    node = None

    try:
        node = CanToRosNode()
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
