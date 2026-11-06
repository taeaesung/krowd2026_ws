import errno
import math

import can
import pytest
import rclpy
from std_msgs.msg import Bool, Float32

from ros_can_bridge import can_to_ros_node as can_to_ros_module
from ros_can_bridge import ros_to_can_node as ros_to_can_module
from ros_can_bridge.can_to_ros_node import CanToRosNode
from ros_can_bridge.ros_to_can_node import RosToCanNode


class FakeBus:
    def __init__(self):
        """테스트에서 송신 프레임과 종료 여부를 저장한다."""
        self.sent = []
        self.shutdown_called = False

    def recv(self, timeout=0.0):
        """수신 프레임이 없는 가상 CAN 상태를 반환한다."""
        return None

    def send(self, frame):
        """실제 송신 대신 CAN 프레임을 테스트 목록에 저장한다."""
        self.sent.append(frame)

    def shutdown(self):
        """가상 CAN 종료 호출 여부를 기록한다."""
        self.shutdown_called = True


class DeadBus(FakeBus):
    """장치가 사라진 SocketCAN socket을 흉내내는 가상 CAN이다."""

    def __init__(self, error_number):
        """송수신에서 발생시킬 errno를 저장한다."""
        super().__init__()
        self.error_number = error_number

    def _raise(self):
        """python-can 3.3.2처럼 OSError를 __context__에 묻은 예외를 낸다."""
        try:
            raise OSError(self.error_number, "device is gone")
        except OSError:
            raise can.CanError("Failed: device is gone")

    def recv(self, timeout=0.0):
        """죽은 socket에서의 수신 실패를 재현한다."""
        self._raise()

    def send(self, frame):
        """죽은 socket에서의 송신 실패를 재현한다."""
        self._raise()


def bus_factory(buses, opened):
    """열린 순서대로 가상 CAN을 돌려주는 can.interface.Bus 대체품을 만든다."""
    def factory(**kwargs):
        bus = buses[min(len(opened), len(buses) - 1)]
        opened.append(bus)
        return bus

    return factory


class FakePublisher:
    def __init__(self):
        """테스트에서 발행된 ROS 메시지를 저장할 목록을 만든다."""
        self.messages = []

    def publish(self, message):
        """실제 발행 대신 ROS 메시지를 테스트 목록에 저장한다."""
        self.messages.append(message)


@pytest.fixture(autouse=True)
def ros_context():
    """각 테스트 전후로 독립적인 ROS context를 생성하고 정리한다."""
    rclpy.init()
    try:
        yield
    finally:
        if rclpy.ok():
            rclpy.shutdown()


def test_non_finite_speed_forces_zero(monkeypatch):
    """비정상 속도값을 받으면 목표 속도가 0으로 바뀌는지 확인한다."""
    bus = FakeBus()
    monkeypatch.setattr(can.interface, "Bus", lambda **kwargs: bus)
    node = RosToCanNode()
    node.start()
    try:
        node._speed_callback(Float32(data=2.0))
        assert node.target_speed == 2.0

        node._speed_callback(Float32(data=math.nan))
        assert node.target_speed == 0.0
        assert not node.speed_command_ready
    finally:
        node.destroy_node()
    assert bus.shutdown_called


def test_non_finite_angle_keeps_last_value(monkeypatch):
    """비정상 조향값을 받으면 마지막 정상 각도를 유지하는지 확인한다."""
    monkeypatch.setattr(can.interface, "Bus", lambda **kwargs: FakeBus())
    node = RosToCanNode()
    node.start()
    try:
        node._angle_callback(Float32(data=0.5))
        node._angle_callback(Float32(data=math.inf))
        assert node.target_angle == 0.5
    finally:
        node.destroy_node()


def test_brake_clears_speed_until_new_command(monkeypatch):
    """브레이크 체결과 해제 뒤 새 속도 명령이 필요한지 확인한다."""
    bus = FakeBus()
    monkeypatch.setattr(can.interface, "Bus", lambda **kwargs: bus)
    node = RosToCanNode()
    node.start()
    try:
        node._speed_callback(Float32(data=3.0))
        node._brake_callback(Bool(data=True))
        assert node.target_speed == 0.0
        assert not node.speed_command_ready

        node._brake_callback(Bool(data=False))
        assert node.target_speed == 0.0
        assert not node.speed_command_ready

        node._speed_callback(Float32(data=1.0))
        assert node.target_speed == 1.0
        assert node.speed_command_ready
    finally:
        node.destroy_node()


def test_steering_center_suppresses_angle_for_three_seconds(monkeypatch):
    """영점 모드에서는 0x120 대신 0x130만 송신하는지 확인한다."""
    bus = FakeBus()
    monkeypatch.setattr(can.interface, "Bus", lambda **kwargs: bus)
    node = RosToCanNode()
    node.start()
    try:
        node._steering_center_callback(Bool(data=True))
        bus.sent.clear()
        node._timer_callback()

        sent_ids = [frame.arbitration_id for frame in bus.sent]
        assert 0x130 in sent_ids
        assert 0x120 not in sent_ids

        node.steering_center_end_ns = 0
        bus.sent.clear()
        node._timer_callback()

        sent_ids = [frame.arbitration_id for frame in bus.sent]
        assert 0x120 in sent_ids
        assert 0x130 not in sent_ids
    finally:
        node.destroy_node()


def test_dead_send_socket_is_closed_and_reopened(monkeypatch):
    """송신이 ENXIO로 실패하면 socket을 닫고 다시 연결하는지 확인한다."""
    buses = [DeadBus(errno.ENXIO), FakeBus()]
    opened = []
    monkeypatch.setattr(can.interface, "Bus", bus_factory(buses, opened))
    node = RosToCanNode()
    node.start()
    try:
        node._speed_callback(Float32(data=5.0))
        assert node.bus is buses[0]

        node._timer_callback()
        assert node.bus is None
        assert buses[0].shutdown_called
        # 재연결 뒤 새 속도 명령이 올 때까지 0 을 유지해야 한다.
        assert node.target_speed == 0.0
        assert not node.speed_command_ready

        node._timer_callback()
        assert node.bus is buses[1]
        assert len(opened) == 2
        assert 0x110 in [frame.arbitration_id for frame in buses[1].sent]
    finally:
        node.destroy_node()


def test_replaced_can_device_forces_reopen(monkeypatch):
    """can0의 ifindex가 바뀌면 죽은 socket을 버리고 다시 여는지 확인한다."""
    buses = [FakeBus(), FakeBus()]
    opened = []
    ifindex = {"value": 7}
    monkeypatch.setattr(can.interface, "Bus", bus_factory(buses, opened))
    monkeypatch.setattr(
        ros_to_can_module,
        "read_ifindex",
        lambda interface: ifindex["value"],
    )
    node = RosToCanNode()
    node.start()
    try:
        assert node.bus_ifindex == 7

        ifindex["value"] = 8
        node._last_device_check_ns = None
        node._timer_callback()

        assert node.bus is buses[1]
        assert node.bus_ifindex == 8
        assert buses[0].shutdown_called
    finally:
        node.destroy_node()


def test_unchanged_can_device_keeps_socket(monkeypatch):
    """장치가 그대로면 socket을 불필요하게 다시 열지 않는지 확인한다."""
    buses = [FakeBus(), FakeBus()]
    opened = []
    monkeypatch.setattr(can.interface, "Bus", bus_factory(buses, opened))
    monkeypatch.setattr(ros_to_can_module, "read_ifindex", lambda interface: 7)
    node = RosToCanNode()
    node.start()
    try:
        node._last_device_check_ns = None
        node._timer_callback()

        assert node.bus is buses[0]
        assert len(opened) == 1
        assert not buses[0].shutdown_called
    finally:
        node.destroy_node()


def test_dead_receive_socket_is_closed_and_reopened(monkeypatch):
    """수신이 ENODEV로 실패하면 socket을 닫고 다시 연결하는지 확인한다."""
    buses = [DeadBus(errno.ENODEV), FakeBus()]
    opened = []
    monkeypatch.setattr(can.interface, "Bus", bus_factory(buses, opened))
    monkeypatch.setattr(can_to_ros_module, "read_ifindex", lambda interface: 7)
    node = CanToRosNode()
    node.start()
    try:
        node._poll_can()
        assert node.bus is None
        assert buses[0].shutdown_called

        node._poll_can()
        assert node.bus is buses[1]
        assert len(opened) == 2
    finally:
        node.destroy_node()


def test_encoder_angle_uses_can_timestamp(monkeypatch):
    """엔코더 각도와 Linux CAN 수신 시각이 함께 발행되는지 확인한다."""
    monkeypatch.setattr(can.interface, "Bus", lambda **kwargs: FakeBus())
    node = CanToRosNode()
    node.start()
    publisher = FakePublisher()
    node.encoder_pub = publisher
    try:
        frame = can.Message(
            arbitration_id=0x201,
            data=[0x9D, 0x00],
            is_extended_id=False,
            timestamp=1_700_000_000.25,
        )
        node._handle_frame(frame)

        assert len(publisher.messages) == 1
        message = publisher.messages[0]
        assert message.angle_rad == pytest.approx(1.57)
        assert message.header.stamp.sec == 1_700_000_000
        assert message.header.stamp.nanosec == pytest.approx(
            250_000_000,
            abs=256,
        )
    finally:
        node.destroy_node()
