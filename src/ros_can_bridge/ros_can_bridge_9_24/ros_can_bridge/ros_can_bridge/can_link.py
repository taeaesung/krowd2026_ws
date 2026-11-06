"""SocketCAN 장치가 교체되었는지 확인하기 위한 공용 helper."""

import errno


# can0 가 slcan(CANable2 + slcand) 이면 daemon 이 재시작될 때마다 netdev 자체가
# 사라졌다가 새로 만들어진다. 커널은 장치를 unregister 할 때 그 장치에 bound 된
# CAN socket 을 끊어버리므로, 그때 살아남은 socket 은 다시는 버스에 닿지 못한다.
# 이 상태에서 송신은 ENXIO 로 실패하지만, 수신은 첫 호출만 ENODEV 를 내고 그
# 뒤로는 조용히 아무 frame 도 주지 않는다. 따라서 오류만으로는 부족하고 장치
# 교체 여부를 ifindex 로 직접 확인해야 한다.
DEAD_SOCKET_ERRNOS = frozenset({errno.ENXIO, errno.ENODEV, errno.EBADF})

# 장치 확인 주기와 재연결 시도 주기. 50 Hz timer 안에서 sysfs 를 매번 읽지
# 않도록 제한하고, 장치가 아직 없을 때 재연결을 과도하게 반복하지 않게 한다.
DEVICE_CHECK_INTERVAL_NS = 500_000_000
RECONNECT_INTERVAL_NS = 1_000_000_000


def read_ifindex(interface):
    """지정한 SocketCAN 장치의 ifindex를 읽고 장치가 없으면 None을 반환한다."""
    try:
        with open(f"/sys/class/net/{interface}/ifindex") as handle:
            return int(handle.read().strip())
    except (OSError, ValueError):
        return None


def error_errno(error):
    """python-can 예외 chain에서 원래 OSError의 errno를 찾아낸다."""
    # python-can 4.x 는 error_code 로 errno 를 넘겨주지만, 3.3.2 는 문자열만
    # 남기고 원래 OSError 를 __context__ 에 묻어둔다.
    code = getattr(error, "error_code", None)
    if isinstance(code, int):
        return code

    seen = set()
    current = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, OSError) and current.errno is not None:
            return current.errno
        current = current.__cause__ or current.__context__
    return None
