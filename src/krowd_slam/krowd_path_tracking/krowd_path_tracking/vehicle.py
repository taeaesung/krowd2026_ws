"""Vehicle geometry: what the car IS, as opposed to how the controller is tuned.

Single source of truth for the whole stack (config/vehicle.yaml). These numbers
used to live inside krowd_embedded_mpc/mpc_algorithm_embedded.py, which meant
every consumer either imported the MPC or copied the value; wheel_odom_bicycle_node
still carries its own 1.340 default for exactly that reason. Putting a different
vehicle on this code should be one YAML edit, not a grep.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Vehicle:
    """Measured geometry plus the steering limit derived from it."""

    wheelbase_m: float
    track_width_m: float
    min_turning_radius_m: float
    max_steering_angle_rad: float
    max_steering_rate_rad_s: float
    # Where the rear axle sits relative to base_link, along base_link's +x.
    # Pure Pursuit's geometry is derived at the REAR AXLE: steering at a point
    # that is not the rear axle biases the whole controller, and the bias grows
    # with the offset. 0.0 declares "base_link is the rear axle" - which is an
    # assumption worth actually measuring on the vehicle, not inheriting.
    rear_axle_offset_m: float

    @property
    def max_curvature(self) -> float:
        return math.tan(self.max_steering_angle_rad) / self.wheelbase_m

    @property
    def implied_turning_radius_m(self) -> float:
        """The turning radius max_steering_angle_rad actually commands."""

        return self.wheelbase_m / math.tan(self.max_steering_angle_rad)


@dataclass(frozen=True)
class SteeringActuator:
    """How a road-wheel angle becomes the /target_angle the steer MCU takes.

    The steer Teensy's angle is its encoder count over a fixed gear ratio - a
    steering-COLUMN-side angle, not the road wheel's. Measured on the 0912_2
    drive (encoder against the road-wheel angle the IMU yaw rate and RTK speed
    imply, kinematic bicycle): about 8.0 command radians per road-wheel radian,
    consistent from 15 to 80 deg of command. The MCU also REJECTS a command
    beyond max_command_rad (ANGLE_LIMIT_RAD, 2.5 since 2026-09-20; it was
    1.7453) rather than clamping it, so the command is clamped here first.
    """

    gain: float
    offset_rad: float
    max_command_rad: float

    def validate(self) -> None:
        if self.gain <= 0.0:
            raise ValueError(f"steering command gain must be positive: {self.gain}")
        if self.max_command_rad <= abs(self.offset_rad):
            raise ValueError(
                f"steering command limit {self.max_command_rad} must exceed the "
                f"offset {self.offset_rad}"
            )

    def command(self, road_wheel_rad: float, sign: float = 1.0) -> float:
        raw = sign * self.gain * road_wheel_rad + self.offset_rad
        return max(-self.max_command_rad, min(self.max_command_rad, raw))

    @property
    def road_wheel_limit_rad(self) -> float:
        """The largest road-wheel angle the command range can still express."""

        return (self.max_command_rad - abs(self.offset_rad)) / self.gain


MPS_TO_KMH = 3.6


def speed_command_kmh(speed_mps: float) -> float:
    """The /target_speed value for a speed in m/s: km/h, never negative.

    ros_can_bridge multiplies /target_speed by 100 into CAN 0x110 and the drive
    MCU reads that as km/h x 100 (mcu_code/*: "실제 속도(km/h) x 100"). The
    controllers here work in m/s (the Trajectory, nav_msgs, the tests), so the
    conversion happens once, at the topic. Until 2026-09-19 the m/s value was
    published as is and the vehicle ran at 1/3.6 of the route speed. The MPC
    stack's can_adapter sends km/h the same way (speed_scale_mps_to_kmh 3.6).
    """

    return max(float(speed_mps), 0.0) * MPS_TO_KMH


def steering_limit_from_turning_radius(
    wheelbase_m: float, min_turning_radius_m: float
) -> float:
    """The road-wheel angle that produces `min_turning_radius_m`.

    The bicycle-model inverse of R = L / tan(delta). Note this is the angle of
    the (virtual, centred) FRONT ROAD WHEEL, not a steering-wheel angle: on a
    car with a steering ratio the two differ by that ratio and are not
    interchangeable anywhere in this package.
    """

    if wheelbase_m <= 0.0:
        raise ValueError(f"wheelbase_m must be positive: {wheelbase_m}")
    if min_turning_radius_m <= 0.0:
        raise ValueError(
            f"min_turning_radius_m must be positive: {min_turning_radius_m}"
        )
    return math.atan(wheelbase_m / min_turning_radius_m)


def implausible_steering_limit_reason(vehicle: Vehicle) -> str | None:
    """Why this steering limit looks like a unit mistake; None when it is sane.

    The one mistake this is here to catch: commanding a STEERING-WHEEL angle
    through an interface that means a ROAD-WHEEL angle. /target_angle is
    radians of road-wheel angle (ros_can_bridge encodes it as rad*100 on CAN
    0x120) and a steering-wheel figure put there is larger by the steering
    ratio, which reads as a physically impossible limit rather than as a wrong
    unit. A warning, not an error: an unmeasured MIN_TURNING_RADIUS is just as
    likely to be the thing that is wrong.
    """

    if vehicle.max_steering_angle_rad >= 0.5 * math.pi:
        return "steering limit is at or beyond 90 deg, which commands no radius at all"
    radius = vehicle.implied_turning_radius_m
    if radius < 0.5 * vehicle.track_width_m:
        return (
            f"steering limit {math.degrees(vehicle.max_steering_angle_rad):.1f} deg "
            f"implies a {radius:.2f} m turning radius, inside the vehicle's own "
            f"half track width ({0.5 * vehicle.track_width_m:.2f} m). This is the "
            "signature of a steering-WHEEL angle used where a road-WHEEL angle is "
            "meant; /target_angle is a road-wheel angle in radians"
        )
    return None
