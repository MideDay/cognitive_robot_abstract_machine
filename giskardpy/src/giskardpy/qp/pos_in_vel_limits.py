from copy import copy

import numpy as np
from sympy.series import acceleration
from typing_extensions import Tuple, List

import krrood.symbolic_math.symbolic_math as sm
from krrood.symbolic_math.symbolic_math import (
    Scalar,
    Vector,
    substitution_cache,
)


def shifted_velocity_profile(
    velocity_profile: Vector,
    acceleration_profile: Vector,
    distance: Scalar,
    delta_time: float,
) -> Tuple[Vector, Vector]:
    velocity_profile = copy(velocity_profile)
    velocity_profile[velocity_profile < 0] = 0
    velocity_if_cases = []
    acceleration_if_cases = []
    for x in range(len(velocity_profile) - 1, -1, -1):
        condition = delta_time * sum(velocity_profile[x:])
        velocity_result = np.concatenate([velocity_profile[x + 1 :], np.zeros(x + 1)])
        acceleration_result = np.concatenate(
            [acceleration_profile[x + 1 :], np.zeros(x + 1)]
        )
        if condition > 0:
            velocity_if_cases.append((condition, sm.Vector(velocity_result)))
            acceleration_if_cases.append((condition, sm.Vector(acceleration_result)))
    velocity_if_cases.append(
        (
            2 * velocity_if_cases[-1][0] - velocity_if_cases[-2][0],
            sm.Vector(velocity_profile),
        )
    )
    default_velocity_profile = np.full(velocity_profile.shape[0], velocity_profile[0])

    shifted_velocity_profile = sm.if_less_eq_cases(
        distance, velocity_if_cases, sm.Vector(default_velocity_profile)
    )
    shifted_acceleration_profile = sm.if_less_eq_cases(
        distance, acceleration_if_cases, sm.Vector(acceleration_profile)
    )
    return shifted_velocity_profile, shifted_acceleration_profile


def reverse_gauss(integral: Scalar) -> Scalar:
    return sm.sqrt(2 * integral + (1 / 4)) - 1 / 2


@substitution_cache
def acceleration_cap(
    current_velocity: Scalar, jerk_limit: Scalar, delta_time: Scalar
) -> Scalar:
    acceleration_integral = sm.abs(current_velocity) / delta_time
    jerk_step = jerk_limit * delta_time
    n = sm.floor(reverse_gauss(sm.abs(acceleration_integral / jerk_step)))
    x = (-sm.gauss(n) * jerk_limit * delta_time + acceleration_integral) / (n + 1)
    return sm.abs(n * jerk_limit * delta_time + x)


@substitution_cache
def compute_next_vel_and_acc(
    current_velocity: Scalar,
    current_acceleration: Scalar,
    velocity_limit: Scalar,
    jerk_limit: Scalar,
    delta_time: Scalar,
    remaining_prediction_horizon: Scalar,
    no_cap: Scalar,
) -> Tuple[Scalar, Scalar]:
    acceleration_cap1 = acceleration_cap(
        current_velocity, jerk_limit, delta_time
    )  # if we start at arbitrary horizon and jerk as strongly as possible, which acc do we have when we reach the vel limit
    acceleration_cap2 = (
        remaining_prediction_horizon * jerk_limit * delta_time
    )  # max acc reachable given horizon depending only on vel
    acceleration_prediction_horizon_max = sm.min(
        acceleration_cap1, acceleration_cap2
    )  # in reality we have a limited horizon, so we have to use the min of the two.
    acceleration_prediction_horizon_min = -acceleration_prediction_horizon_max

    next_acceleration_min = (
        current_acceleration - jerk_limit * delta_time
    )  # looking from the other side, these are the actual acc we can achieve with the jerk limits
    next_acceleration_max = current_acceleration + jerk_limit * delta_time

    acceleration_to_velocity = (
        velocity_limit - current_velocity
    ) / delta_time  # the total acc needed to reach vel target vel

    target_acceleratino = sm.max(next_acceleration_min, acceleration_to_velocity)
    target_acceleratino = sm.if_else(
        no_cap,
        target_acceleratino,
        sm.limit(
            target_acceleratino,
            acceleration_prediction_horizon_min,
            acceleration_prediction_horizon_max,
        ),
    )  # skip when vel_limit is negative
    next_acceleration = sm.limit(
        target_acceleratino, next_acceleration_min, next_acceleration_max
    )

    next_velocity = current_velocity + next_acceleration * delta_time
    return next_velocity, next_acceleration


@substitution_cache
def compute_slowdown_asap_vel_profile(
    current_velocity: Scalar,
    current_acceleration: Scalar,
    target_velocity_profile: Vector,
    jerk_limit: Scalar,
    delta_time: Scalar,
    prediction_horizon: int,
    skip_first: Scalar,
) -> Tuple[Vector, Vector, Vector]:
    """
    Compute the vel, acc and jerk profile for slowing down asap.
    """
    velocity_profile = []
    acceleration_profile = []
    next_velocity, next_acceleration = current_velocity, current_acceleration
    for i in range(prediction_horizon):
        next_velocity, next_acceleration = compute_next_vel_and_acc(
            next_velocity,
            next_acceleration,
            target_velocity_profile[i],
            jerk_limit,
            delta_time,
            prediction_horizon - i - 1,
            sm.logic_and(skip_first, sm.Scalar(i == 0)),
        )
        velocity_profile.append(next_velocity)
        acceleration_profile.append(next_acceleration)
    acceleration_profile = copy(Vector(acceleration_profile))
    acceleration_profile2 = copy(Vector(acceleration_profile))
    acceleration_profile2[1:] = acceleration_profile[:-1]
    acceleration_profile2[0] = current_acceleration
    jerk_profile = (acceleration_profile - acceleration_profile2) / delta_time

    return Vector(velocity_profile), acceleration_profile, jerk_profile


def implicit_vel_profile(
    acceleration_limit: float,
    jerk_limit: float,
    delta_time: float,
    prediction_horizon: int,
) -> List[float]:
    velocity_profile = [0, 0]  # because last two vel are always 0
    velocity = 0
    acceleration = 0
    for i in range(prediction_horizon - 2):
        acceleration += jerk_limit * delta_time
        acceleration = min(acceleration, acceleration_limit)
        velocity += acceleration * delta_time
        velocity_profile.append(velocity)
    return list(reversed(velocity_profile))
