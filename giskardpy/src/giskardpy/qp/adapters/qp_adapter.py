from __future__ import annotations

import logging
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import List, TYPE_CHECKING

import numpy as np

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.qp.constraint import (
    DirectLimits,
    DofLimits,
)
from giskardpy.qp.constraint_collection import ConstraintCollection
from krrood.symbolic_math.symbolic_math import Vector, Matrix
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from giskardpy.qp.qp_controller_config import QPControllerConfig


@dataclass
class QPConstraintComponent(ABC):
    """
    A kind of factory method that produces parts of the QP problem.
    It has to compute a matrix and bounds for the matrix.
    The bounds are decided by the subclasses EqualityConstraintComponent and InequalityConstraintComponent.
    """

    degrees_of_freedom: List[DegreeOfFreedom]
    constraint_collection: ConstraintCollection
    config: QPControllerConfig

    matrix: sm.Matrix = field(init=False)
    slack_matrix: sm.Matrix = field(init=False)
    slack_variables: DirectLimits = field(init=False)

    @property
    def number_of_free_variables(self) -> int:
        return len(self.degrees_of_freedom)

    @property
    def number_of_velocity_columns(self) -> int:
        return self.number_of_free_variables * (self.config.prediction_horizon - 2)

    @property
    def number_of_jerk_columns(self) -> int:
        return self.number_of_free_variables * self.config.prediction_horizon

    @property
    def position_variables(self) -> Vector:
        return Vector([dof.variables.position for dof in self.degrees_of_freedom])

    @property
    def velocity_variables(self) -> Vector:
        return Vector([dof.variables.velocity for dof in self.degrees_of_freedom])

    @property
    def acceleration_variables(self) -> Vector:
        return Vector([dof.variables.acceleration for dof in self.degrees_of_freedom])

    @property
    @abstractmethod
    def constraint_names(self) -> list[str]: ...


@dataclass
class EqualityDerivativeLinkModel(QPConstraintComponent):
    r"""
    The constraints produced by this class describe the discrete-time relationships between variables
    in the prediction horizon :math:`N` using a semi-implicit euler integration method:

    .. math::

        v_k = v_{k-1} + a_{k} \, \Delta t

        a_k = a_{k-1} + j_{k} \, \Delta t

    Where v, a and j are velocity, acceleration and jerk, respectively, and k is the time step.
    Acceleration variables are removed using substitution.
    The first two row links the MPC to the current state:

    .. math::

        -v_{current} - a_{current} \, \Delta t = -v_0 + j_0 \, \Delta t^2

        v_{current} = - v_1 + 2 v_0 + j_1 \, \Delta t^2

    Row from 2 until k-2 have this form:

    .. math::

        0 = - v_k + 2 v_{k-1} - v_{k-2} + j_k \, \Delta t^2

    The final two rows have this form:

    .. math::

        0 = 2 v_{k-1} - v_{k-2} + j_k \, \Delta t^2

        0 = - v_{k-2} + j_k \, \Delta t^2

    For a prediciton horizon of 5 with 1 degree of freedom, the matrix looks like this:

    ::

        |  equality_bounds |   |           equality constraint matrix          |   |    v_0    |
        |------------------|   |-----------------------------------------------|   |    v_1    |
        | - v_c - a_c * dt |   | -1  |     |     |  1  |     |     |     |     |   |    v_2    |
        |       v_c        |   |  2  | -1  |     |     |  1  |     |     |     |   | j_0*dt**2 |
        |        0         | = | -1  |  2  | -1  |     |     |  1  |     |     | @ | j_1*dt**2 |
        |        0         |   |     | -1  |  2  |     |     |     |  1  |     |   | j_2*dt**2 |
        |        0         |   |     |     | -1  |     |     |     |     |  1  |   | j_3*dt**2 |
        |------------------|   |-----------------------------------------------|   | j_4*dt**2 |
    """

    bounds: Vector = field(init=False)

    def __post_init__(self):
        self.create_matrix()
        self.compute_bounds()
        self.slack_matrix = Matrix.zeros(self.matrix.shape[0], 0)

    @property
    def constraint_names(self) -> list[str]:
        names = []
        for k in range(self.config.prediction_horizon):
            for dof in self.degrees_of_freedom:
                names.append(f"{dof.name} k_{k} vel/jerk link")
        return names

    def create_matrix(self):
        matrix = np.zeros(
            (
                self.number_of_jerk_columns,
                self.number_of_velocity_columns + self.number_of_jerk_columns,
            )
        )
        identity = np.eye(self.number_of_velocity_columns)
        velocity_at_k = -identity
        velocity_at_k_minus1 = -identity
        velocity_at_k_minus2 = 2 * identity
        matrix[
            : -self.number_of_free_variables * 2, : self.number_of_velocity_columns
        ] += velocity_at_k
        matrix[
            self.number_of_free_variables : -self.number_of_free_variables,
            : self.number_of_velocity_columns,
        ] += velocity_at_k_minus2
        matrix[
            self.number_of_free_variables * 2 :, : self.number_of_velocity_columns
        ] += velocity_at_k_minus1

        matrix[:, self.number_of_velocity_columns :] = np.eye(
            self.number_of_jerk_columns
        )

        self.matrix = sm.Matrix(matrix)

    def compute_bounds(self):
        self.bounds = sm.Vector.zeros(self.number_of_jerk_columns)
        self.bounds[: self.number_of_free_variables] = (
            -self.velocity_variables - self.acceleration_variables * self.config.mpc_dt
        )
        self.bounds[
            self.number_of_free_variables : self.number_of_free_variables * 2
        ] = self.velocity_variables


@dataclass
class QPDataSymbolic:
    """
    Takes free variables and constraints and converts them to a QP problem in the following format, depending on the
    class attributes:
    min_x 0.5 x^T H x + g^T x
    s.t.  lb <= x <= ub     (box constraints)
          Edof x <= bE_dof          (equality constraints)
          Eslack x <= bE_slack        (equality constraints)
          lbA <= Adof x <= ubA_dof  (lower/upper inequality constraints)
          lbA <= Aslack x <= ubA_slack  (lower/upper inequality constraints)
    """

    degrees_of_freedom: List[DegreeOfFreedom]
    constraint_collection: ConstraintCollection
    config: QPControllerConfig

    quadratic_weights: Vector = field(init=False)
    linear_weights: Vector = field(init=False)

    box_lower_constraints: Vector = field(init=False)
    box_upper_constraints: Vector = field(init=False)

    free_variable_names: list[str] = field(init=False)

    eq_matrix_dofs: Matrix = field(init=False)
    eq_matrix_slack: Matrix = field(init=False)
    eq_bounds: Vector = field(init=False)
    eq_constraint_names: list[str] = field(init=False)

    neq_matrix_dofs: Matrix = field(init=False)
    neq_matrix_slack: Matrix = field(init=False)
    neq_lower_bounds: Vector = field(init=False)
    neq_upper_bounds: Vector = field(init=False)
    neq_constraint_names: list[str] = field(init=False)

    def __post_init__(self):
        direct_limits = DofLimits.create(self.degrees_of_freedom, self.config)
        mpc_model = EqualityDerivativeLinkModel(
            degrees_of_freedom=self.degrees_of_freedom,
            constraint_collection=self.constraint_collection,
            config=self.config,
        )
        quadratic_weights = [direct_limits.quadratic_weights]
        linear_weights = [direct_limits.linear_weights]
        box_lower_constraints = [direct_limits.lower_bounds]
        box_upper_constraints = [direct_limits.upper_bounds]
        eq_matrix_dofs = [mpc_model.matrix]
        eq_matrix_slack = [mpc_model.slack_matrix]
        eq_bounds = [mpc_model.bounds]
        self.eq_constraint_names = mpc_model.constraint_names
        self.free_variable_names = direct_limits.names

        for (
            enforcement_strategy,
            constraints,
        ) in self.constraint_collection.get_equality_constraint_blocks().items():
            strategy = enforcement_strategy(self.degrees_of_freedom, self.config)

            slack_variables = strategy.create_slack_variables(constraints)
            quadratic_weights.append(slack_variables.quadratic_weights)
            linear_weights.append(slack_variables.linear_weights)
            box_lower_constraints.append(slack_variables.lower_bounds)
            box_upper_constraints.append(slack_variables.upper_bounds)

            matrix = strategy.create_matrix(constraints)
            slack_matrix = strategy.create_slack_matrix(constraints)
            bounds = strategy.create_bounds(
                [c.bound.bound for c in constraints],
                [c.normalization_factor for c in constraints],
            )
            eq_matrix_dofs.append(matrix)
            eq_matrix_slack.append(slack_matrix)
            eq_bounds.append(bounds)
            self.eq_constraint_names.extend(strategy.create_names(constraints))
            self.free_variable_names.extend(slack_variables.names)

        ineq_matrix_dofs = []
        ineq_matrix_slack = []
        lower_bounds = []
        upper_bounds = []
        self.neq_constraint_names = []

        for (
            enforcement_strategy,
            constraints,
        ) in self.constraint_collection.get_inequality_constraint_blocks().items():
            strategy = enforcement_strategy(self.degrees_of_freedom, self.config)

            slack_variables = strategy.create_slack_variables(constraints)
            quadratic_weights.append(slack_variables.quadratic_weights)
            linear_weights.append(slack_variables.linear_weights)
            box_lower_constraints.append(slack_variables.lower_bounds)
            box_upper_constraints.append(slack_variables.upper_bounds)

            matrix = strategy.create_matrix(constraints)
            slack_matrix = strategy.create_slack_matrix(constraints)
            lower_bound = strategy.create_bounds(
                [c.bound.lower_bound for c in constraints],
                [c.normalization_factor for c in constraints],
            )
            upper_bound = strategy.create_bounds(
                [c.bound.upper_bound for c in constraints],
                [c.normalization_factor for c in constraints],
            )
            ineq_matrix_dofs.append(matrix)
            ineq_matrix_slack.append(slack_matrix)
            lower_bounds.append(lower_bound)
            upper_bounds.append(upper_bound)
            self.neq_constraint_names.extend(strategy.create_names(constraints))
            self.free_variable_names.extend(slack_variables.names)

        self.quadratic_weights = sm.concatenate(*quadratic_weights)
        self.linear_weights = sm.concatenate(*linear_weights)
        self.box_lower_constraints = sm.concatenate(*box_lower_constraints)
        self.box_upper_constraints = sm.concatenate(*box_upper_constraints)
        self.eq_matrix_dofs = sm.vstack(eq_matrix_dofs)
        self.eq_matrix_slack = sm.diag_stack(eq_matrix_slack)
        self.eq_bounds = sm.concatenate(*eq_bounds)

        self.neq_matrix_dofs = sm.vstack(ineq_matrix_dofs)
        self.neq_matrix_slack = sm.diag_stack(ineq_matrix_slack)
        self.neq_lower_bounds = sm.Vector(lower_bounds)
        self.neq_upper_bounds = sm.Vector(upper_bounds)

    def __hash__(self):
        return hash(id(self))

    @property
    def num_free_variable_constraints(self) -> int:
        return len(self.degrees_of_freedom)

    @property
    def num_eq_slack_variables(self) -> int:
        return self.eq_matrix_slack.shape[1]

    @property
    def num_neq_slack_variables(self) -> int:
        return self.neq_matrix_slack.shape[1]

    @property
    def num_slack_variables(self) -> int:
        return self.num_eq_slack_variables + self.num_neq_slack_variables

    @property
    def num_non_slack_variables(self) -> int:
        return self.quadratic_weights.shape[0] - self.num_slack_variables
