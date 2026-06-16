from dataclasses import dataclass, field

from giskardpy.motion_statechart.monitors.payload_monitors import (
    ThreadedPredicateMonitor,
)
from krrood.entity_query_language.factories import ConditionType, evaluate_condition
from pycram.exceptions import ConditionNotSatisfied
from pycram.plans.plan_node import PlanNode


@dataclass
class ConditionNode(PlanNode):
    """
    Node representing a pre or post condition of an action
    """

    condition: ConditionType = field(kw_only=True)
    """
    The EQL condition to be evaluated
    """

    pre_condition: bool = field(kw_only=True)
    """
    If this is a pre or post condition
    """

    def notify(self):
        pass


def condition_monitor(condition_node: ConditionNode) -> ThreadedPredicateMonitor:
    """
    Build a giskard monitor that evaluates a PyCRAM condition inside a motion
    state chart.

    The EQL condition is wrapped in a plain callable, so giskard never sees any
    PyCRAM/EQL types. The condition is evaluated in a background thread (see
    :class:`~giskardpy.motion_statechart.monitors.payload_monitors.ThreadedPredicateMonitor`),
    its observation state becoming TRUE/FALSE once evaluation finishes.

    :param condition_node: The pre- or post-condition node to evaluate.
    :return: A monitor whose observation reflects the condition's truth value.
    """
    name = "pre_condition" if condition_node.pre_condition else "post_condition"
    return ThreadedPredicateMonitor(
        predicate=lambda: bool(evaluate_condition(condition_node.condition)),
        name=name,
    )
