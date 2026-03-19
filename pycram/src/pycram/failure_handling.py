from __future__ import annotations

import logging
from dataclasses import dataclass, field

from typing_extensions import (
    Any,
    List,
    Type,
    Callable,
    TYPE_CHECKING,
    Dict,
)

from pycram.datastructures.enums import TaskStatus
from pycram.failures import PlanFailure
from pycram.plans.plan_entity import PlanEntity


@dataclass
class FailureHandling(PlanEntity):
    """
    Base class for failure handling mechanisms in automated systems or workflows.

    This class provides a structure for implementing different strategies to handle
    failures that may occur during the execution of a plan or process. It is designed
    to be extended by subclasses that implement specific failure handling behaviors.
    """

    def perform(self):
        """
        Abstract method to perform the failure handling mechanism.

        This method should be overridden in subclasses to implement the specific
        behavior for handling failures.

        :raises NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError()


@dataclass
class Retry(FailureHandling):
    """
    Retries the plan until it works or `max_tries` is reached.
    """

    max_tries: int
    """
    The maximum number of attempts to retry the action.
    """

    def perform(self):
        tries = 0
        for action in iter(self.plan.current_node):
            tries += 1
            try:
                action.perform()
                break
            except PlanFailure as e:
                if tries >= self.max_tries:
                    raise e


class RetryMonitor(Retry):
    """
    Retry a plan by extracting recovery behaviors from the `recovery` dictionary.
    """

    recovery: Dict[Exception, Callable] = field(default_factory=dict)
    """
    A dictionary that maps exception types to recovery methods
    """

    def perform(self) -> List[Any]:
        """
        This method attempts to perform the Monitor + plan specified in the designator_description. If the action
        fails, it is retried up to max_tries times. If all attempts fail, the last exception is raised. In every
        loop, we need to clear the kill_event, and set all relevant 'interrupted' variables to False, to make sure
        the Monitor and plan are executed properly again.

        :raises PlanFailure: If all retry attempts fail.

        :return: The state of the execution performed, as well as a flattened list of the
        results, in the correct order
        """

        def reset_interrupted(child):
            child.status = TaskStatus.CREATED
            try:
                for sub_child in child.children:
                    reset_interrupted(sub_child)
            except AttributeError:
                pass

        def flatten(result):
            flattened_list = []
            if result:
                for item in result:
                    if isinstance(item, list):
                        flattened_list.extend(item)
                    else:
                        flattened_list.append(item)
                return flattened_list
            return None

        status, res = None, None
        with self.lock:
            tries = 0
            while True:
                self.plan.interrupted = False
                for child in self.plan.root.children:
                    reset_interrupted(child)
                try:
                    if tries >= 1:
                        self.plan.re_perform()
                        break
                    else:
                        res = self.plan.perform()
                        break
                except PlanFailure as e:
                    tries += 1
                    if tries >= self.max_tries:
                        raise e
                    exception_type = type(e)
                    if exception_type in self.recovery:
                        self.recovery[exception_type]()
        return flatten(res)


def try_action(action: Any, failure_type: Type[Exception], max_tries: int = 3):
    """
    A generic function to retry an action a certain number of times before giving up, with a specific failure type.

    :param action: The action to be performed, it must have a perform() method.
    :param failure_type: The type of exception to catch.
    :param max_tries: The maximum number of attempts to retry the action. Defaults to 3.
    """
    return try_method(
        method=lambda: action.perform(),
        failure_type=failure_type,
        max_tries=max_tries,
        name="action",
    )


def try_method(
    method: Callable,
    failure_type: Type[Exception],
    max_tries: int = 3,
    name: str = "method",
):
    """
    A generic function to retry a method a certain number of times before giving up, with a specific exception.

    :param method: The method to be called.
    :param failure_type: The type of exception to catch.
    :param max_tries: The maximum number
    :param name: The name of the method to be called.
    """
    current_retry = 0
    result = None
    while current_retry < max_tries:
        try:
            result = method()
            break
        except failure_type as e:
            logging.debug(
                f"Caught exception {e} during {name} execution {method}. Retrying..."
            )
            current_retry += 1
    if current_retry == max_tries:
        logging.error(f"Failed to execute {name} {method} after {max_tries} retries.")
    return result
