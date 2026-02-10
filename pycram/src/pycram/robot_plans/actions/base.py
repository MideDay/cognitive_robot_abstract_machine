from __future__ import annotations

import os.path
from abc import abstractmethod
import logging
from dataclasses import dataclass

from typing_extensions import Any, Optional, Callable, TypeVar

from ...designator import DesignatorDescription
from ...failures import PlanFailure
from ...has_parameters import HasParameters

logger = logging.getLogger(__name__)


@dataclass
class ActionDescription(DesignatorDescription, HasParameters):
    _pre_perform_callbacks = []
    _post_perform_callbacks = []

    def __post_init__(self):
        pass
        # self._pre_perform_callbacks.append(self._update_robot_params)

    def perform(self) -> Any:
        """
        Full execution: pre-check, plan, post-check
        """
        logger.info(f"Performing action {self.__class__.__name__}")

        for pre_cb in self._pre_perform_callbacks:
            pre_cb(self)

        self.pre_condition()

        result = None
        try:
            result = self.execute()
        except PlanFailure as e:
            raise e
        finally:
            pass
            # for post_cb in self._post_perform_callbacks:
            #     post_cb(self)
            #
            # self.validate_postcondition(result)

        return result

    @abstractmethod
    def execute(self) -> Any:
        """
        Symbolic plan. Should only call motions or sub-actions.
        """
        pass

    @abstractmethod
    def pre_condition(self):
        pass

    @abstractmethod
    def post_condition(self):
        pass

    @property
    def validate_precondition(self) -> bool:
        """
        Symbolic/world state precondition validation.
        """
        return True

    @property
    def validate_postcondition(self) -> bool:
        """
        Symbolic/world state postcondition validation.
        """
        return True

    @classmethod
    def pre_perform(cls, func) -> Callable:
        cls._pre_perform_callbacks.append(func)
        return func

    @classmethod
    def post_perform(cls, func) -> Callable:
        cls._post_perform_callbacks.append(func)
        return func


ActionType = TypeVar("ActionType", bound=ActionDescription)
