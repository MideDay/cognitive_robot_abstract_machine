from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import pytest
import rclpy
import std_msgs.msg
from rclpy.node import Node

from giskardpy.middleware.ros2.client_presence import (
    ClientHeartbeatPublisher,
    ClientPresence,
    ClientWatchdog,
    GraphPresence,
    HeartbeatPresence,
)
from giskardpy.middleware.ros2.exceptions import NoWatchedClientError
from krrood.adapters.json_serializer import to_json
from semantic_digital_twin.adapters.ros.messages import MetaData

# %% helpers


@dataclass
class SteppingClock:
    """
    A clock that only moves when a test moves it.
    """

    now: float = 0.0
    """
    The time this clock currently reads.
    """

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        """
        Let the given number of seconds pass.
        """
        self.now += seconds


def wait_until(condition: Callable[[], bool], timeout: float = 5.0) -> bool:
    """
    Give the ros graph time to catch up with what a test just did.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


def create_client(node_name: str = "some_client") -> MetaData:
    """
    The identity a client names itself with.
    """
    return MetaData(node_name=node_name, process_id=1234)


def heartbeat_of(client: MetaData) -> std_msgs.msg.String:
    """
    The message that client sends to announce itself.
    """
    return std_msgs.msg.String(data=json.dumps(to_json(client)))


@dataclass
class KnownClientPresence(ClientPresence):
    """
    Stands in for a check that recognizes one client and is told whether it is there.
    """

    known_client: Optional[MetaData] = None
    """
    The only client this check recognizes.
    """

    present: bool = True
    """
    What this check reports about the watched client.
    """

    def start_watching(self, client: MetaData) -> bool:
        if client != self.known_client:
            return False
        self.watched_client = client
        return True

    def is_client_present(self) -> bool:
        return self.present


@dataclass
class UnknownClientPresence(ClientPresence):
    """
    Stands in for a check that recognizes no client at all.
    """

    asked_about: List[MetaData] = field(default_factory=list)
    """
    Every client this check was offered.
    """

    def start_watching(self, client: MetaData) -> bool:
        self.asked_about.append(client)
        return False

    def is_client_present(self) -> bool:
        return False


# %% the heartbeats a client sends


class TestClientHeartbeat:
    """
    The heartbeat has to reach Giskard, which means both sides have to agree on where it
    is sent and what it says.
    """

    def test_giskard_receives_the_heartbeat_of_a_client(self, rclpy_node: Node):
        presence = HeartbeatPresence(node=rclpy_node)
        client_node = rclpy.create_node("heartbeat_sender")
        client = MetaData(node_name=client_node.get_name(), process_id=7)
        publisher = ClientHeartbeatPublisher(
            node=client_node,
            client=client,
            giskard_node_name=rclpy_node.get_name(),
        )
        try:
            assert wait_until(
                lambda: publisher.publish() or client in presence.last_heartbeat
            )
        finally:
            publisher.stop()
            client_node.destroy_node()

        assert presence.start_watching(client)
        assert presence.is_client_present()


# %% reading the heartbeats


class TestHeartbeatPresence:
    """
    A client counts as gone once its heartbeats stop arriving.
    """

    def test_a_client_that_just_announced_itself_is_present(self, rclpy_node: Node):
        clock = SteppingClock()
        presence = HeartbeatPresence(node=rclpy_node, clock=clock)
        client = create_client()
        presence.receive_heartbeat(heartbeat_of(client))

        assert presence.start_watching(client)
        assert presence.is_client_present()

    def test_a_client_that_stopped_announcing_itself_is_gone(self, rclpy_node: Node):
        clock = SteppingClock()
        presence = HeartbeatPresence(node=rclpy_node, clock=clock)
        client = create_client()
        presence.receive_heartbeat(heartbeat_of(client))
        presence.start_watching(client)

        clock.advance(presence.timeout.total_seconds() + 0.01)

        assert not presence.is_client_present()

    def test_a_client_stays_present_while_it_keeps_announcing_itself(
        self, rclpy_node: Node
    ):
        clock = SteppingClock()
        presence = HeartbeatPresence(node=rclpy_node, clock=clock)
        client = create_client()
        presence.receive_heartbeat(heartbeat_of(client))
        presence.start_watching(client)

        for _ in range(5):
            clock.advance(presence.timeout.total_seconds())
            presence.receive_heartbeat(heartbeat_of(client))

        assert presence.is_client_present()

    def test_a_client_that_never_announced_itself_is_not_watched(
        self, rclpy_node: Node
    ):
        presence = HeartbeatPresence(node=rclpy_node, clock=SteppingClock())

        assert not presence.start_watching(create_client())

    def test_the_heartbeat_of_one_client_says_nothing_about_another(
        self, rclpy_node: Node
    ):
        presence = HeartbeatPresence(node=rclpy_node, clock=SteppingClock())
        presence.receive_heartbeat(heartbeat_of(create_client("other_client")))

        assert not presence.start_watching(create_client("some_client"))

    def test_a_check_that_watches_nothing_cannot_be_asked(self, rclpy_node: Node):
        presence = HeartbeatPresence(node=rclpy_node, clock=SteppingClock())

        with pytest.raises(NoWatchedClientError):
            presence.is_client_present()


# %% watching the ros graph


class TestGraphPresence:
    """
    A client that sends no heartbeat is watched through the subscriptions its action
    client keeps on the feedback of the action.
    """

    action_name = "mimic_giskard/command"
    """
    The action whose clients are watched.
    """

    def create_feedback_subscriber(self, node_name: str) -> Node:
        """
        A node that listens to the feedback of the action, the way an action client
        does.
        """
        node = rclpy.create_node(node_name)
        node.create_subscription(
            std_msgs.msg.String,
            topic=f"{self.action_name}/_action/feedback",
            callback=lambda message: None,
            qos_profile=10,
        )
        return node

    def test_a_subscribed_client_is_present(self, rclpy_node: Node):
        presence = GraphPresence(node=rclpy_node, action_name=self.action_name)
        client_node = self.create_feedback_subscriber("graph_watched_client")
        client = MetaData(node_name=client_node.get_name(), process_id=7)
        try:
            assert wait_until(lambda: presence.start_watching(client))

            assert presence.is_client_present()
        finally:
            client_node.destroy_node()

    def test_a_client_that_unsubscribed_is_gone(self, rclpy_node: Node):
        presence = GraphPresence(node=rclpy_node, action_name=self.action_name)
        client_node = self.create_feedback_subscriber("graph_leaving_client")
        client = MetaData(node_name=client_node.get_name(), process_id=7)
        assert wait_until(lambda: presence.start_watching(client))

        client_node.destroy_node()

        assert wait_until(lambda: not presence.is_client_present())

    def test_a_client_that_is_not_subscribed_is_not_watched(self, rclpy_node: Node):
        presence = GraphPresence(node=rclpy_node, action_name=self.action_name)

        assert not presence.start_watching(create_client("never_subscribed_client"))

    def test_a_namesake_of_the_dead_client_does_not_pass_as_the_dead_client(
        self, rclpy_node: Node
    ):
        """
        A client that is restarted under the same node name is a different client, and
        the goal of the one that died is still nobody's.
        """
        presence = GraphPresence(node=rclpy_node, action_name=self.action_name)
        client_node = self.create_feedback_subscriber("graph_restarted_client")
        client = MetaData(node_name=client_node.get_name(), process_id=7)
        assert wait_until(lambda: presence.start_watching(client))
        client_node.destroy_node()
        assert wait_until(lambda: not presence.is_client_present())

        namesake = self.create_feedback_subscriber("graph_restarted_client")
        try:
            assert wait_until(
                lambda: bool(presence.subscribed_endpoints_of(client)), timeout=2.0
            )

            assert not presence.is_client_present()
        finally:
            namesake.destroy_node()


# %% watching the client of a goal


class TestClientWatchdog:
    """
    The watchdog reports on the client of the running goal, using whichever check knows
    that client.
    """

    def test_the_first_check_that_recognizes_the_client_is_used(self):
        client = create_client()
        heartbeat_like = KnownClientPresence(known_client=client)
        graph_like = KnownClientPresence(known_client=client)
        watchdog = ClientWatchdog(checks=[heartbeat_like, graph_like])

        watchdog.watch(client)

        assert watchdog.watching is heartbeat_like
        assert graph_like.watched_client is None

    def test_a_client_the_first_check_does_not_know_falls_through(self):
        client = create_client()
        unknown = UnknownClientPresence()
        known = KnownClientPresence(known_client=client)
        watchdog = ClientWatchdog(checks=[unknown, known])

        watchdog.watch(client)

        assert watchdog.watching is known
        assert unknown.asked_about == [client]

    def test_a_client_no_check_knows_is_never_reported_gone(self):
        """
        Stopping a goal because nothing recognized its client would break every client
        that Giskard simply cannot see.
        """
        watchdog = ClientWatchdog(checks=[UnknownClientPresence()])

        watchdog.watch(create_client())

        assert watchdog.watching is None
        assert not watchdog.is_client_gone()

    def test_a_client_that_left_is_reported_gone(self):
        client = create_client()
        check = KnownClientPresence(known_client=client)
        watchdog = ClientWatchdog(checks=[check])
        watchdog.watch(client)

        check.present = False

        assert watchdog.is_client_gone()
        assert watchdog.client == client

    def test_a_finished_goal_releases_its_check(self):
        client = create_client()
        check = KnownClientPresence(known_client=client)
        watchdog = ClientWatchdog(checks=[check])
        watchdog.watch(client)

        watchdog.stop_watching()

        assert watchdog.watching is None
        assert check.watched_client is None
        assert not watchdog.is_client_gone()

    def test_the_client_of_no_goal_cannot_be_asked_for(self):
        watchdog = ClientWatchdog(checks=[])

        with pytest.raises(NoWatchedClientError):
            watchdog.client
