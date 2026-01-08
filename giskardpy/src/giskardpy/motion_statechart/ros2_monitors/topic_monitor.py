from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from rclpy.node import Node, MsgType
from rclpy.qos import QoSProfile
from rclpy.subscription import Subscription
from typing_extensions import Generic

from ..context import ExecutionContext
from ..data_types import ObservationStateValues

from ..graph_node import MotionStatechartNode


@dataclass
class TopicSubscriberNode(MotionStatechartNode, Generic[MsgType]):
    _ros2_node: Node
    topic_name: str
    msg_type: MsgType
    qos_profile: QoSProfile | int = 10
    _subscriber: Subscription = field(init=False)
    current_msg: MsgType | None = field(init=False, default=None)
    __last_msg: MsgType | None = field(init=False, default=None)

    def __post_init__(self):
        super().__post_init__()
        self._subscriber = self._ros2_node.create_subscription(
            msg_type=self.msg_type,
            topic=self.topic_name,
            callback=self.callback,
            qos_profile=self.qos_profile,
        )

    def callback(self, msg: MsgType):
        self.__last_msg = msg

    def has_msg(self) -> bool:
        return self.__last_msg is not None

    def clear_msg(self):
        self.__last_msg = None

    def on_tick(self, context: ExecutionContext) -> Optional[ObservationStateValues]:
        self.current_msg = self.__last_msg
