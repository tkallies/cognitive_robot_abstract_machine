from nav2_msgs.action import NavigateToPose

from giskardpy.motion_statechart.tasks.ros_tasks import (
    ActionServerTask,
    NavigateActionServerTask,
)
from semantic_digital_twin.robots.hsrb import HSRB
from ..datastructures.enums import ExecutionType
from ..robot_plans import MoveMotion, MoveGripperMotion

from ..robot_plans.motions.base import AlternativeMotion


class HSRBMoveMotion(MoveMotion, AlternativeMotion[HSRB]):
    execution_type = ExecutionType.REAL

    def perform(self):
        return

    @property
    def _motion_chart(self) -> NavigateActionServerTask:
        return NavigateActionServerTask(
            target_pose=self.target.to_spatial_type(),
            base_link=self.robot_view.root,
            action_topic="/hsrb/move_base",
            node_handle=self.plan.context.ros_node,
            message_type=NavigateToPose,
        )
