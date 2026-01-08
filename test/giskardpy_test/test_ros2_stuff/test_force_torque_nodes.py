from time import sleep

from geometry_msgs.msg import WrenchStamped

from giskardpy.executor import Executor
from giskardpy.motion_statechart.data_types import (
    LifeCycleValues,
    ObservationStateValues,
)
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.ros2_monitors.force_torque_monitor import (
    ForceImpactMonitor,
)
from giskardpy.motion_statechart.test_nodes.test_nodes import ConstTrueNode
from semantic_digital_twin.world import World


def test_force_impact_node(rclpy_node):
    topic_name = "force_torque_topic"
    publisher = rclpy_node.create_publisher(WrenchStamped, topic_name, 10)

    msg_below = WrenchStamped()

    msg_above = WrenchStamped()
    msg_above.wrench.force.x = 20.0

    msc = MotionStatechart()
    msc.add_node(
        node := Sequence(
            nodes=[
                ConstTrueNode(),
                ft_node := ForceImpactMonitor(
                    _ros2_node=rclpy_node, topic_name=topic_name, threshold=10
                ),
                ConstTrueNode(),
                ConstTrueNode(),
            ]
        )
    )
    msc.add_node(EndMotion.when_true(node))

    kin_sim = Executor(world=World())
    kin_sim.compile(motion_statechart=msc)
    kin_sim.tick()
    assert ft_node.observation_state == ObservationStateValues.UNKNOWN
    kin_sim.tick()
    assert ft_node.life_cycle_state == LifeCycleValues.RUNNING
    # unknown because no message received
    assert ft_node.observation_state == ObservationStateValues.UNKNOWN

    publisher.publish(msg_below)
    for i in range(100):
        if ft_node.has_msg():
            break
        sleep(0.1)
    else:
        assert False, "No message received"
    kin_sim.tick()
    # false because force is below threshold
    assert ft_node.observation_state == ObservationStateValues.FALSE

    ft_node.clear_msg()
    publisher.publish(msg_above)
    kin_sim.tick_until_end()
    msc.draw("muh.pdf")
