from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Union, Type

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.world import World

from .datastructures.enums import Arms, StaticJointState
from .helper import Singleton


@dataclass
class PyCRAMJointState:
    """
    Represents a named joint state of a robot. For example, the park position of the arms.
    """

    name: PrefixedName
    """
    Name of the joint state
    """

    joint_names: List[str]
    """
    Names of the joints in this state
    """

    joint_positions: List[float]
    """
    position of the joints in this state, must correspond to the joint_names
    """

    state_type: Enum = None
    """
    Enum type of the joints tate (e.g., Park, Open)
    """

    def apply_to_world(self, world: World):
        """
        Applies the joint state to the robot in the given world.
        :param world: The world in which the robot is located.
        """
        for joint_name, joint_position in zip(self.joint_names, self.joint_positions):
            dof = list(world.get_connection_by_name(joint_name).dofs)[0]
            world.state[dof.id].position = joint_position
        world.notify_state_change()


@dataclass
class ArmStatePyCRAM(PyCRAMJointState):
    arm: Arms = None


@dataclass
class GripperStatePyCRAM(PyCRAMJointState):
    """
    Represents the state of a gripper, such as open or closed.
    """

    gripper: Arms = None


@dataclass
class JointStateManager(metaclass=Singleton):
    """
    Manages joint states for different robot arms and their configurations.
    """

    joint_states: Dict[Type[AbstractRobot], List[PyCRAMJointState]] = field(
        default_factory=dict
    )
    """
    A list of joint states that can be applied to the robot.
    """

    def get_arm_state(
        self, arm: Arms, state_type: StaticJointState, robot_view: AbstractRobot
    ) -> Union[ArmStatePyCRAM, None]:
        """
        Retrieves the joint state for a specific arm and state type.

        :param arm: The arm for which the state is requested.
        :param state_type: The type of state (e.g., Park).
        :param robot_view: The robot view to which the arm belongs.
        :return: The corresponding ArmState or None if not found.
        """
        for joint_state in self.joint_states[robot_view.__class__]:
            if (
                isinstance(joint_state, ArmStatePyCRAM)
                and joint_state.arm == arm
                and joint_state.state_type == state_type
            ):
                return joint_state
        return None

    def get_gripper_state(
        self, gripper: Arms, state_type: StaticJointState, robot_view: AbstractRobot
    ) -> Union[GripperStatePyCRAM, None]:
        """
        Retrieves the joint state for a specific gripper and state type.

        :param gripper: The gripper for which the state is requested.
        :param state_type: The type of state (e.g., Open, Close).
        :param robot_view: The robot view to which the gripper belongs.
        :return: The corresponding GripperState or None if not found.
        """
        for joint_state in self.joint_states[robot_view.__class__]:
            if (
                isinstance(joint_state, GripperStatePyCRAM)
                and joint_state.gripper == gripper
                and joint_state.state_type == state_type
            ):
                return joint_state
        return None

    def get_joint_state(
        self, state: Enum, robot_view: AbstractRobot
    ) -> List[PyCRAMJointState]:
        """
        Retrieves all joint states of a specific type for a given robot.

        :param state: The type of joint state to retrieve (e.g., Park, Open).
        :param robot_view: The robot class for which the joint states are requested.
        :return: A list of JointState objects matching the specified type.
        """
        return [
            js
            for js in self.joint_states.get(robot_view.__class__, [])
            if js.state_type == state
        ]

    def add_joint_states(
        self, robot: Type[AbstractRobot], joint_states: List[PyCRAMJointState]
    ):
        """
        Adds joint states for a specific robot type.

        :param robot: The robot class for which the joint states are added.
        :param joint_states: A list of joint states to be added.
        """
        if robot not in self.joint_states:
            self.joint_states[robot] = []
        self.joint_states[robot].extend(joint_states)
