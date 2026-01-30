from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.hsrb import HSRB

from ..datastructures.enums import (
    StaticJointState,
    Arms,
    GripperState as GripperStateEnum,
    TorsoState,
)
from ..joint_state import (
    PyCRAMJointState,
    ArmStatePyCRAM,
    GripperStatePyCRAM,
    JointStateManager,
)

left_arm = ArmStatePyCRAM(
    name=PrefixedName("hsrb", "arm"),
    joint_names=[
        "arm_flex_joint",
        "arm_roll_joint",
        "wrist_flex_joint",
        "wrist_roll_joint",
    ],
    joint_positions=[0.0, 1.5, -1.85, 0.0],
    state_type=StaticJointState.Park,
    arm=Arms.LEFT,
)

both_arm = ArmStatePyCRAM(
    name=PrefixedName("hsrb", "arm"),
    joint_names=[
        "arm_flex_joint",
        "arm_roll_joint",
        "wrist_flex_joint",
        "wrist_roll_joint",
    ],
    joint_positions=[0.0, 1.5, -1.85, 0.0],
    state_type=StaticJointState.Park,
    arm=Arms.BOTH,
)

left_gripper_open = GripperStatePyCRAM(
    name=PrefixedName("hsrb", "left_gripper_open"),
    joint_names=["hand_l_proximal_joint", "hand_r_proximal_joint", "hand_motor_joint"],
    joint_positions=[0.3, 0.3, 0.3],
    state_type=GripperStateEnum.OPEN,
    gripper=Arms.LEFT,
)

left_gripper_close = GripperStatePyCRAM(
    name=PrefixedName("hsrb", "left_gripper_close"),
    joint_names=["hand_l_proximal_joint", "hand_r_proximal_joint", "hand_motor_joint"],
    joint_positions=[0.0, 0.0, 0.0],
    state_type=GripperStateEnum.CLOSE,
    gripper=Arms.LEFT,
)

torso_low = PyCRAMJointState(
    name=PrefixedName("hsrb", "torso_low"),
    joint_names=["torso_lift_joint"],
    joint_positions=[0.34],
    state_type=TorsoState.LOW,
)

torso_mid = PyCRAMJointState(
    name=PrefixedName("hsrb", "torso_mid"),
    joint_names=["torso_lift_joint"],
    joint_positions=[0.17],
    state_type=TorsoState.MID,
)

torso_high = PyCRAMJointState(
    name=PrefixedName("hsrb", "torso_high"),
    joint_names=["torso_lift_joint"],
    joint_positions=[0.0],
    state_type=TorsoState.HIGH,
)


JointStateManager().add_joint_states(
    HSRB,
    [
        both_arm,
        left_arm,
        left_gripper_open,
        left_gripper_close,
        torso_low,
        torso_mid,
        torso_high,
    ],
)
