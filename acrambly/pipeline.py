import rclpy
from lark.visitors import TransformerChain
from perception_interfaces.msg import CubePoses
from perception_interfaces.srv import GetCubePoses, GetObjPoses

import geometry_msgs.msg
from example_interfaces.srv import Trigger

from pycram.datastructures.pose import PoseStamped
from rclpy import Node

from semantic_digital_twin.spatial_types.spatial_types import HomogeneousTransformationMatrix as TransformationMatrix
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World

from pycram.ros import create_subscriber

prefix = "PhysicalObject"
transform: TransformationMatrix

def update_obj_positions(poses, world: World):
    for pose in poses:
        obj_name = pose[0]
        obj_pose = geometry_msgs.msg.PoseStamped()
        obj_trans = TransformationMatrix.from_xyz_quaternion(obj_pose.pose.position.x, obj_pose.pose.position.y,
                                                             obj_pose.pose.position.z, obj_pose.pose.orientation.x,
                                                             obj_pose.pose.orientation.y, obj_pose.pose.orientation.z,
                                                             obj_pose.pose.orientation.w, 
                                                             world.get_kinematic_structure_entity_by_name(
                                                                 PrefixedName(obj_pose.header.frame_id, "tracy")))
        if not transform:
            body = world.get_kinematic_structure_entity_by_name(PrefixedName(obj_pose.header.frame_id, "tracy"))
            transform = body.parent_connection.origin
            body = body.parent_connection.parent
            while body.name != PrefixedName(name='map', prefix='tracy'):
                transform = transform @ body.parent_connection.origin
                body = body.parent_connection.parent

        with world.modify_world():
            world.get_connection_by_name(PrefixedName("map_T_" + obj_name, prefix)).origin = obj_trans @ transform
class PerceptionClient:
    def __init__(self, world, node):
        self.world = world
        self.node = node
        self.client = self.node.create_client(Trigger, "save_camera_frames")
        create_subscriber("/Current_OBJ_position_0", geometry_msgs.msg.PoseStamped, self.callback0, 10)
        create_subscriber("/Current_OBJ_position_1", geometry_msgs.msg.PoseStamped, self.callback1, 10)
        create_subscriber("/Current_OBJ_position_2", geometry_msgs.msg.PoseStamped, self.callback2, 10)
        self.cube1_pose = PoseStamped()
        self.cube2_pose = PoseStamped()
        self.cube3_pose = PoseStamped()
        self.req = Trigger.Request()

    def callback0(self, msg):
        self.cube1_pose = msg
        print("received cube1 pose")

    def callback1(self, msg):
        self.cube2_pose = msg
        print("received cube2 pose")

    def callback2(self, msg):
        self.cube3_pose =msg
        print("received cube3 pose")

    def request(self):
        future = self.client.call_async(self.req)
        rclpy.spin_until_future_complete(self.node, future)
        response = future.result()
        print(response)
        poses = [("Cube_1", self.cube1_pose), ("Cube_2", self.cube2_pose), ("Cube_3", self.cube3_pose)]
        print(poses)
        update_obj_positions(poses, self.world)

class PerceptionClientNew:
    def __init__(self, world, node):
        self.world = world
        self.node = node
        self.client = self.node.create_client(GetCubePoses, "save_camera_frames")
        self.req = GetCubePoses.Request()

    def request(self):
        future = self.client.call_async(self.req)
        rclpy.spin_until_future_complete(self.node, future)
        response = future.result()
        poses = [("Cube_1", response.cube_1_pose), ("Cube_2", response.cube_2_pose), ("Cube_3", response.cube_3_pose)]
        update_obj_positions(poses, self.world)

class PerceptionClientSingle:
    def __init__(self, world, node):
        self.world = world
        self.node = node
        self.client = self.node.create_client(GetObjPoses, "save_camera_frames")
        self.req = GetObjPoses.Request()

    def request(self, obj):
        self.req.object = obj
        future = self.client.call_async(self.req)
        rclpy.spin_until_future_complete(self.node, future)
        response = future.result()
        poses = [(obj, response.cube_1_pose)]
        update_obj_positions(poses, self.world)