# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
# @FileName       : Bestman_sim_flexiv_robotiq_2f85.py
# @Time           : 2024-08-03 15:08:13
# @Author         : yk
# @Email          : yangkui1127@gmail.com
# @Description:   : flexiv robot
"""

import math
import time

import pybullet as p
from scipy.spatial.transform import Rotation as R

from .Bestman_sim import Bestman_sim
from .Pose import Pose


class Bestman_sim_flexiv_robotiq_2f85(Bestman_sim):
    """
    A class representing a simulation for the Bestman robot equipped with a flexiv arm.
    """

    def __init__(self, client, visualizer, cfg):
        """
        Initialize the Bestman_sim_flexiv with the given parameters.

        Args:
            client (int): The PyBullet client ID.
            visualizer (bool): Flag indicating whether visualization is enabled.
            cfg (dict): Configuration settings.
        """

        # Init parent class: BestMan_sim
        super().__init__(client, visualizer, cfg)

        # change robot color
        self.visualizer.set_object_color(self.base_id, "light_white")

        # gripper range
        self.gripper_range = [0, 0.085]

        # create gripper constraints
        self.create_eef_constraints()

        # reset gripper
        self.sim_reset_gripper()

    # ----------------------------------------------------------------
    # Functions for arm
    # ----------------------------------------------------------------

    def sim_get_sync_arm_pose(self):
        """
        Get synchronized pose of the robot arm with the base.
        """
        base_pose = self.sim_get_current_base_pose()
        arm_pose = Pose(
            [*base_pose.get_position()[:2], self.arm_place_height],
            base_pose.get_orientation(),
        )
        return arm_pose

    # ----------------------------------------------------------------
    # Functions for gripper
    # ----------------------------------------------------------------

    def create_eef_constraints(self):
        """
        Create constraints for the gripper to control the gripper state。
        """

        # reset gripper joint state
        for jointID in self.controllable_joints:
            if jointID not in self.arm_controllable_joints:
                p.setJointMotorControl2(
                    self.arm_id, jointID, p.VELOCITY_CONTROL, targetVelocity=0, force=0
                )

        # defines the child joints that move synchronously with the parent joint and their movement ratio
        mimic_parent_name = "finger_joint"
        mimic_children_names = {
            "right_outer_knuckle_joint": 1,
            "left_inner_knuckle_joint": 1,
            "right_inner_knuckle_joint": 1,
            "left_inner_finger_joint": -1,
            "right_inner_finger_joint": -1,
        }

        # establish gear ratios for the parent and child joints in the gripper
        self.mimic_parent_id = [
            jointInfo.id
            for jointInfo in self.arm_jointInfo
            if jointInfo.name == mimic_parent_name
        ][0]
        self.mimic_child_multiplier = {
            jointInfo.id: mimic_children_names[jointInfo.name]
            for jointInfo in self.arm_jointInfo
            if jointInfo.name in mimic_children_names
        }
        for joint_id, multiplier in self.mimic_child_multiplier.items():
            c = p.createConstraint(
                self.arm_id,
                self.mimic_parent_id,
                self.arm_id,
                joint_id,
                jointType=p.JOINT_GEAR,
                jointAxis=[0, 1, 0],
                parentFramePosition=[0, 0, 0],
                childFramePosition=[0, 0, 0],
            )
            p.changeConstraint(
                c, gearRatio=-multiplier, maxForce=100, erp=1
            )  # Note: the mysterious `erp` is of EXTREME importance

    def sim_reset_gripper(self):
        "reset gripper"
        self.sim_open_gripper()

    def sim_open_gripper(self):
        """open gripper"""
        self.sim_move_gripper(self.gripper_range[1])
        print("[BestMan_Sim][Gripper] \033[34mInfo\033[0m: Gripper open!")

    def sim_close_gripper(self):
        """close gripper"""
        self.sim_move_gripper(self.gripper_range[0])
        print("[BestMan_Sim][Gripper] \033[34mInfo\033[0m: Gripper close!")

    def sim_move_gripper(self, open_width):
        # open_length = np.clip(open_length, *self.gripper_range)
        open_angle = 0.715 - math.asin(
            (open_width - 0.010) / 0.1143
        )  # angle calculation
        # Control the mimic gripper joint(s)
        p.setJointMotorControl2(
            self.arm_id,
            self.mimic_parent_id,
            p.POSITION_CONTROL,
            targetPosition=open_angle,
            force=self.arm_jointInfo[self.mimic_parent_id].maxForce,
            maxVelocity=self.arm_jointInfo[self.mimic_parent_id].maxVelocity,
        )

    # ----------------------------------------------------------------
    # Functions for interact
    # ----------------------------------------------------------------

    def sim_interactive_control_eef(self, duration=20):
        print("[BestMan_Sim][Gripper] \033[34mInfo\033[0m: Interact start!")
        curr_x, curr_y, curr_z = self.sim_get_current_eef_pose().get_position()
        if "x" not in self.interact_params:
            self.interact_params["x"] = p.addUserDebugParameter(
                "x", curr_x - 0.2, curr_x + 0.2, curr_x
            )
        if "y" not in self.interact_params:
            self.interact_params["y"] = p.addUserDebugParameter(
                "y", curr_y - 0.2, curr_y + 0.2, curr_y
            )
        if "z" not in self.interact_params:
            self.interact_params["z"] = p.addUserDebugParameter(
                "z", curr_z - 0.5, curr_z + 0.5, curr_z
            )
        if "roll" not in self.interact_params:
            self.interact_params["roll"] = p.addUserDebugParameter(
                "roll", -math.pi, math.pi, 0
            )
        if "pitch" not in self.interact_params:
            self.interact_params["pitch"] = p.addUserDebugParameter(
                "pitch", -math.pi, math.pi, math.pi
            )
        if "yaw" not in self.interact_params:
            self.interact_params["yaw"] = p.addUserDebugParameter(
                "yaw", -math.pi / 2, math.pi / 2, -math.pi / 2
            )
        if "gripper_open_width" not in self.interact_params:
            self.interact_params["gripper_open_width"] = p.addUserDebugParameter(
                "gripper_open_width", self.gripper_range[0], self.gripper_range[1], 0.04
            )
        import time

        start_time = time.time()
        while time.time() - start_time < duration:
            x = p.readUserDebugParameter(self.interact_params["x"])
            y = p.readUserDebugParameter(self.interact_params["y"])
            z = p.readUserDebugParameter(self.interact_params["z"])
            roll = p.readUserDebugParameter(self.interact_params["roll"])
            pitch = p.readUserDebugParameter(self.interact_params["pitch"])
            yaw = p.readUserDebugParameter(self.interact_params["yaw"])
            gripper_opening_width = p.readUserDebugParameter(
                self.interact_params["gripper_open_width"]
            )
            self.sim_move_eef_to_goal_pose(Pose([x, y, z], [roll, pitch, yaw]))
            self.sim_move_gripper(gripper_opening_width)
            self.client.run(120)
        print("[BestMan_Sim][Gripper] \033[34mInfo\033[0m: Interact over!")
