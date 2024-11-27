# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
# @FileName       : Bestman_sim_realman.py
# @Time           : 2024-08-03 15:08:13
# @Author         : yk
# @Email          : yangkui1127@gmail.com
# @Description:   : Realman robot
"""

import math

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R

from .Bestman_sim import Bestman_sim
from .Pose import Pose


class Bestman_sim_realman_ag95(Bestman_sim):
    """
    A class representing a simulation for the Bestman robot equipped with a Panda arm.
    """

    def __init__(self, client, visualizer, cfg):
        """
        Initialize the Bestman_sim_panda with the given parameters.

        Args:
            client (int): The PyBullet client ID.
            visualizer (bool): Flag indicating whether visualization is enabled.
            cfg (dict): Configuration settings.
        """

        # Init parent class: BestMan_sim
        super().__init__(client, visualizer, cfg)

    # ----------------------------------------------------------------
    # Functions for arms
    # ----------------------------------------------------------------

    def sim_get_sync_arm_pose(self):
        """
        Get synchronized pose of the robot arm with the base.
        """
        base_pose = self.sim_get_current_base_pose()
        base_position = base_pose.get_position()
        base_orientation = base_pose.get_orientation("rotation_matrix")

        arm_position = base_position + 0.455 * base_orientation[:, 0]
        arm_position[2] = self.arm_place_height
        arm_rotate_matrix = base_orientation @ np.array(
            [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
        )
        arm_pose = Pose(arm_position, arm_rotate_matrix)
        return arm_pose

    # ----------------------------------------------------------------
    # Functions for gripper
    # ----------------------------------------------------------------

    def sim_open_gripper(self):
        """open gripper"""
        self.sim_move_gripper(self.gripper_range[1])
        print("[BestMan_Sim][Gripper] \033[34mInfo\033[0m: Gripper open!")

    def sim_close_gripper(self):
        """close gripper"""
        self.sim_move_gripper(self.gripper_range[0])
        print("[BestMan_Sim][Gripper] \033[34mInfo\033[0m: Gripper close!")

    def sim_move_gripper(self, open_width):
        """move gripper to special width

        Args:
            open_width (float): gripper open width
        """
        # TODO