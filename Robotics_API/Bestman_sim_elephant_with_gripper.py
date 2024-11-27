# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
# @FileName       : Bestman_sim_elephant.py
# @Time           : 2024-08-03 15:08:13
# @Author         : yk
# @Email          : yangkui1127@gmail.com
# @Description:   : elephant robot
"""

import math
import time

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R

from .Bestman_sim import Bestman_sim
from .Pose import Pose


class Bestman_sim_elephant_with_gripper(Bestman_sim):
    """
    A class representing a simulation for the Bestman robot equipped with a elephant arm.
    """

    def __init__(self, client, visualizer, cfg):
        """
        Initialize the Bestman_sim_elephant with the given parameters.

        Args:
            client (int): The PyBullet client ID.
            visualizer (bool): Flag indicating whether visualization is enabled.
            cfg (dict): Configuration settings.
        """

        # Init parent class: BestMan_sim
        super().__init__(client, visualizer, cfg)

        # change robot color
        self.visualizer.set_object_color(self.base_id, "light_white")

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
