# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
# @FileName       : grasp_bowl_on_table_vacuum_gripper.py
# @Time           : 2024-08-03 15:03:52
# @Author         : yk
# @Email          : yangkui1127@gmail.com
# @Description:   : A example to grasp bowl on table use vacuum_gripper (simplified to just display robot model)
"""

import os

from Config import load_config
from Env import Client
from Motion_Planning.Navigation import *
from Robotics_API import Bestman_sim_ur5e_robotiq_2f85
from Visualization import Visualizer


def main(filename):

    # Load config
    config_path = "Config/ur5e_robotiq_interact.yaml"
    cfg = load_config(config_path)
    print(cfg)

    # Init client and visualizer
    client = Client(cfg.Client)
    visualizer = Visualizer(client, cfg.Visualizer)
    visualizer.draw_axes()

    # Start record
    visualizer.start_record(filename)

    # Init robot
    ur5e = Bestman_sim_ur5e_robotiq_2f85(client, visualizer, cfg)

    # Interact with arm
    # ur5e.sim_interactive_control_arm(1000)
    ur5e.sim_interactive_control_eef(100)

    # client.wait(10)
    # visualizer.capture_screen("ur5e")

    # End record
    visualizer.end_record()

    # disconnect from server
    client.wait(5)
    client.disconnect()


if __name__ == "__main__":

    # set work dir to Examples
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # get current file name
    filename = os.path.splitext(os.path.basename(__file__))[0]

    main(filename)
