# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
# @FileName       : load_realman.py
# @Time           : 2024-08-03 15:04:49
# @Author         : yk
# @Email          : yangkui1127@gmail.com
# @Description:   : A example to load realman robot
"""


import math
import os
import pybullet as p
from Config import load_config
from Env.Client import Client
from Motion_Planning.Navigation import *
from Robotics_API import Bestman_sim_realman_ag95, Pose
from SLAM import simple_slam
from Visualization.Visualizer import Visualizer

def control_robot_with_keyboard(client, realman):
    """
    Controls the robot using keyboard input in PyBullet.

    Args:
        realman (Bestman_sim_realman): The robot object to control.
    """
    
    print("\n====================================")
    print("====================================")
    print("Keyboard controls:")
    print("↑: Move forward")
    print("↓: Move backward")
    print("←: Rotate counter-clockwise")
    print("→: Rotate clockwise")
    print("ESC: Exit")
    print("====================================")
    print("====================================")


    while True:
        
        # Capture keyboard events
        keys = p.getKeyboardEvents()

        # Check for specific keys
        if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
            realman.sim_move_base_forward(0.05)
        elif p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
            realman.sim_move_base_backward(0.05)
        elif p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
            realman.sim_rotate_base(1, "counter-clockwise")
        elif p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
            realman.sim_rotate_base(1, "clockwise")
        elif 27 in keys and keys[27] & p.KEY_IS_DOWN:  # ESC
            print("Exit control...")
            return

        client.run()

def main(filename):

    # load config
    config_path = "Config/load_realman.yaml"
    cfg = load_config(config_path)
    print(cfg)

    # Init client and visualizer
    client = Client(cfg.Client)
    visualizer = Visualizer(client, cfg.Visualizer)
    visualizer.draw_axes()

    # Start record
    visualizer.start_record(filename)

    # Init robot
    realman = Bestman_sim_realman_ag95(client, visualizer, cfg)

    # # Rotate base
    # realman.sim_rotate_base(180, "counter-clockwise")

    # realman.sim_move_base_forward(2)

    # realman.sim_move_base_backward(2)

    # realman.sim_move_base_left(2)

    # realman.sim_move_base_right(2)
    
    control_robot_with_keyboard(client, realman)

    # End record
    visualizer.end_record()

    # disconnect
    client.wait(10)
    client.disconnect()


if __name__ == "__main__":

    # set work dir to Examples
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # get current file name
    file_name = os.path.splitext(os.path.basename(__file__))[0]

    main(file_name)
