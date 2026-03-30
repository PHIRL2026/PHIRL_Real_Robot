import rospy
from std_msgs.msg import Int32
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.phirl_arm import Arm
from demonstrations.controller_interface import ControllerInterface
from demonstrations.DiscreteActionWrapper import DiscreteActionWrapper, DiscreteActions
from env.BrushEnv import BrushEnv, DURATION, SCALE

"""
Node for listening to controller inputs, converting it to discrete action, and publishing the action. 
"""
class ControllerNode:
    def __init__(self, teleop=True):
        
        self.action_pub = rospy.Publisher("/discrete_action", Int32, queue_size=1)
        self.teleop = teleop

        # Arm components
        self.arm = Arm()
        self.controller = ControllerInterface()
        self.wrapper = DiscreteActionWrapper()
        self.env = BrushEnv()

        # Arm set up
        self.arm.clear_faults()
        self.arm.home_arm()

        rospy.on_shutdown(self.arm.stop)
        rospy.loginfo("Controller Node ready!")

    def step(self):
        # handle close/open gripper
        buttons = self.controller.get_user_buttons()
        self.handle_buttons(buttons)

        # handle joystick commands to discrete action
        cmd = self.controller.get_user_command()
        action = self.wrapper.get_action(cmd, buttons)

        velocity = np.array(action.to_cartesian_array()) * SCALE

        if action is None or action == DiscreteActions.NO_OP:
            return
        
        if self.env._valid_action(action):
            self.arm.cartesian_velocity_command(
                velocity,
                duration=DURATION,
                radians=True
            )
            print(self.arm.get_cartesian_pose())
        else:
            rospy.sleep(0.5)
        
        self.action_pub.publish(int(action))     

    def handle_buttons(self, buttons):
        # close or open gripper
        if buttons[4]:
            self.arm.send_relative_gripper_command(0.1)
        elif buttons[5]:
            self.arm.send_relative_gripper_command(-0.1)