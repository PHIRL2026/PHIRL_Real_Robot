#!/usr/bin/env python3
# Author: Isaac Sheidlower

"""
    High llevel class to interface with the controller. 
    This class communicates input_profile.py to get 
    mappings of controller input to cartesian commands.

    #TODO: maybe just make the profile with the scales a dictionary?

"""
import rospy
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
from std_msgs.msg import Int32MultiArray

linear_scale = 1
angular_scale = 1
deadzone = .05

class ControllerInterface:
    def __init__(self, profile="default", linear_scale=1, angular_scale=1, deadzone=1, estop_index=0,
        num_buttons=11):
        try:
            rospy.init_node('controller_interface', anonymous=True)
        except:
            pass
        self.profile = profile
        self.rate = rospy.Rate(1000) # 10hz
        self.estop_index = estop_index
        self.linear_scale = linear_scale
        self.angular_scale = angular_scale
        self.deadzone = deadzone
        self.latest_commmad = Twist()
        self.num_buttons = num_buttons
        self.latest_buttons = np.array(np.zeros(self.num_buttons))
        self.joy_sub = rospy.Subscriber("joy", Joy, self.user_input_callback)

    def user_input_callback(self, data):
        global linear_scale, angular_scale, deadzone

        # Get the joystick input
        joy_x = data.axes[0]
        joy_y = data.axes[1]
        joy_z = data.axes[4]

        joy_theta_x = 0
        joy_theta_y = data.axes[3]
        joy_theta_z = 0

        # Apply the deadzone
        if abs(joy_x) < deadzone:
            joy_x = 0
        if abs(joy_y) < deadzone:
            joy_y = 0
        if abs(joy_z) < deadzone:
            joy_z = 0
        if abs(joy_theta_x) < deadzone:
            joy_theta_x = 0
        if abs(joy_theta_y) < deadzone:
            joy_theta_y = 0
        if abs(joy_theta_z) < deadzone:
            joy_theta_z = 0

        # Scale the joystick input
        self.latest_commmad.linear.x = joy_x * linear_scale
        self.latest_commmad.linear.y = joy_y * linear_scale
        self.latest_commmad.linear.z = joy_z * angular_scale
        self.latest_commmad.angular.x = joy_theta_x * angular_scale
        self.latest_commmad.angular.y = joy_theta_y * angular_scale
        self.latest_commmad.angular.z = joy_theta_z * angular_scale

        # joystick buttons
        self.latest_buttons = np.array(data.buttons)

    def dynamic_callback(self, config):
        self.profile = config["profile"]
        self.linear_scale = config["linear_scale"]
        self.angular_scale = config["angular_scale"]
        self.deadzone = config["deadzone"]

    def update_input_profile(self, profile=None, linear_scale=None, angular_scale=None, deadzone=None):
        """
            Function to update any part of the input profile.
        """
        if profile is not None:
            self.profile = profile
        if linear_scale is not None:
            self.linear_scale = linear_scale
        if angular_scale is not None:
            self.angular_scale = angular_scale
        if deadzone is not None:
            self.deadzone = deadzone

        self.dynamic_callback.update_configuration(({"profile":self.profile, "linear_scale":self.linear_scale, \
            "angular_scale":self.angular_scale, "deadzone":self.deadzone}))        

    def get_user_command(self, as_list=False):
        """
            Function to get the user command from the controller augmented by the input profile.
        """
        if not as_list:
            return self.latest_commmad
        else:
            return [self.latest_commmad.linear.x, self.latest_commmad.linear.y, self.latest_commmad.linear.z, \
                self.latest_commmad.angular.x, self.latest_commmad.angular.y, self.latest_commmad.angular.z]

    def get_user_buttons(self):
        """
            Function to get the user button state.
        """
        return self.latest_buttons
    
    def get_user_input(self):
        """
            Function to get the user input from the controller.
            This function returns a tuple of the cartesian command
            and the button state. 
        """
        return self.latest_commmad, self.latest_buttons

    def get_profile(self):
        return self.profile
    
    def get_linear_scale(self):
        return self.linear_scale
    
    def get_angular_scale(self):
        return self.angular_scale
    
    def get_deadzone(self):
        return self.deadzone

if __name__ == '__main__':
    try:
        controller_interface = ControllerInterface()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass