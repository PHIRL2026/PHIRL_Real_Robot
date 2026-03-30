#!/usr/bin/env python
import rospy
import rosbag
from sensor_msgs.msg import JointState
from std_msgs.msg import Int32
from pynput import keyboard
import time
from armpy import kortex_arm
from geometry_msgs.msg import Pose
from std_msgs.msg import String
import json
import cv2
import numpy as np
import os
from pathlib import Path
import argparse

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.BrushEnv import BrushEnv
from utils.phirl_arm import Arm




class ArmRecorder:
    def __init__(self, demo_number=None):
        self.folder_name = f"demo_{demo_number}"
        self.parent_dir = Path(__file__).resolve().parent
        self.path = self.parent_dir
        self.env = BrushEnv()
        self.latest_env_state, _ = self.env.reset()

        if self.folder_name is not None:
            self.path = f'{self.parent_dir}/bags/' + self.folder_name + '/'

            # create folder if not exist
            if not os.path.exists(self.path):
                os.makedirs(self.path)
                rospy.loginfo(f"Created path: {self.path}")

        self.bag_name = self.path + self.folder_name + '.bag'
        self.bag = rosbag.Bag(self.bag_name, 'w')
        self.is_recording = False
        self.latest_joint_state = None
        self.joint_state_sub = rospy.Subscriber(
            "/my_gen3_lite/joint_states", JointState, self.joint_state_callback)
        self.action_sub = rospy.Subscriber(
            "/discrete_action", Int32, self.action_callback)

        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

        rospy.loginfo("Press s to start recording. Press e to end the recording.")

    def on_press(self, key):
        try:
            if key.char == 's':
                self.start_recording()
            elif key.char == 'e':
                self.end_script()
            elif key.char == 'r':
                self.reset_recording()
        except AttributeError:
            pass

    def reset_recording(self):
        self.stop_recording()
        self.close_bag()
        self.latest_env_state, _ = self.env.reset()
        rospy.sleep(10)
        rospy.loginfo("Re-Record the bag.")
        self.bag = rosbag.Bag(self.bag_name, 'w')

    def start_recording(self):
        if not self.is_recording:
            self.is_recording = True
            rospy.loginfo("Recording started.")

    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            self.env.close()
            rospy.loginfo("Recording stopped.")
            print("Recording stopped.")

    def end_script(self):
        self.close_bag()
        self.listener.stop()
        self.env.close()
        rospy.signal_shutdown("Script ended by user.")
        print("Script ended by user.")

    def joint_state_callback(self, msg):
        self.latest_joint_state = msg

    def action_callback(self, action_msg):
        if self.is_recording and time.time() and self.latest_joint_state:
            try:
                t = rospy.Time.now()

                # take step in environment
                observation, reward, terminated, truncated, info = self.env.step(
                    action_msg.data, move=False)

                # serialize transition into String for ROS message
                transition = {"obs": self.latest_env_state, "action": action_msg.data,
                              "reward": reward, "next_obs": observation, "done": terminated}

                transition_msg = String(data=json.dumps(transition))

                # write joint state (for replay purposes) and transition
                self.bag.write("/my_gen3_lite/joint_states",
                               self.latest_joint_state, t)
                self.bag.write("/transition", transition_msg, t)

                rospy.loginfo(
                    "Recorded step:\n%s",
                    json.dumps(
                        {
                            "obs": self.latest_env_state,
                            "action": action_msg.data,
                            "reward": reward,
                            "next_obs": observation,
                            "done": terminated,
                        },
                    ),
                )
                print("-----------------------------------------------------------")

                # transition into next state
                self.latest_env_state = observation

                # end recording if finished task
                if terminated or truncated:
                    rospy.loginfo("Environment terminated or truncated.")
                    self.end_script()

            except ValueError as e:
                rospy.logwarn("Error writing to bag file: " + str(e))

    def close_bag(self):
        time.sleep(3)
        self.bag.close()
        rospy.loginfo("Bag file saved.")


if __name__ == '__main__':
    rospy.init_node('demo_recorder', anonymous=True, disable_signals=True)
    rospy.Rate(30)  # This rate is not used for recording, only for rospy spin
    parser = argparse.ArgumentParser()
    parser.add_argument("demo_number", help="Demo number for recording")
    args = parser.parse_args()

    recorder = ArmRecorder(args.demo_number)

    while not rospy.is_shutdown():
        rospy.spin()