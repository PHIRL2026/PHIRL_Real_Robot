#!/usr/bin/env python
import rospy
import rosbag
from kortex_driver.msg import *
from trajectory_msgs.msg import *
from sensor_msgs.msg import JointState
from kortex_driver.srv import *
import time
import numpy as np
from pathlib import Path
import argparse
import json

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.phirl_arm import Arm



class ArmReplayer:
    def __init__(self, demo_number=None):
        self.folder_name = f"demo_{demo_number}"
        self.parent_dir = Path(__file__).resolve().parent
        self.path = self.parent_dir
        if self.folder_name is not None:
            self.path = f'{self.parent_dir}/bags/' + self.folder_name + '/'
        self.cnt = 0
        self.bag_name = self.path + self.folder_name + '.bag'


        # create annotations folder if not exist
        if not os.path.exists(f'{self.parent_dir}/annotations'):
            os.makedirs(f'{self.parent_dir}/annotations')
            rospy.loginfo(f"Created path: {self.parent_dir}/annotations")

        self.joint_trajectory = ConstrainedJointAngles()
        self.arm = Arm()
        self.arm.clear_faults()
        self.msg = []
        self.transitions = []

    def read_bag(self, cnt=None):
        if cnt is None:
            cnt = self.cnt
        self.bag = rosbag.Bag(self.bag_name, 'r')
        self.msg = []

        state_messages = self.bag.read_messages(
            topics=['/my_gen3_lite/joint_states'])
        transition_messages = self.bag.read_messages(topics=['/transition'])

        for (_, state_msg, _), (_, transition_msg, _) in zip(state_messages, transition_messages):
            temp = JointState()
            temp.header = state_msg.header
            temp.position = state_msg.position
            temp.velocity = state_msg.velocity
            temp.name = state_msg.name
            temp.effort = state_msg.effort

            self.msg.append(temp)
            self.transitions.append(transition_msg.data)

        return self.msg, self.transitions

    def replay_via_joint_state(self, msg=None):
        if msg is None:
            msg = self.msg
        waypoints = []
        for i in range(len(msg)):
            waypoints.append(tuple(msg[i].position[:6]))

        self.arm.send_joint_trajectory(waypoints)

        self.cnt += 1

    def replay_via_joint_state_with_gripper(self, msg=None):
        self.arm.home_arm()
        if msg is None:
            msg = self.msg
        waypoints = []
        for i in range(len(msg)):
            waypoints.append(msg[i].position)
        # print(waypoints)
        self.arm.goto_joint_gripper_waypoints(waypoints)
        self.cnt += 1

    def replay_with_progress_collect(self, frequency=10, cnt=None, msg=None, auto=False):
        self.arm.home_arm()
        progresses = []
        scalars = []
        step = []
        if msg is None:
            msg = self.msg
        for i in range(0, frequency):

            waypoints = []
            for j in range(int(len(msg)/frequency) * i, min(len(msg), int(len(msg)/frequency) * (i + 1))):
                waypoints.append(msg[j].position)
            if i == frequency - 1:
                for j in range(int(len(msg)/frequency) * (i + 1), len(msg)):
                    waypoints.append(msg[j].position)
            self.arm.goto_joint_gripper_waypoints(waypoints)
            step.append(len(waypoints))
            random_number = np.random.rand()
            if auto:
                pass
            else:
                print("your replaying step is: ", i)
                if random_number < 0.5:

                    if len(progresses) > 0:
                        print("your last progress is: ", progresses[-1])

                    progress = input('Please input the progress: ')
                    progresses.append(progress)
                    scalar = input('Please input the scalar: ')
                    scalars.append(scalar)
                else:
                    scalar = input('Please input the scalar: ')
                    scalars.append(scalar)
                    if len(progresses) > 0:
                        print("your last progress is: ", progresses[-1])
                    progress = input('Please input the progress: ')
                    progresses.append(progress)

        print(progresses)
        print(step)
        # write progresses to file
        if cnt is None:
            cnt = self.cnt
        with open(self.path + str(cnt) + "_progress.txt", 'w') as f:
            for item in progresses:
                f.write("%s\n" % item)
            f.close()
        with open(self.path + str(cnt) + "_scalar.txt", 'w') as f:
            for item in scalars:
                f.write("%s\n" % item)
            f.close()
        with open(self.path + str(cnt) + "_step.txt", 'w') as f:
            for item in step:
                f.write("%s\n" % item)
            f.close()
        self.cnt += 1

    def replay_via_joint_state_with_speed(self, msg=None):
        if msg is None:
            msg = self.msg
        for i in range(len(msg)):
            # this duration needs to match demo recorder frequency
            self.arm.joint_velocity_command(msg[i].velocity[:6], 0.1)

    def annotate_replay_via_joint(self, msg=None, num_chunks=5):
        joint_state_chunks = np.array_split(self.msg, num_chunks)
        annotations = self.build_annotation_dict(self.transitions)
        annotations["num_annotations"] = num_chunks
        annotations["num_steps"] = len(self.transitions)
        annotations["annotations"] = []

        counter = 0
        for i in range(len(joint_state_chunks)):
            joint_state_chunk = joint_state_chunks[i]

            self.replay_via_joint_state(joint_state_chunk)
            start_step = counter
            end_step = counter + len(joint_state_chunk)
            print(f"Chunk {i} replayed")

            # get progress
            while True:
                try:
                    user_input = input(
                        f"Please input the progress for time steps {start_step}-{end_step}: ")
                    user_input = float(user_input)
                    break
                except ValueError:
                    continue

            annotation_row = {
                "start_step": start_step,
                "end_step": end_step,
                "start_progress": 0.0 if i == 0 else annotations["annotations"][i-1]["end_progress"],
                "end_progress": user_input
            }
            annotations["annotations"].append(annotation_row)

            counter += len(joint_state_chunk)

        self.write_progress_to_file(annotations)

    def build_annotation_dict(self, transitions):
        annotations = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": []
        }

        for s in transitions:
            t = json.loads(s)

            annotations["observations"].append([round(o, 3) for o in t["obs"]])  # round to 3 decimal places
            annotations["actions"].append(t["action"])
            annotations["rewards"].append(t["reward"])
            annotations["dones"].append(t["done"])

        return annotations

    def write_progress_to_file(self, annotations):
        print("Writing progress to file...")
        with open(f"{self.parent_dir}/annotations/{self.folder_name}_progress.json", "w") as f:
            json.dump(annotations, f, indent=2)

    def out_put_positions(self, msg=None):
        if msg is None:
            msg = self.msg
        poses = []

        for i in range(len(msg)):
            temp = JointState()

            temp.position = msg[i].position
            temp.name = msg[i].name
            # print(temp)
            poses.append(self.arm.get_fk(temp))

        return poses

    def close_bag(self):
        self.bag.close()

    def write_poses_to_file(self, poses, cnt=None):
        if cnt is None:
            cnt = self.cnt
        with open(self.path + str(cnt) + "_poses.txt", 'w') as f:
            for item in poses:
                # print(item.pose.position.x, item.pose.position.y, item.pose.position.z)
                f.write("%s %s %s\n" % (item.pose.position.x,
                        item.pose.position.y, item.pose.position.z))
            f.close()


if __name__ == '__main__':
    rospy.init_node('arm_replayer', anonymous=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("demo_number", help="Demo number for replaying")
    parser.add_argument(
        "--num_chunks",
        type=int,
        default=5, 
        help="Number of chunks to split trajectory into"
    )
    args = parser.parse_args()

    replayer = ArmReplayer(args.demo_number)
    bag = 0
    replayer.read_bag(bag)

    poses = replayer.out_put_positions()
    print(f"Total Steps: {len(poses)}")

    now = time.time()
    replayer.arm.home_arm()
    replayer.annotate_replay_via_joint(num_chunks=int(args.num_chunks))
    print(f"Total time: {time.time() - now}")

    replayer.close_bag()
