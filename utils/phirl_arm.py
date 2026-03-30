#!/usr/bin/env python3

import sys
import rospy
import time
from typing import Tuple, List
import numpy as np
import threading


from kortex_driver.srv import *
from kortex_driver.msg import *
from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotState, PositionIKRequest
from moveit_msgs.srv import GetPositionFK
from tf.transformations import euler_from_quaternion


Pose = Tuple[float, float, float, float, float, float]
JointPose = Tuple[float, float, float, float, float, float]


class Arm:
    def __init__(self):
        try:
            # rospy.init_node('HRI_PROJECT')
            rospy.loginfo(f"Loading robot...")

            self.HOME_ACTION_IDENTIFIER = 2

            # Get node params
            # self.robot_name = rospy.get_param('~robot_name', "my_gen3")
            # self.degrees_of_freedom = rospy.get_param(
            #     "/" + self.robot_name + "/degrees_of_freedom", 7)
            # self.is_gripper_present = rospy.get_param(
            #     "/" + self.robot_name + "/is_gripper_present", False)
            self.robot_name = "my_gen3_lite"
            self.degrees_of_freedom = 6
            self.is_gripper_present = True

            rospy.loginfo("Using robot_name " + self.robot_name + " , robot has " + str(self.degrees_of_freedom) +
                          " degrees of freedom and is_gripper_present is " + str(self.is_gripper_present))

            # Init the action topic subscriber
            self.action_topic_sub = rospy.Subscriber(
                "/" + self.robot_name + "/action_topic", ActionNotification, self.cb_action_topic)
            self.last_action_notif_type = None

            # Init the services
            self.__init_services()

            self.stop_publisher = rospy.Publisher(
                f"/{self.robot_name}/in/stop", std_msgs.msg.Empty, queue_size=1, latch=True)
        except:
            self.is_init_success = False
        else:
            self.is_init_success = True

    def __init_services(self):
        # name: (service type, attribute name)
        services = {
            "base/clear_faults":              (Base_ClearFaults, "clear_faults_service"),
            "base/read_action":              (ReadAction, "read_action"),
            "base/execute_action":           (ExecuteAction, "execute_action_service"),
            "control_config/set_cartesian_reference_frame":
            (SetCartesianReferenceFrame, "set_cartesian_reference_frame_service"),
            "base/send_gripper_command":     (SendGripperCommand, "send_gripper_command_service"),
            "base/activate_publishing_of_action_topic":
            (OnNotificationActionTopic, "activate_publishing_of_action_notification"),
            "base/validate_waypoint_list":   (ValidateWaypointList, "validate_waypoint_list"),
            "compute_fk": (GetPositionFK, "compute_fk_service")
        }

        for name, (srv_type, attr_name) in services.items():
            full_name = f"/{self.robot_name}/{name}"
            rospy.wait_for_service(full_name)
            setattr(self, attr_name, rospy.ServiceProxy(full_name, srv_type))

    def cb_action_topic(self, notif):
        self.last_action_notif_type = notif.action_event

    def stop(self):
        """
        Stops the arm from moving
        """
        # rospy.loginfo("Stopping the arm from moving")
        # self.stop_publisher.publish()

        # this is a hacky solution to stopping. Instead of using the stop service, we publish 0 velocity to the arm.
        # This makes the trajectory when changing directions rapidly smoother when doing teleop. Not sure why using the stop
        # service makes it clunky.
        cartesian_vel_publisher = rospy.Publisher(
            f"/{self.robot_name}/in/cartesian_velocity", TwistCommand, queue_size=1, latch=True)

        cmd = TwistCommand()

        # Fill message
        cmd.twist.linear_x = 0
        cmd.twist.linear_y = 0
        cmd.twist.linear_z = 0
        cmd.twist.angular_x = 0
        cmd.twist.angular_y = 0
        cmd.twist.angular_z = 0

        cmd.reference_frame = 0
        cmd.duration = 1
        cartesian_vel_publisher.publish(cmd)

        return

    def get_fk(self, joints=None):
        """
        Returns fk of joints as a PoseStamped message. If no 
        fk solution is found will return -1.

        joints: list or dictionary of joints, 
            lists are assumed to be in order of joint1-6/7
        """

        if joints is None:
            joint_state = rospy.wait_for_message(
                f"/{self.robot_name}/joint_states", JointState)
        elif isinstance(joints, JointState):
            joint_state = joints
        else:
            joint_state = JointState()
            if isinstance(joints, list):
                print("get in!")
                joint_state.name = [
                    f"joint{i+1}" for i in range(self.degrees_of_freedom)]
                joint_state.position = joints
            else:
                joint_state.name = list(joints.keys())
                # order is not technically guaranteed
                joint_state.position = [joints[n] for n in joint_state.name]

        result = self.compute_fk_service(
            fk_link_names=['tool_frame'], robot_state=RobotState(joint_state=joint_state))

        if result.error_code.val != 1:
            return -1
        else:
            return result.pose_stamped[0]

    def FillAngularWaypoint(self, joint1, joint2, joint3, joint4, joint5, joint6, duration=0, radians=True):
        waypoint = Waypoint()
        angularWaypoint = AngularWaypoint()

        angularWaypoint.angles = [
            joint1, joint2, joint3, joint4, joint5, joint6]
        if radians:
            angularWaypoint.angles = [np.degrees(
                joint) % 360 for joint in angularWaypoint.angles]
        angularWaypoint.duration = duration
        waypoint.oneof_type_of_waypoint.angular_waypoint.append(
            angularWaypoint)

        return waypoint

    def FillCartesianWaypoint(self, new_x, new_y, new_z, new_theta_x, new_theta_y, new_theta_z, blending_radius):
        waypoint = Waypoint()
        cartesianWaypoint = CartesianWaypoint()

        cartesianWaypoint.pose.x = new_x
        cartesianWaypoint.pose.y = new_y
        cartesianWaypoint.pose.z = new_z
        cartesianWaypoint.pose.theta_x = new_theta_x
        cartesianWaypoint.pose.theta_y = new_theta_y
        cartesianWaypoint.pose.theta_z = new_theta_z
        cartesianWaypoint.reference_frame = CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_BASE
        cartesianWaypoint.blending_radius = blending_radius
        waypoint.oneof_type_of_waypoint.cartesian_waypoint.append(
            cartesianWaypoint)

        return waypoint

    def execute_action(self, req, error_message="Failed to call Service"):
        try:
            self.execute_action_service(req)
        except rospy.ServiceException:
            rospy.logerr(error_message)
            return False
        else:
            return self.wait_for_action_end_or_abort()

    def wait_for_action_end_or_abort(self, timeout=10):
        start_time = time.time()
        while not rospy.is_shutdown():
            if (self.last_action_notif_type == ActionEvent.ACTION_END):
                rospy.loginfo("Received ACTION_END notification")
                return True
            elif (self.last_action_notif_type == ActionEvent.ACTION_ABORT):
                rospy.loginfo("Received ACTION_ABORT notification")
                return False
            elif time.time()-start_time > timeout:
                return False
            else:
                rospy.sleep(0.01)

    def subscribe_to_a_robot_notification(self):
        # Activate the publishing of the ActionNotification
        req = OnNotificationActionTopicRequest()
        rospy.loginfo("Activating the action notifications...")
        try:
            self.activate_publishing_of_action_notification(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to call OnNotificationActionTopic")
            return False
        else:
            rospy.loginfo("Successfully activated the Action Notifications!")
        rospy.sleep(1.0)
        return True

    def clear_faults(self):
        try:
            self.clear_faults_service()
        except rospy.ServiceException:
            rospy.logerr("Failed to call ClearFaults")
            return False
        else:
            rospy.loginfo("Cleared the faults successfully")
            rospy.sleep(2.5)
            return True

    def home_arm(self):
        # The Home Action is used to home the robot. It cannot be deleted and is always ID #2:
        self.last_action_notif_type = None
        req = ReadActionRequest()
        req.input.identifier = self.HOME_ACTION_IDENTIFIER
        try:
            res = self.read_action(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to call ReadAction")
            return False
        # Execute the HOME action if we could read it
        else:
            # What we just read is the input of the ExecuteAction service
            req = ExecuteActionRequest()
            req.input = res.output
            rospy.loginfo("Sending the robot home...")
            return self.execute_action(req, error_message="Failed to call ExecuteAction")

    def set_cartesian_reference_frame(self, mode=CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_MIXED):
        """
            CARTESIAN_REFERENCE_FRAME_UNSPECIFIED (0):   Unspecified Cartesian reference frame

            CARTESIAN_REFERENCE_FRAME_MIXED (1):   Mixed reference frame where translation reference = base and  orientation reference = tool

            CARTESIAN_REFERENCE_FRAME_TOOL (2):   Tool reference frame where translation reference = tool and orientation reference = tool

            CARTESIAN_REFERENCE_FRAME_BASE (3):   Base reference frame where the translation reference = base and orientation reference = base
        """
        self.last_action_notif_type = None
        # Prepare the request with the frame we want to set
        req = SetCartesianReferenceFrameRequest()
        req.input.reference_frame = mode

        # Call the service
        try:
            self.set_cartesian_reference_frame_service(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to call SetCartesianReferenceFrame")
            return False
        else:
            rospy.loginfo("Set the cartesian reference frame successfully")
        # Wait a bit
        rospy.sleep(0.25)
        return True

    def get_eef_pose(self):
        feedback = rospy.wait_for_message(
            "/" + self.robot_name + "/base_feedback", BaseCyclic_Feedback)

        return feedback.base.tool_pose_x, \
            feedback.base.tool_pose_y, \
            feedback.base.tool_pose_z, \
            feedback.base.tool_pose_theta_x, \
            feedback.base.tool_pose_theta_y, \
            feedback.base.tool_pose_theta_z

    def get_cartesian_pose(self):
        feedback = rospy.wait_for_message(
            "/" + self.robot_name + "/base_feedback", BaseCyclic_Feedback)

        return feedback.base.commanded_tool_pose_x, \
            feedback.base.commanded_tool_pose_y, \
            feedback.base.commanded_tool_pose_z, \
            feedback.base.commanded_tool_pose_theta_x, \
            feedback.base.commanded_tool_pose_theta_y, \
            feedback.base.commanded_tool_pose_theta_z

    def get_joint_pose(self):
        """
        Returns current joints as a list in order of joints
        """
        joint_states = rospy.wait_for_message(
            f"{self.robot_name}/joint_states", JointState)

        return list(joint_states.position[:self.degrees_of_freedom])

    def send_cartesian_pose(self, pose: Pose, relative=False, duration=0, optimal_blending=True):
        self.last_action_notif_type = None
        x, y, z, theta_x, theta_y, theta_z = pose
        if relative:
            curr_pose = self.get_cartesian_pose()
            x += curr_pose[0]
            y += curr_pose[1]
            z += curr_pose[2]
            theta_x += curr_pose[3]
            theta_y += curr_pose[4]
            theta_z += curr_pose[5]

        # Possible to execute waypointList via execute_action service or use execute_waypoint_trajectory service directly
        req = ExecuteActionRequest()
        trajectory = WaypointList()
        trajectory.waypoints.append(
            self.FillCartesianWaypoint(
                x,
                y,
                z,
                theta_x,
                theta_y,
                theta_z,
                0)
        )

        trajectory.duration = duration
        trajectory.use_optimal_blending = optimal_blending

        req.input.oneof_action_parameters.execute_waypoint_list.append(
            trajectory)

        # Call the service
        rospy.loginfo("Sending the robot to the cartesian pose...")
        return self.execute_action(req, error_message="Failed to call ExecuteWaypointTrajectory")

    def send_joint_pose(self, joints: JointPose, relative=False, duration=0, optimal_blending=False):
        assert len(
            joints) == self.degrees_of_freedom, "Number of angles given does not match degrees of freedom"

        self.last_action_notif_type = None

        joint1, joint2, joint3, joint4, joint5, joint6 = joints
        if relative:
            curr_pose = self.get_joint_pose()
            joint1 += curr_pose[0]
            joint2 += curr_pose[1]
            joint3 += curr_pose[2]
            joint4 += curr_pose[3]
            joint5 += curr_pose[4]
            joint6 += curr_pose[5]

        req = ExecuteActionRequest()

        trajectory = WaypointList()
        trajectory.waypoints.append(
            self.FillAngularWaypoint(
                joint1,
                joint2,
                joint3,
                joint4,
                joint5,
                joint6,
                duration=duration
            )
        )

        trajectory.duration = duration
        trajectory.use_optimal_blending = optimal_blending

        try:
            res = self.validate_waypoint_list(trajectory)
        except rospy.ServiceException:
            rospy.logerr("Failed to call ValidateWaypointList")
            return False

        error_number = len(
            res.output.trajectory_error_report.trajectory_error_elements)
        MAX_ANGULAR_DURATION = 30

        while (error_number >= 1 and duration != MAX_ANGULAR_DURATION):
            duration += 1
            trajectory.waypoints[0].oneof_type_of_waypoint.angular_waypoint[0].duration = duration

            try:
                res = self.validate_waypoint_list(trajectory)
            except rospy.ServiceException:
                rospy.logerr("Failed to call ValidateWaypointList")
                return False

            error_number = len(
                res.output.trajectory_error_report.trajectory_error_elements)

        if (duration == MAX_ANGULAR_DURATION):
            # It should be possible to reach position within 30s
            # WaypointList is invalid (other error than angularWaypoint duration)

            rospy.loginfo("WaypointList is invalid")
            return False

        req.input.oneof_action_parameters.execute_waypoint_list.append(
            trajectory)

        # Send the angles
        rospy.loginfo("Sending the robot to the joint pose...")
        return self.execute_action(req, error_message="Failed to call ExecuteWaypointjectory")

    def send_gripper_command(self, value):
        # Initialize the request
        # Close the gripper
        req = SendGripperCommandRequest()
        finger = Finger()
        finger.finger_identifier = 0
        finger.value = value
        req.input.gripper.finger.append(finger)
        req.input.mode = GripperMode.GRIPPER_POSITION

        rospy.loginfo("Sending the gripper command...")

        # Call the service
        try:
            self.send_gripper_command_service(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to call SendGripperCommand")
            return False
        else:
            return True

    def get_gripper_value(self):
        return rospy.wait_for_message(f"{self.robot_name}/base_feedback/joint_state", JointState).position[self.degrees_of_freedom]

    def send_relative_gripper_command(self, delta):
        # Update stored gripper value instead of reading from ROS each time
        self.gripper_value = getattr(
            self, "gripper_value", self.get_gripper_value())

        new_value = max(0.0, min(1.0, self.gripper_value + delta))

        req = SendGripperCommandRequest()
        finger = Finger()
        finger.finger_identifier = 0
        finger.value = new_value
        req.input.gripper.finger.append(finger)
        req.input.mode = GripperMode.GRIPPER_POSITION

        rospy.loginfo(f"Sending gripper command: {new_value}")

        try:
            self.send_gripper_command_service(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to call SendGripperCommand")
            return False
        else:
            base_duration = 1.5
            scaled_sleep = base_duration * abs(delta)
            rospy.sleep(scaled_sleep)

        self.gripper_value = new_value
        return True

    def open_gripper(self):
        resp = self.send_gripper_command(0)
        rospy.sleep(1.5)
        return resp

    def close_gripper(self):
        resp = self.send_gripper_command(1)
        rospy.sleep(1.5)
        return resp

    def send_cartesian_trajectory(self, poses: List[Pose], use_optimal_blending=True):
        self.last_action_notif_type = None

        req = ExecuteActionRequest()
        trajectory = WaypointList()

        for pose in poses:
            x, y, z, theta_x, theta_y, theta_z = pose
            trajectory.waypoints.append(
                self.FillCartesianWaypoint(
                    x,
                    y,
                    z,
                    theta_x,
                    theta_y,
                    theta_z,
                    0)
            )
        trajectory.use_optimal_blending = use_optimal_blending
        req.input.oneof_action_parameters.execute_waypoint_list.append(
            trajectory)

        # Call the service
        rospy.loginfo("Sending robot through a cartesian trajectory...")
        return self.execute_action(req, error_message="Failed to call ExecuteWaypointjectory")

    def send_joint_trajectory(self, joint_poses: List[JointPose], use_optimal_blending=True):
        self.last_action_notif_type = None

        req = ExecuteActionRequest()
        trajectory = WaypointList()
        durations = [0] * len(joint_poses)

        for i, joints in enumerate(joint_poses):
            joint1, joint2, joint3, joint4, joint5, joint6 = joints
            trajectory.waypoints.append(
                self.FillAngularWaypoint(
                    joint1,
                    joint2,
                    joint3,
                    joint4,
                    joint5,
                    joint6,
                    duration=durations[i]
                )
            )
        trajectory.use_optimal_blending = use_optimal_blending
        req.input.oneof_action_parameters.execute_waypoint_list.append(
            trajectory)

        try:
            rospy.loginfo("Validating joint poses...")
            res = self.validate_waypoint_list(trajectory)
        except rospy.ServiceException:
            rospy.logerr("Failed to call ValidateWaypointList")
            return False

        error_number = len(
            res.output.trajectory_error_report.trajectory_error_elements)
        MAX_ANGULAR_DURATION = 1000

        # increment durations of each waypoint in trajectory until we get a valid trajectory
        while (error_number >= 1 and sum(durations) <= MAX_ANGULAR_DURATION):
            durations = [d+0.1 for d in durations]
            for i, waypoint in enumerate(trajectory.waypoints):
                waypoint.oneof_type_of_waypoint.angular_waypoint[0].duration = durations[i]

            try:
                res = self.validate_waypoint_list(trajectory)
            except rospy.ServiceException:
                rospy.logerr("Failed to call ValidateWaypointList")
                return False

            error_number = len(
                res.output.trajectory_error_report.trajectory_error_elements)

        if (sum(durations) > MAX_ANGULAR_DURATION):
            # It should be possible to reach position within the max angular duration
            # WaypointList is invalid (other error than angularWaypoint duration)
            rospy.loginfo("WaypointList is invalid")
            return False
    

        req.input.oneof_action_parameters.execute_waypoint_list.append(
            trajectory)

        # Send the angles
        rospy.loginfo("Sending the robot to the joint pose...")
        return self.execute_action(req, error_message="Failed to call ExecuteWaypointjectory")

    def joint_velocity_command(self, values, duration, duration_timeout=None, collision_check=False):
        """
        Sends velocity commands to the joints for the specified duration. 
        Returns a 1 on completion. 

        --------------
        values: A list or numpy array of joint velocity. 
        If it is a list or dictionary, it is assumed to be ordered 1-7.

        duration: The length of time the robot will execute the velocity command for.

        duration_timeout: if None, the function will return after duration. Else,
        the function will return after duration timeout even though the arm
        will move for time=duration. Currently this functionallity uses
        python threading so use with caution. 

        collision_check: if True, will calculate the arms future position and return -1 if
        it is in collision.

        TODO: -add dictionary functionality
              -add collision check
        """

        def velocity_command(values, duration):
            joint_vel_publisher = rospy.Publisher(
                f"/{self.robot_name}/in/joint_velocity", Base_JointSpeeds, queue_size=1, latch=True)
            joint_speeds = Base_JointSpeeds()
            joint_speeds.duration = 0
            for i in range(len(values)):
                speed = JointSpeed()
                speed.joint_identifier = i
                speed.value = values[i]
                joint_speeds.joint_speeds.append(speed)

            joint_vel_publisher.publish(joint_speeds)
            rospy.sleep(duration)
            self.stop()
            return 1

        if duration_timeout is not None:
            move_thread = threading.Thread(
                target=velocity_command, args=(values, duration))
            move_thread.start()
            rospy.sleep(duration_timeout)
            return 1
        else:
            return velocity_command(values, duration)

    def cartesian_velocity_command(self, values, duration, duration_timeout=None, collision_check=False, radians=False):
        """
        Sends a carteian velocity command for a specified duration. 
        Returns 1 on completion.

        ---------
        values: list or np array of the form: [x,y,z, x-twist, y-twist, z-twist, w-twist]

        duration: The length of time the robot will execute the velocity command for.

        duration_timeout: if None, the function will return after duration. Else,
        the function will return after duration timeout even though the arm
        will move for time=duration. Currently this functionionallity uses
        python threading so use with caution. 

        collision_check: if True, will calculate the arms furture position and return -1 if
        it is in collision.

        TODO: add collision check
        """

        def velocity_command(values, duration):
            cartesian_vel_publisher = rospy.Publisher(
                f"/{self.robot_name}/in/cartesian_velocity", TwistCommand, queue_size=1, latch=True)
            # stop_publisher = rospy.Publisher(        TODO: add collision check

            cmd = TwistCommand()

            if not radians:
                # convert degrees to radians
                linear = values[:3]
                angular = np.radians(values[3:6])
            else:
                linear = values[:3]
                angular = values[3:6]

            # Fill message
            cmd.twist.linear_x = linear[0]
            cmd.twist.linear_y = linear[1]
            cmd.twist.linear_z = linear[2]
            cmd.twist.angular_x = angular[0]
            cmd.twist.angular_y = angular[1]
            cmd.twist.angular_z = angular[2]

            cmd.reference_frame = 0
            cmd.duration = 1
            cartesian_vel_publisher.publish(cmd)
            rospy.sleep(duration)
            self.stop()

            return 1

        if duration_timeout is not None:
            move_thread = threading.Thread(
                target=velocity_command, args=(values, duration))
            move_thread.start()
            rospy.sleep(duration_timeout)
            return 1
        else:
            return velocity_command(values, duration)

    def main(self):
        # For testing purposes
        success = self.is_init_success

        if success:
            success &= self.clear_faults()

            # Activate the action notifications
            success &= self.subscribe_to_a_robot_notification()

            # Move the robot to the Home position with an Action
            # success &= self.send_joint_pose(joints=(0, 0, 0, 0, 0, 0))
            # self.close_gripper()
            self.home_arm()

            self.get_eef_pose()
            return
            # self.send_cartesian_pose(pose=(0.3075523376464844, 0.07795211672782898, 0.7412508726119995, -5.554068565368652, 7.461278438568115, 91.13240814208984))

            print(self.get_cartesian_pose())
            success &= self.send_cartesian_pose(pose=(0.4, 0, 0, 90, 0, 150))
            # success &= self.send_cartesian_pose(pose=(0.4, 0.1, 0.1, 90, 0, 150))
            # success &= self.send_cartesian_pose(pose=(0.4, 0.2, 0.2, 90, 0, 150))
            # success &= self.send_cartesian_pose(pose=(0.4, 0.3, 0.3, 90, 0, 150))
            # success &= self.send_cartesian_pose(pose=(0.4, 0.4, 0.4, 90, 0, 150))
            # success &= self.send_cartesian_pose(pose=(0.4, 0.5, 0.5, 90, 0, 150))
            success &= self.send_cartesian_pose(
                pose=(0.4, 0.6, 0.6, 90, 0, 150))
            # success &= self.send_cartesian_trajectory(poses=[
            #     # (0.4386, 0.1935, 0.4491, 90, 0, 150, 0),
            #     # (0.4386, 0.1935, 0.3491, 90, 0, 150, 0),
            #     # (0.4386, 0.2935, 0.3491, 90, 0, 150, 0),
            #     # (0.3686, 0.2335, 0.4091, 90, 0, 150, 0),
            #     # (0.4386, 0.2935, 0.3491, 90, 0, 150, 0),
            #     # (0.3686, 0.1535, 0.4991, 90, 0, 150, 0),
            #     # (0.4086, 0.2135, 0.4991, 90, 0, 150, 0),

            #     (0.4, 0, 0, 90, 0, 150),
            #     (0.4, 0.1, 0.1, 90, 0, 150),
            #     (0.4, 0.2, 0.2, 90, 0, 150),
            #     (0.4, 0.3, 0.3, 90, 0, 150),
            #     (0.4, 0.4, 0.4, 90, 0, 150),
            #     (0.4, 0.5, 0.5, 90, 0, 150),
            #     (0.4, 0.6, 0.6, 90, 0, 150),
            #     # (0.2386,  0.3935, 0.3491, 90, 0, 150, 0),
            #     # (0.1886,  0.0935, 0.6491, 90, 0, 150, 0),
            #     # (0.3386, -0.2065, 0.2491, 90, 0, 150, 0),
            #     # (0.6386,  0.2435, 0.1491, 90, 0, 150, 0),
            #     # (0.2386,  0.0435, 0.5991, 90, 0, 150, 0)
            # ])
            print("finished task1")

            return

            # success &= self.send_cartesian_pose(pose=(0.5712380409240723, 0.1556951403617859, 0.625953733921051, 67.15894317626953, 3.0942842960357666, 111.74005126953125))

            success &= self.open_gripper()
            # success &= self.send_joint_pose(joints=(0, 0, 0, 0, 0, 0))

            # Open gripper

            # Set the reference frame
            success &= self.set_cartesian_reference_frame(
                mode=CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_MIXED)

            # Sending cartesian pose
            success &= self.send_cartesian_pose(pose=(
                0.4584118723869324, 0.0920306071639061, -0.018976755440235138, 163.62733459472656, -5.928205490112305, 154.56910705566406))

            success &= self.close_gripper()
            rospy.sleep(1)

            success &= self.home_arm()
            success &= self.send_cartesian_pose(pose=(-0.10657188296318054, 0.4826011657714844,
                                                0.30631712079048157, -54.33583450317383, 160.96853637695312, 60.597450256347656))
            success &= self.open_gripper()

            # Sending arm vertically
            # success &= self.send_joint_pose(joints=(0, 0, 0, 0, 0, 0))

            # Close gripper to 50%
            # success &= self.send_gripper_command(0.5)

            # Move the robot to the Home position with an Action
            # success &= self.home_arm()

            # Cartesian trajectory
            # success &= self.send_cartesian_trajectory(poses=[
            #     (0.7,  0.0,   0.5,  90, 0, 90, 0),
            #     (0.7,  0.0,   0.33, 90, 0, 90, 0.1),
            #     (0.7,  0.48,  0.33, 90, 0, 90, 0.1),
            #     (0.61, 0.22,  0.4,  90, 0, 90, 0.1),
            #     (0.7,  0.48,  0.33, 90, 0, 90, 0.1),
            #     (0.63, -0.22, 0.45, 90, 0, 90, 0.1),
            #     (0.65, 0.05,  0.45, 90, 0, 90, 0)
            # ])

            # Move the robot to the Home position
            # success &= self.home_arm()

        if not success:
            rospy.logerr("The example encountered an error.")
