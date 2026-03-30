#!/usr/bin/env python3

from ControllerNode import ControllerNode
import rospy

def main():
    rospy.init_node('controller_testing', anonymous=True)

    node = ControllerNode()
    rate = rospy.Rate(50)

    while not rospy.is_shutdown():
        node.step()
        rate.sleep()
        

if __name__ == "__main__":
    main()

        