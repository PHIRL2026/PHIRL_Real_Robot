#!/bin/bash

roscore &
sleep 2

rosrun usb_cam usb_cam_node _video_device:=/dev/video2 &
rosrun joy joy_node &
roslaunch kortex_driver kortex_driver.launch &
wait
