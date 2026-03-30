import cv2 
from BrushPerception import BrushPerception
import rospy

def main():
    rospy.init_node("brush_perception_debug")

    perception = BrushPerception()

    rate = rospy.Rate(10)  # 10 Hz
    while not rospy.is_shutdown():
        perception.show_debug_view()
        rate.sleep()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()