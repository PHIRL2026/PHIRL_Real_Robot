import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import threading
import cv2 
import numpy as np

class BrushPerception:
    def __init__(self, topic="/usb_cam/image_raw"):
        self.bridge = CvBridge()
        self.latest_image = None
        self.lock = threading.Lock()  # for thread-safe access

        # bounding box for knot detection
        self.bbox = (155, 169, 140, 86)

        self.sub = rospy.Subscriber(
            topic,
            Image,
            self.image_callback,
            queue_size=1
        )

        rospy.loginfo(f"Subscribed to camera topic: {topic}")

    def image_callback(self, msg):
        """ROS callback that updates the latest image"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            with self.lock:
                self.latest_image = cv_image

        except Exception as e:
            rospy.logerr(f"Failed to convert image: {e}")

    def get_latest_image(self):
        with self.lock:
            if self.latest_image is not None:
                return self.latest_image.copy()
            else:
                return None

    def count_knot_pixels(self):
        """
        Detects and counts knots in the latest image.

        Assumes:
        - Knots are darker than the plushie.
        - Plushie is mostly in a plain background color.
        - Knots are small and roughly circular.

        Returns:
            int: Number of pixels that are knots.
        """
    
        img = self.get_latest_image()
        if img is None:
            return 0

        # Bounding box of plushie's belly compatible with the physical setup
        x, y, w, h = self.bbox

        # Crop ROI
        roi = img[y:y+h, x:x+w]

        # Convert to HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Split channels
        h, s, v = cv2.split(hsv)

        # Threshold saturation (tune this)
        _, sat_mask = cv2.threshold(s, 40, 255, cv2.THRESH_BINARY)

        # Count colorful pixels
        colorful_pixel_count = cv2.countNonZero(sat_mask)

        # print("Colorful pixels:", colorful_pixel_count)

        # Decide
        return colorful_pixel_count

    def detect_pompom_locations(self):
        img = self.get_latest_image()
        if img is None:
            return []

        x, y, w, h = self.bbox
        roi = img[y:y+h, x:x+w]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1]
        _, sat_mask = cv2.threshold(s, 40, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(sat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 50:
                continue

            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue

            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])

            mask = np.zeros(sat_mask.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mean_hue = cv2.mean(hsv[:, :, 0], mask=mask)[0]

            if mean_hue <= 10 or mean_hue >= 160:
                color = "red"
            elif mean_hue <= 25:
                color = "orange"
            elif mean_hue <= 35:
                color = "yellow"
            elif mean_hue <= 85:
                color = "green"
            elif mean_hue <= 125:
                color = "blue"
            else:
                color = "purple"

            results.append({
                'x': cx + x,
                'y': cy + y,
                'color': color,
                'area': int(area)
            })

        results.sort(key=lambda d: d['area'], reverse=True)
        return results

    def show_debug_view(self):
        """
        Displays the ROS image with the plushie bounding box overlaid.
        """
        img = self.get_latest_image()
        if img is None:
            return

        x, y, w, h = self.bbox

        debug_img = img.copy()
        cv2.rectangle(
            debug_img,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        knot_pixels = self.count_knot_pixels()
        cv2.putText(
            debug_img,
            f"Knot pixels: {knot_pixels}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow("Plushie Bounding Box (ROS)", debug_img)
        cv2.waitKey(1)
