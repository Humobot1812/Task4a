#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Team ID: 1732

import sys
from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.node import Node
from sensor_msgs.msg import Image
from tf2_ros import TransformBroadcaster

SHOW_IMAGE = True
DISABLE_MULTITHREADING = False


class FruitsTF(Node):
    """
    ROS2 Node for fruit detection and TF publishing.
    """

    def __init__(self):
        super().__init__("fruits_tf")
        self.team_id = "1732"
        self.bridge = CvBridge()
        self.cv_image = None
        self.depth_image = None

        if DISABLE_MULTITHREADING:
            self.cb_group = MutuallyExclusiveCallbackGroup()
        else:
            self.cb_group = ReentrantCallbackGroup()

        self.create_subscription(
            Image,
            "/camera/task3a",
            self.colorimagecb,
            10,
            callback_group=self.cb_group,
        )
        self.create_subscription(
            Image,
            "/camera/depth/image_raw",
            self.depthimagecb,
            10,
            callback_group=self.cb_group,
        )
        self.tf_broadcaster = TransformBroadcaster(self)
        self.create_timer(0.2, self.process_image, callback_group=self.cb_group)

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(
            self.aruco_dict, self.aruco_params
        )

        if SHOW_IMAGE:
            cv2.namedWindow("fruits_tf_view", cv2.WINDOW_NORMAL)

        self.get_logger().info(f"FruitsTF node started - Team ID: {self.team_id}")

    def depthimagecb(self, data):
        try:
            self.original_depth_msg = data

            if data.encoding == "32FC1":
                depth_data = self.bridge.imgmsg_to_cv2(data, desired_encoding="32FC1")
                self.depth_image = depth_data
            elif data.encoding == "16UC1":
                depth_data = self.bridge.imgmsg_to_cv2(data, desired_encoding="16UC1")
                depth_meters = depth_data.astype(np.float32) / 1000.0
                self.depth_image = depth_meters
            else:
                depth_data = self.bridge.imgmsg_to_cv2(
                    data, desired_encoding="passthrough"
                )
                self.depth_image = depth_data.astype(np.float32)

            if self.depth_image is not None:
                self.depth_image[self.depth_image <= 0] = np.nan
                self.depth_image[np.isinf(self.depth_image)] = np.nan

        except Exception as e:
            self.get_logger().error(f"Depth conversion error: {e}")
            self.depth_image = None

    def colorimagecb(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"RGB conversion error: {e}")

    def detect_aruco_markers(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.aruco_detector.detectMarkers(gray)
        image_with_aruco = image.copy()

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(image_with_aruco, corners, ids)

        return image_with_aruco

    def process_aruco_markers(self, image, depth_image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.aruco_detector.detectMarkers(gray)

        if ids is None:
            return image

        camera_matrix = np.array(
            [
                [915.3003540039062, 0.0, 642.724365234375],
                [0.0, 914.0320434570312, 361.9780578613281],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        focalX = camera_matrix[0, 0]
        focalY = camera_matrix[1, 1]
        centerCamX = camera_matrix[0, 2]
        centerCamY = camera_matrix[1, 2]

        image_out = image.copy()

        for i, corner in enumerate(corners):
            pts = corner[0]
            cX = int(np.mean(pts[:, 0]))
            cY = int(np.mean(pts[:, 1]))
            marker_id = int(ids[i][0])

            distance = None
            if depth_image is not None:
                distance = self.get_accurate_depth(cX, cY, depth_image)

            if distance is None:
                continue

            optical_pos = self.pixel_to_3d(
                cX, cY, distance, focalX, focalY, centerCamX, centerCamY
            )
            base_pos = self.transform_optical_to_base_frame(optical_pos)

            if marker_id == 3:
                frame_name = f"{self.team_id}_fertiliser_can"
            elif marker_id == 6:
                frame_name = f"{self.team_id}_aruco_6"
            else:
                continue

            transform = self.create_transform_stamped("base_link", frame_name, base_pos)
            self.tf_broadcaster.sendTransform(transform)

            cv2.circle(image_out, (cX, cY), 6, (255, 0, 0), -1)
            cv2.putText(
                image_out,
                f"Aruco {marker_id}",
                (cX - 30, cY - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )

        return image_out

    def bad_fruit_detection(self, rgb_image):
        bad_fruits = []
        height, width = rgb_image.shape[:2]
        
        # Create ROI mask - only detect in lower-left quadrant
        roi_mask = np.zeros((height, width), dtype=np.uint8)
        half_height = height // 2
        half_width = width // 2
        roi_mask[half_height:, :half_width] = 255  # Lower-left quadrant only
        
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        
        # NEW APPROACH: Detect the GREEN ROUND CAP on top of GREY fruits ONLY
        
        # STEP 1: Detect GREEN caps (the round green circles on top)
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        # STEP 2: Detect GREY fruit body (low saturation = grey, NOT red/purple)
        lower_grey = np.array([0, 0, 30])
        upper_grey = np.array([180, 50, 150])  # Key: saturation < 50 = grey only
        mask_grey = cv2.inRange(hsv, lower_grey, upper_grey)
        
        # STEP 3: EXCLUDE red/purple fruits (high saturation colors)
        lower_red1 = np.array([0, 60, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 60, 50])
        upper_red2 = np.array([180, 255, 255])
        lower_purple = np.array([125, 60, 50])
        upper_purple = np.array([165, 255, 255])
        
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)
        mask_colored = cv2.bitwise_or(mask_red1, mask_red2)
        mask_colored = cv2.bitwise_or(mask_colored, mask_purple)
        
        # Dilate colored fruit mask to create exclusion zone
        mask_colored = cv2.dilate(mask_colored, None, iterations=5)
        
        # STEP 4: Apply ROI mask
        mask_green = cv2.bitwise_and(mask_green, roi_mask)
        mask_grey = cv2.bitwise_and(mask_grey, roi_mask)
        mask_colored = cv2.bitwise_and(mask_colored, roi_mask)
        
        # STEP 5: Clean up masks
        kernel = np.ones((3, 3), np.uint8)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel, iterations=2)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Debug visualization (uncomment to see masks)
        # cv2.imshow("1_green_caps", mask_green)
        # cv2.imshow("2_grey_body", mask_grey)
        # cv2.imshow("3_colored_fruits", mask_colored)

        # STEP 6: Find contours of GREEN CAPS
        contours, _ = cv2.findContours(
            mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Sort contours left-to-right
        if len(contours) > 0:
            contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

        fruit_id = 0

        # STEP 7: Process each GREEN CAP contour
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Lower threshold to catch smaller green caps
            if area > 200:  # Reduced from 300 to catch more
                # Calculate centroid of the GREEN CAP
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h
                    
                    # Calculate circularity - green caps should be round
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                    else:
                        circularity = 0
                    
                    # More permissive shape filtering
                    if 0.3 < aspect_ratio < 2.5 and circularity > 0.25:
                        
                        # SIMPLE CHECK: Is this green cap on a RED/PURPLE fruit?
                        # Check for colored pixels directly below the green cap
                        check_y_start = cY + 5  # Just below the cap
                        check_y_end = min(cY + 35, height)
                        check_x_start = max(cX - 15, 0)
                        check_x_end = min(cX + 15, width)
                        
                        on_colored_fruit = False
                        if check_y_end < height and check_x_end < width:
                            region_colored = mask_colored[check_y_start:check_y_end, check_x_start:check_x_end]
                            colored_pixel_count = cv2.countNonZero(region_colored)
                            # If significant red/purple pixels below, skip
                            if colored_pixel_count > 150:
                                on_colored_fruit = True
                        
                        # Skip if on colored fruit, otherwise accept
                        if not on_colored_fruit:
                            distance = None
                            if self.depth_image is not None:
                                distance = self.get_accurate_depth(cX, cY, self.depth_image)
                            
                            fruit_info = {
                                "center": (cX, cY),
                                "distance": distance,
                                "angle": 0,
                                "width": w,
                                "id": fruit_id,
                                "area": area,
                                "contour": contour,
                                "circularity": circularity,
                            }
                            bad_fruits.append(fruit_info)
                            fruit_id += 1
        
        return bad_fruits

    def get_accurate_depth(
        self, x: int, y: int, depth_image: np.ndarray
    ) -> Optional[float]:
        if depth_image is None:
            return None
        height, width = depth_image.shape
        if x < 0 or x >= width or y < 0 or y >= height:
            return None
        depth_samples = []
        valid_samples = 0
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                sample_x = x + dx
                sample_y = y + dy
                if 0 <= sample_x < width and 0 <= sample_y < height:
                    depth_val = depth_image[sample_y, sample_x]
                    if not np.isnan(depth_val) and 0.1 < depth_val < 3.0:
                        depth_samples.append(depth_val)
                        valid_samples += 1
        if valid_samples < 5:
            return None
        depth_samples = np.array(depth_samples)
        Q1 = np.percentile(depth_samples, 25)
        Q3 = np.percentile(depth_samples, 75)
        IQR = Q3 - Q1
        filtered_depths = depth_samples[
            (depth_samples >= (Q1 - 1.5 * IQR)) & (depth_samples <= (Q3 + 1.5 * IQR))
        ]
        if len(filtered_depths) == 0:
            return None
        return np.median(filtered_depths)

    def pixel_to_3d(self, u, v, depth, focalX, focalY, centerCamX, centerCamY):
        z = depth
        x = (u - centerCamX) * z / focalX
        y = (v - centerCamY) * z / focalY
        return (x, y, z)

    def transform_optical_to_base_frame(self, point_optical):
        x_opt, y_opt, z_opt = point_optical
        x_cam = z_opt
        y_cam = -x_opt
        z_cam = -y_opt
        pitch = -0.733
        cos_p = np.cos(pitch)
        sin_p = np.sin(pitch)
        x_base = (x_cam * cos_p - z_cam * sin_p) - 1.095239
        y_base = y_cam
        z_base = (x_cam * sin_p + z_cam * cos_p) + 1.10058
        return (x_base, y_base, z_base)

    def create_transform_stamped(self, parent_frame, child_frame, translation):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = parent_frame
        t.child_frame_id = child_frame
        t.transform.translation.x = float(translation[0])
        t.transform.translation.y = float(translation[1])
        t.transform.translation.z = float(translation[2])
        t.transform.rotation.w = 1.0
        return t

    def process_image(self):
        centerCamX = 642.724365234375
        centerCamY = 361.9780578613281
        focalX = 915.3003540039062
        focalY = 914.0320434570312

        if self.cv_image is None or self.depth_image is None:
            return

        try:
            # Process ArUco (full image)
            vis_image = self.process_aruco_markers(self.cv_image, self.depth_image)

            # Draw ROI boundaries for visualization
            height, width = vis_image.shape[:2]
            half_height = height // 2
            half_width = width // 2
            
            # Draw horizontal line
            cv2.line(vis_image, (0, half_height), (width, half_height), 
                    (0, 255, 255), 2)
            # Draw vertical line
            cv2.line(vis_image, (half_width, half_height), (half_width, height), 
                    (0, 255, 255), 2)
            # Draw detection zone rectangle
            cv2.rectangle(vis_image, (0, half_height), (half_width, height), 
                         (0, 255, 255), 3)
            cv2.putText(vis_image, "FRUIT DETECTION ZONE", 
                       (10, half_height + 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8, (0, 255, 255), 2)

            # Process Bad Fruits (only in lower-left quadrant)
            detections = self.bad_fruit_detection(self.cv_image)

            if not detections and SHOW_IMAGE:
                cv2.imshow("fruits_tf_view", vis_image)
                cv2.waitKey(1)
                return

            published_count = 0
            for fruit_info in detections:
                cX, cY = fruit_info["center"]
                distance = fruit_info["distance"]
                fruit_id = fruit_info["id"]
                
                # Visualization
                contour = fruit_info["contour"]
                x, y, w, h = cv2.boundingRect(contour)
                circularity = fruit_info.get("circularity", 0)
                
                # Draw the contour in cyan
                cv2.drawContours(vis_image, [contour], -1, (255, 255, 0), 2)
                # Draw bounding box
                cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Draw centroid
                cv2.circle(vis_image, (cX, cY), 5, (0, 0, 255), -1)
                # Show fruit info
                cv2.putText(
                    vis_image,
                    f"bad fruit {fruit_id} (c:{circularity:.2f})",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

                if distance is None:
                    continue

                # Compute Transform
                optical_pos = self.pixel_to_3d(
                    cX, cY, distance, focalX, focalY, centerCamX, centerCamY
                )
                base_link_pos = self.transform_optical_to_base_frame(optical_pos)

                # Publish TF
                frame_name = f"{self.team_id}_bad_fruit_{fruit_id}"
                transform = self.create_transform_stamped(
                    "base_link", frame_name, base_link_pos
                )
                self.tf_broadcaster.sendTransform(transform)

                self.get_logger().info(f"Published {frame_name} at distance {distance:.3f}m")
                published_count += 1

            if SHOW_IMAGE:
                cv2.imshow("fruits_tf_view", vis_image)
                cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Processing error: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = FruitsTF()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        if SHOW_IMAGE:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()