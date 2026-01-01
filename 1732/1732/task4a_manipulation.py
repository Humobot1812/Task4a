#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
*****************************************************************************************
* eYRC Krishi CoBot 2025-26 | Team ID: 1732
* UR5 Pick & Place Extended Task (FINAL FIXED VERSION)
*****************************************************************************************
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from std_srvs.srv import SetBool
from std_msgs.msg import Float32



from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
import numpy as np
import time
from scipy.spatial.transform import Rotation
# from linkattacher_msgs.srv import AttachLink, DetachLink

BASE_FRAME = "base_link"
EEF_FRAME = "wrist_3_link"

# --- Motion Constants ---
LINEAR_KP = 1.5
MAX_LINEAR_VEL = 0.1 # m/s  30.0
POSITION_TOLERANCE = 0.08
ANGULAR_KP = 0.8
ORIENTATION_TOLERANCE = 0.1
WAIT_AT_WAYPOINT = 0.8

# --- Waypoint Orientations ---
PICK_ORIENTATION = Rotation.from_euler('x', 90, degrees=True)
DROP_ORIENTATION = Rotation.from_euler('y', 180, degrees=True)
FRUIT_PICK_ORIENTATION = Rotation.from_euler('x', 180, degrees=True) * Rotation.from_euler('z', 90, degrees=True)






INTERMEDIATE_P2_ORN = Rotation.from_quat(np.array([0.029, 0.997, 0.045, 0.033]))

TRASH_BIN_POS = np.array([-0.806, 0.010, 0.182])
INTERMEDIATE_1_POS = np.array([-0.159, 0.501, 0.600])
INTERMEDIATE_0_POS = np.array([0.150, 0, 0.600])

# Offsets
HOVER_OFFSET = np.array([0.0, 0.0, 0.2])
FERTILIZER_PICK_OFFSET = np.array([0.0, 0.0, 0.02])
FRUIT_PICK_OFFSET = np.array([0.0, 0.0, -0.04])

# Object names
FERTILIZER_MODEL = "fertiliser_can"
BAD_FRUIT_MODEL = "bad_fruit"
ROBOT_MODEL = "ur5"
ROBOT_GRIP_LINK = "wrist_3_link"
OBJECT_LINK = "body"








class UR5PickPlace(Node):
    def __init__(self):
        super().__init__("ur5_pick_place")
        self.get_logger().info("=== UR5 Pick & Place Node Started (Team 1732) ===")

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Publishers
        self.cmd_pub = self.create_publisher(TwistStamped, "/delta_twist_cmds", 10)

        self.status_pub = self.create_publisher(String, "/ur5_status", 10)

        # Services
        self.magnet_client = self.create_client(SetBool, '/magnet')
        while not self.magnet_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Magnet service not available, waiting...')


        
            
        self.net_force = 0.0

        self.force_sub = self.create_subscription(
            Float32,
            '/net_wrench',
            self.force_callback,
            10
        )


        # SUBSCRIBE TO DOCK_STATION

        self.timer = self.create_timer(0.1, self.control_loop)

        # Internal vars
        self.team_id = "1732"
        self.state = "WAIT_FOR_TFS"
        self.tf_positions = {}
        self.sequence = []
        self.current_index = 0
        self.reached_target = False
        self.last_reach_time = None
        # self.service_call_in_progress = False
        # self.service_future = None

   
    # ===================== TF HELPERS =====================
    def get_tf_pos(self, frame):
        try:
            tf = self.tf_buffer.lookup_transform(BASE_FRAME, frame, rclpy.time.Time())
            return np.array([
                tf.transform.translation.x,
                tf.transform.translation.y,
                tf.transform.translation.z
            ])
        except:
            return None
        
    def force_callback(self, msg):
        self.net_force = msg.data


    def stop(self):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        self.cmd_pub.publish(msg)


    def get_eef_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(BASE_FRAME, EEF_FRAME, rclpy.time.Time())
            pos = np.array([
                tf.transform.translation.x,
                tf.transform.translation.y,
                tf.transform.translation.z
            ])
            orn = Rotation.from_quat([
                tf.transform.rotation.x,
                tf.transform.rotation.y,
                tf.transform.rotation.z,
                tf.transform.rotation.w
            ])
            return pos, orn
        except:
            return None, None

    # ===================== MOTION CONTROL =====================
    def move_to_pose(self, target_pos, target_orn):
        pos, orn = self.get_eef_pose()
        if pos is None:
            return False

        # Linear
        error = target_pos - pos
        dist = np.linalg.norm(error)
        linear_cmd = LINEAR_KP * error if dist > POSITION_TOLERANCE else np.zeros(3)

        if np.linalg.norm(linear_cmd) > MAX_LINEAR_VEL:
            linear_cmd = linear_cmd / np.linalg.norm(linear_cmd) * MAX_LINEAR_VEL

        # Angular
        rot_err = target_orn * orn.inv()
        ang_err = rot_err.as_rotvec()
        ang_dist = np.linalg.norm(ang_err)
        angular_cmd = ANGULAR_KP * ang_err if ang_dist > ORIENTATION_TOLERANCE else np.zeros(3)

        # Reached
        if dist < POSITION_TOLERANCE and ang_dist < ORIENTATION_TOLERANCE:
            self.stop()
            return True
        
        # =====================================================
        #  FORCE SAFETY CHECK (CORRECT LOCATION)
        # =====================================================
        if self.net_force > 6.0:
            self.get_logger().warn(
                f"⚠ High force detected: {self.net_force:.2f}N → stopping motion"
            )
            self.stop()
            return True
        # =====================================================

        # Publish motion
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.twist.linear.x = linear_cmd[0]
        msg.twist.linear.y = linear_cmd[1]
        msg.twist.linear.z = linear_cmd[2]

        msg.twist.angular.x = angular_cmd[0]
        msg.twist.angular.y = angular_cmd[1]
        msg.twist.angular.z = angular_cmd[2]

        self.cmd_pub.publish(msg)

        return False

    # ===================== SERVICE CALL =====================
    
    def control_magnet(self, state: bool):
        request = SetBool.Request()
        request.data = state   # True = ON, False = OFF
        future = self.magnet_client.call_async(request)
        return future



    # ===================== CONTROL LOOP =====================   
    def control_loop(self):
        if self.state == "WAIT_FOR_TFS":
            self.collect_all_tfs()
            return

        if self.state == "MOVE_SEQUENCE":
            self.follow_sequence()
            return

        if self.state == "DONE":
            self.stop()
            return


    # ===================== COLLECT TFs =====================
    def collect_all_tfs(self):
        frames = [
            f"{self.team_id}_fertiliser_can",
            f"{self.team_id}_aruco_6",
            f"{self.team_id}_bad_fruit_0",
            f"{self.team_id}_bad_fruit_1",
            f"{self.team_id}_bad_fruit_2",
        ]

        for f in frames:
            pos = self.get_tf_pos(f)
            if pos is None:
                self.get_logger().info(f"Waiting for TF: {f}")
                return
            self.tf_positions[f] = pos

        # Fixed landmarks
        self.tf_positions["trash_bin"] = TRASH_BIN_POS
        self.tf_positions["intermediate_0"] = INTERMEDIATE_0_POS
        self.tf_positions["intermediate_1"] = INTERMEDIATE_1_POS

        self.get_logger().info("✅ All TF positions collected.")

        # ===================== BUILD SEQUENCE =====================
        self.sequence = []

        # PICK FERTILIZER
        pick = self.tf_positions[f"{self.team_id}_fertilizer_1"]
        self.sequence.append({
            'pos': pick + FERTILIZER_PICK_OFFSET,
            'orn': PICK_ORIENTATION,
            'label': "Pick Fertilizer",
            'action': "attach",
            'model': FERTILIZER_MODEL
        })

        # INTERMEDIATE-0
        p0 = self.tf_positions["intermediate_0"]
        self.sequence.append({
            'pos': p0,
            'orn': DROP_ORIENTATION,
            'label': "Intermediate 0",
            'action': "none"
        })

        # DROP FERTILIZER
        drop = self.tf_positions[f"{self.team_id}_aruco_6"]
        self.sequence.append({
            'pos': drop + HOVER_OFFSET,
            'orn': DROP_ORIENTATION,
            'label': "Drop Fertilizer",
            'action': "detach",
            'model': FERTILIZER_MODEL
        })

        # PUBLISH FERTILIZER DONE
        self.sequence.append({
            'pos': drop + HOVER_OFFSET,
            'orn': DROP_ORIENTATION,
            'label': "Notify Done",
            'action': "notify",
            'model': ""
        })

        # INTERMEDIATE-1
        p2 = self.tf_positions["intermediate_1"]
        self.sequence.append({
            'pos': p2,
            'orn': FRUIT_PICK_ORIENTATION,
            'label': "Intermediate-1",
            'action': "none"
        })

        # BAD FRUIT LOOP
        for i in range(3):
            fruit = self.tf_positions[f"{self.team_id}_bad_fruit_{i}"]

            # Pick fruit
            self.sequence.append({
                'pos': fruit + FRUIT_PICK_OFFSET,
                'orn': FRUIT_PICK_ORIENTATION,
                'label': f"Pick Fruit {i}",
                'action': "attach",
                'model': BAD_FRUIT_MODEL
            })

            # P2
            self.sequence.append({
                'pos': p2,
                'orn': INTERMEDIATE_P2_ORN,
                'label': "P2",
                'action': "none"
            })

            # DROP fruit
            trash = self.tf_positions["trash_bin"]
            self.sequence.append({
                'pos': trash + HOVER_OFFSET,
                'orn': DROP_ORIENTATION,
                'label': f"Drop Fruit {i}",
                'action': "detach",
                'model': BAD_FRUIT_MODEL
            })

            if i < 2:
                self.sequence.append({
                    'pos': p2,
                    'orn': INTERMEDIATE_P2_ORN,
                    'label': "Return P2",
                    'action': "none"
                })

        self.current_index = 0
        self.state = "MOVE_SEQUENCE"

    # ===================== FOLLOW SEQUENCE (WITH FIXED DETACH) =====================
    def follow_sequence(self):

        # --- WAIT FOR ONGOING SERVICE ---
        # if self.service_call_in_progress:
        #     if self.service_future.done():
        #         self.get_logger().info("✔ Gripper service completed")
        #         self.service_call_in_progress = False
        #         self.current_index += 1
        #         self.reached_target = False
        #     return

        # END OF SEQUENCE
        if self.current_index >= len(self.sequence):
            self.state = "DONE"
            return

        step = self.sequence[self.current_index]
        action = step.get("action", "none")

        # ===================== FORCE DETACH FIX =====================
        if action == "detach":
            pos, _ = self.get_eef_pose()
            if pos is not None:
                dist = np.linalg.norm(step['pos'] - pos)
                if dist < POSITION_TOLERANCE:
                    self.get_logger().info("📌 Position close enough → FORCING DETACH")
                    self.control_magnet(False)
                    self.net_force = 0.0

                    self.current_index += 1
                    self.reached_target = False
                    return

                    
        # =============================================================

        reached = self.move_to_pose(step['pos'], step['orn'])
        if not reached:
            return

        if not self.reached_target:
            self.reached_target = True
            self.last_reach_time = time.time()
            return

        if time.time() - self.last_reach_time < WAIT_AT_WAYPOINT:
            return

        # ATTACH
        if action == "attach":
            # Wait for contact force
            if self.net_force > 3.0:   # threshold (tune between 2.5–4.0)
                self.get_logger().info("🧲 Contact detected → Magnet ON")
                self.control_magnet(True)
                self.net_force = 0.0

                self.current_index += 1
                self.reached_target = False
            return


        # NOTIFY
        if action == "notify":
            msg = String()
            msg.data = "UR5_FERTILIZER_DONE"
            self.status_pub.publish(msg)
            self.get_logger().info("🌱 Published UR5_FERTILIZER_DONE")
            self.current_index += 1
            self.reached_target = False
            return

        # NORMAL waypoint
        self.current_index += 1
        self.reached_target = False


# ===================== MAIN =====================
def main(args=None):
    rclpy.init(args=args)
    node = UR5PickPlace()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.stop()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

