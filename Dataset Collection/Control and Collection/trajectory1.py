import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Imu
from rclpy.qos import QoSProfile, ReliabilityPolicy
import math
import csv

class PathControl(Node):
    def __init__(self):
        super().__init__('Path_Control')

        self.namespace = '/'
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        self.cmd_vel_pub = self.create_publisher(Twist, f'/{self.namespace}/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, f'/{self.namespace}/platform/odom/filtered', self.odom_callback, qos)
        self.scan_sub = self.create_subscription(LaserScan, f'/{self.namespace}/sensors/lidar2d_0/scan', self.scan_callback, qos)
        self.imu_sub = self.create_subscription(Imu, f'/{self.namespace}/sensors/imu_0/data_raw', self.imu_callback, qos)        

        self.csv_filename = "path11.csv"

        self.csv_file = open(self.csv_filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)

        header = [
            'timestamp',                    
            'task_type',                        
            'cmd_v', 'cmd_w',                  
            'odom_x', 'odom_y', 'odom_z',
            'odom_yaw',
            'odom_ori_x', 'odom_ori_y', 'odom_ori_z', 'odom_ori_w',    
            'odom_lin_x_v', 'odom_ang_z_w',
            'imu_lin_x', 'imu_lin_y', 'imu_lin_z',                  
            'imu_ang_x', 'imu_ang_y', 'imu_ang_z',
            'imu_ori_x', 'imu_ori_y', 'imu_ori_z', 'imu_ori_w',                
        ]
        self.csv_writer.writerow(header)
        self.get_logger().info(f"Begin: {self.csv_filename}")

        self.state = 'IDLE'     
        self.task_index = 0     
        self.wait_count = 0
        self.initialized = False  

        self.curr_x = 0.0
        self.curr_y = 0.0
        self.curr_z = 0.0
        self.curr_yaw = 0.0
        self.odom_quat = [0.0, 0.0, 0.0, 1.0]
        
        self.curr_odom_v = 0.0
        self.curr_odom_w = 0.0
        self.min_scan_dist = 99.0 

        self.imu_acc = [0.0, 0.0, 0.0]
        self.imu_gyro = [0.0, 0.0, 0.0]
        self.imu_quat = [0.0, 0.0, 0.0, 0.0]
        
        self.start_x = 0.0
        self.start_y = 0.0
        self.start_yaw = 0.0
        
        self.prev_yaw_error = None 

        self.speed_lib = {
            'G1': 0.6,
            'G2': 0.7,
            'G3': 4.0,
            'G4': 0.8,
            'G5': 0.9,
            'G6': 1.5,
            'G7': 2.5,
            'G8': 3.0,
            'G9': 2.0,
            'G10': 1.0,
            'G11': 0.4,
            'G12': 1.1,
            'G13': 3.5,

            'T1': 0.7,
            'T2': 0.6,
            'T3': 0.8,
            'T4': 1.5,
            'T5': 0.7,
            'T6': 1.8,
            'T7': 1.1,
            'T8': 1.2,
            'T9': 1.3,
            'T10': 0.8,
            'T11': 0.9,
            'T12': 1.4,
        }

        self.mission_tasks = [
            {'type': 'OBSTACLE', 'val': 0.5, 'spd': 'G1'},
            {'type': 'TURN', 'val': 90.0, 'spd': 'T1'},
            {'type': 'DIST', 'val': 3.5, 'spd': 'G2'},
            {'type': 'TURN', 'val': -90.0, 'spd': 'T2'},
            {'type': 'OBSTACLE', 'val': 1.0, 'spd': 'G3'},
            {'type': 'TURN', 'val': 90.0, 'spd': 'T3'},
            {'type': 'DIST', 'val': 3.0, 'spd': 'G4'},
            {'type': 'TURN', 'val': -90.0, 'spd': 'T4'},
            {'type': 'DIST', 'val': 3.0, 'spd': 'G5'},
            {'type': 'TURN', 'val': -90.0, 'spd': 'T5'},
            {'type': 'OBSTACLE', 'val': 1.0, 'spd': 'G6'},
            {'type': 'TURN', 'val': 90.0, 'spd': 'T6'},
            {'type': 'OBSTACLE', 'val': 1.5, 'spd': 'G7'},
            {'type': 'TURN', 'val': 90.0, 'spd': 'T7'},
            {'type': 'OBSTACLE', 'val': 1.0, 'spd': 'G8'},
            {'type': 'TURN', 'val': 90.0, 'spd': 'T8'},
            {'type': 'OBSTACLE', 'val': 0.5, 'spd': 'G9'},
            {'type': 'TURN', 'val': -90.0, 'spd': 'T9'},
            {'type': 'OBSTACLE', 'val': 0.5, 'spd': 'G10'},
            {'type': 'TURN', 'val': -90.0, 'spd': 'T10'},
            {'type': 'OBSTACLE', 'val': 0.5, 'spd': 'G11'},
            {'type': 'TURN', 'val': 90.0, 'spd': 'T11'},
            {'type': 'OBSTACLE', 'val': 0.5, 'spd': 'G12'},
            {'type': 'TURN', 'val': 90.0, 'spd': 'T12'},
            {'type': 'DIST', 'val': 5.0, 'spd': 'G13'},
        ]

        self.timer = self.create_timer(0.02, self.control_loop)
        self.get_logger().info("PathControl node started, waiting for Odom data initialization...")

    def reset_reference(self):
        self.start_x = self.curr_x
        self.start_y = self.curr_y
        self.start_yaw = self.curr_yaw
    
    def euler_from_quaternion(self, q):
        x, y, z, w = q.x, q.y, q.z, q.w
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        return math.atan2(t3, t4)

    def odom_callback(self, msg):
        self.curr_x = msg.pose.pose.position.x
        self.curr_y = msg.pose.pose.position.y
        self.curr_z = msg.pose.pose.position.z
        self.odom_quat = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, 
                          msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        self.curr_yaw = self.euler_from_quaternion(msg.pose.pose.orientation)
        self.curr_odom_v = msg.twist.twist.linear.x
        self.curr_odom_w = msg.twist.twist.angular.z
        
        if not self.initialized:
            self.reset_reference()
            self.initialized = True

    def normalize_angle(self, angle):
        while angle > math.pi: angle -= 2.0 * math.pi
        while angle < -math.pi: angle += 2.0 * math.pi
        return angle
    
    def scan_callback(self, msg):
        if not msg.ranges: return
        mid_idx = len(msg.ranges) // 2
        window_size = 10 
        valid_ranges = []
        raw_window = []
        
        for i in range(mid_idx - window_size, mid_idx + window_size):
            if 0 <= i < len(msg.ranges):
                val = msg.ranges[i]
                raw_window.append(val)
                if val > 0.1 and not math.isinf(val):
                    valid_ranges.append(val)
        
        self.min_scan_dist = min(valid_ranges) if valid_ranges else 99.0
    
    def imu_callback(self, msg):
        self.imu_acc = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
        self.imu_gyro = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
        self.imu_quat = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]

    def control_loop(self):
        cmd_v, cmd_w = 0.0, 0.0

        if self.task_index < len(self.mission_tasks):
            current_task_type = self.mission_tasks[self.task_index]['type']
        else:
            current_task_type = 'DONE'
        
        if self.state == 'IDLE':
            self.publish_vel(0.0, 0.0) 
            
            if not self.initialized: return 

            if self.task_index >= len(self.mission_tasks):
                self.state = 'DONE'
                self.get_logger().info("All tasks completed -> Entering DONE state")
                return

            wait_target = 150 if self.task_index == 0 else 25

            if self.wait_count < wait_target:
                if self.wait_count == 0 and self.task_index == 0:
                    self.get_logger().info("Waiting for Gazebo physics engine stabilization (3 seconds)...")
                self.wait_count += 1
                self.save_data(0.0, 0.0, 'IDLE')
                return
            
            self.wait_count = 0
            self.reset_reference() 
            self.prev_yaw_error = None 
            self.state = 'RUNNING'
            self.save_data(0.0, 0.0, current_task_type)

        elif self.state == 'RUNNING':
            cmd_v, cmd_w = self.execute_mission_logic()
            self.publish_vel(cmd_v, cmd_w)
            self.save_data(cmd_v, cmd_w, current_task_type)

        elif self.state == 'DONE':
            self.publish_vel(0.0, 0.0)
            self.save_data(0.0, 0.0, 'DONE')
            self.get_logger().info("Path finished!")
            self.csv_file.close()
            self.timer.cancel()
            raise KeyboardInterrupt 

    def execute_mission_logic(self):
        current_task = self.mission_tasks[self.task_index]
        task_type, target_val, speed_key = current_task['type'], current_task['val'], current_task['spd']
        
        target_max_spd = self.speed_lib.get(speed_key, 0.2)
        decel_dist = 2.0    
        min_speed = 0.1     

        cmd_v, cmd_w = 0.0, 0.0
        is_task_finished = False
        
        if task_type == 'OBSTACLE':
            dx, dy = self.curr_x - self.start_x, self.curr_y - self.start_y
            actual_dist_moved = math.sqrt(dx**2 + dy**2)
            dist_remaining = self.min_scan_dist - target_val
            
            if dist_remaining <= 0:
                is_task_finished = True
                self.get_logger().info(f"Obstacle avoidance finished! Actual distance moved: {actual_dist_moved:.2f}m")
            else:
                if dist_remaining < decel_dist:
                    slow_spd = dist_remaining * 0.5
                    cmd_v = max(min_speed, min(slow_spd, target_max_spd))
                else:
                    cmd_v = target_max_spd

        elif task_type == 'DIST':
            dx, dy = self.curr_x - self.start_x, self.curr_y - self.start_y
            distance_moved = math.sqrt(dx**2 + dy**2)
            dist_remaining = target_val - distance_moved

            if dist_remaining <= 0:
                is_task_finished = True
                self.get_logger().info(f"Distance task finished! Target: {target_val}m, Actual: {distance_moved:.2f}m")
            else:
                if dist_remaining < decel_dist:
                    slow_spd = dist_remaining * 0.5
                    cmd_v = max(min_speed, min(slow_spd, target_max_spd))
                else:
                    cmd_v = target_max_spd

        elif task_type == 'TURN':
            target_rad = math.radians(target_val)
            target_yaw = self.normalize_angle(self.start_yaw + target_rad)
            yaw_error = self.normalize_angle(target_yaw - self.curr_yaw)
            
            tolerance = 0.005 
            min_turn_spd = 0.15  
            
            turn_kp = 1.2  
            turn_kd = 0.6  
            
            if abs(yaw_error) < tolerance:
                is_task_finished = True
                self.get_logger().info(f"Turn finished! Final error: {math.degrees(yaw_error):.3f} degrees")
            else:
                if self.prev_yaw_error is None:
                    self.prev_yaw_error = yaw_error

                derivative = yaw_error - self.prev_yaw_error
                
                cmd_w_raw = (turn_kp * yaw_error) + (turn_kd * derivative)
                
                self.prev_yaw_error = yaw_error
                
                abs_cmd_w = abs(cmd_w_raw)
                clamped_abs_w = max(min_turn_spd, min(abs_cmd_w, target_max_spd))
                cmd_w = clamped_abs_w if cmd_w_raw > 0 else -clamped_abs_w

        if is_task_finished:
            self.task_index += 1       
            self.state = 'IDLE'        
            return 0.0, 0.0
        else:
            return cmd_v, cmd_w
    
    def save_data(self, cmd_v, cmd_w, task_type):
        if not self.initialized: return

        timestamp = self.get_clock().now().nanoseconds

        row = [
            timestamp, task_type, cmd_v, cmd_w,
            self.curr_x, self.curr_y, self.curr_z,
            self.curr_yaw,
            self.odom_quat[0], self.odom_quat[1], self.odom_quat[2], self.odom_quat[3],
            self.curr_odom_v, self.curr_odom_w,
            self.imu_acc[0], self.imu_acc[1], self.imu_acc[2],
            self.imu_gyro[0], self.imu_gyro[1], self.imu_gyro[2],
            self.imu_quat[0], self.imu_quat[1], self.imu_quat[2], self.imu_quat[3],  
        ]
        self.csv_writer.writerow(row)
        self.csv_file.flush() 

    def publish_vel(self, v, w):
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.cmd_vel_pub.publish(msg)

    def destroy_node(self):
        if hasattr(self, 'csv_file') and not self.csv_file.closed:
            self.csv_file.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = PathControl()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()