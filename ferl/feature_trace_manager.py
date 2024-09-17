import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


from sensor_msgs.msg import JointState
import numpy as np
import time


class FeatureTraceManager(Node):
    def __init__(self):
        super().__init__('feature_trace_manager')

        self.user_input_sub = self.create_subscription(String, '/user_input', self.user_input_callback, 10)

        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)

        self.interaction_start_time = None
        self.feature_trace_pub = self.create_publisher(JointTrajectory, '/feature_trace', 10)
        self.feature_trace_msg = JointTrajectory()
        self.track_data = False


    def joint_state_callback(self, msg):
        self.feature_trace_msg.joint_names = msg.name

        if self.track_data:
            point = JointTrajectoryPoint()
            point.positions = np.roll(msg.position, 1).tolist()
            duration = time.time() - self.interaction_start_time
            point.time_from_start = rclpy.time.Duration(seconds=duration).to_msg()
            self.feature_trace_msg.points.append(point)


    def user_input_callback(self, msg):
        if msg.data == "1":
            self.track_data = True
            self.feature_trace_msg.points = []
            self.interaction_start_time = time.time()
        if msg.data == "2":
            self.track_data = False
            self.feature_trace_pub.publish(self.feature_trace_msg)
            self.feature_trace_msg.points= []
            self.interaction_start_time = None


def main(args=None):
    rclpy.init(args=args)

    node = FeatureTraceManager()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
