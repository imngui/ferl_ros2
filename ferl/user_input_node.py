import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float64MultiArray
# from builtin_interfaces.msg import Duration

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import time
import numpy as np


class UserInputNode(Node):
    def __init__(self):
        super().__init__('user_input_node')

        # Publisher to send user input to 'output_topic'
        self.publisher = self.create_publisher(String, '/user_input', 10)

        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)

        self.interaction_start_time = None
        self.feature_trace_pub = self.create_publisher(Float64MultiArray, '/feature_trace', 10)
        self.feature_trace_msg = Float64MultiArray()
        self.feature_trace = []
        self.track_data = False
        self.track_data_sub = self.create_subscription(Bool, '/track_data', self.track_data_callback, 10)


    def joint_state_callback(self, msg):
        # self.feature_trace_msg.joint_names = msg.name

        if self.track_data:
            self.feature_trace.append(np.roll(msg.position))



    def track_data_callback(self, msg):
        self.track_data = msg.data

        if msg.data:
            self.feature_trace_msg.data = []

        
    def run(self):
        print("Move the robot to a position ")
        msg = String()
        while rclpy.ok():
            line = input("Press [1] to collect feature trace, [2] to stop collecting feature trace, and [3] to quit: ")
            msg.data = line
            self.publisher.publish(msg)
        rclpy.spin_once(self)


def main(args=None):
    rclpy.init(args=args)

    node = UserInputNode()

    try:
        node.run()
        # rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
