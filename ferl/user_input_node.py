import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool


class UserInputNode(Node):
    def __init__(self):
        super().__init__('user_input_node')

        # Publisher to send user input to 'output_topic'
        self.publisher = self.create_publisher(String, '/user_input', 10)
        
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
