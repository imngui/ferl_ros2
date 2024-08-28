import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class UserInputNode(Node):
    def __init__(self):
        super().__init__('user_input_node')

        # Subscriber to listen to incoming messages on 'input_topic'
        self.subscription = self.create_subscription(
            String,
            '/req_user_input',
            self.listener_callback,
            10
        )

        # Publisher to send user input to 'output_topic'
        self.publisher = self.create_publisher(String, '/user_input', 10)

        self.get_logger().info("String Relay Node initialized. Waiting for messages...")

    def listener_callback(self, msg):
        # Display the received message
        self.get_logger().info(f"{msg.data}")

        # Wait for user input after the message is displayed
        user_input = input("Enter your response: ")

        # Publish the user's input to the 'output_topic'
        output_msg = String()
        output_msg.data = user_input
        self.publisher.publish(output_msg)

        self.get_logger().info(f"Published user input: {user_input}")

def main(args=None):
    rclpy.init(args=args)

    node = UserInputNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
