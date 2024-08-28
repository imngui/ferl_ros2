from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ferl',  # Replace with the name of your package
            executable='user_input_node',  # The name of your executable script without the .py extension
            name='user_input_node',
            output='screen',
        )
    ])