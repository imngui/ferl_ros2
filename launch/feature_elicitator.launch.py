import os

from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    config = LaunchConfiguration('config',
                                 default=os.path.join(
                                     get_package_share_directory('ferl'),
                                     'config',
                                     'feature_elicitator.yaml'))
    
    return LaunchDescription([
        DeclareLaunchArgument(
            'config',
            default_value=config,
            description='Path to the config file'
        ),
        Node(
            package='ferl',
            executable='feature_elicitator',
            name='feature_elicitator',
            output='screen',
            parameters=[
                config
            ]
        )
    ])