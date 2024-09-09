from ament_index_python.packages import get_package_share_directory

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor # Enables the description of parameters

import math
import sys, select, os
import time
import torch
import pickle

import rclpy.wait_for_message
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import String, Float64MultiArray, Bool
from std_srvs.srv import Trigger
from sensor_msgs.msg import JointState

from moveit_msgs.srv import ServoCommandType
from controller_manager_msgs.srv import SwitchController
from geometry_msgs.msg import WrenchStamped, TwistStamped, Vector3
from tf2_ros import Buffer, TransformListener
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from scipy.spatial.transform import Rotation as R

from ferl.controllers.pid_controller import PIDController
from ferl.planners.trajopt_planner import TrajoptPlanner
from ferl.learners.phri_learner import PHRILearner
from ferl.utils import ros2_utils, openrave_utils
from ferl.utils.environment import Environment
from ferl.utils.trajectory import Trajectory

from collections import defaultdict

import ast
import numpy as np

def convert_string_array_to_dict(string_array):
    feat_range_dict = defaultdict(None)
    for item in string_array:
        key, value = item.split(':')
        feat_range_dict[key] = float(value)  # Convert the value to a float
    return feat_range_dict
    
def convert_string_array_to_dict_of_lists(string_array):
    object_centers_dict = defaultdict(None)
    for item in string_array:
        key, value = item.split(':')
        # Use ast.literal_eval to safely evaluate the string as a Python list
        object_centers_dict[key] = ast.literal_eval(value)
    return object_centers_dict
    
def get_parameter_as_dict(string_array):
    """
    Convert a StringArray parameter to a dictionary with the appropriate Python types.
    """
    converted_dict = defaultdict(None)
    for item in string_array:
        key, value_str = item.split(':', 1)  # Split on the first colon
        converted_dict[key] = convert_string_to_appropriate_type(value_str)
    return converted_dict

def convert_string_to_appropriate_type(value_str):
    """
    Attempt to convert a string to its appropriate Python type.
    """
    try:
        # Try to evaluate the string as a Python literal (e.g., list, dict, int, float, bool, None)
        return ast.literal_eval(value_str)
    except (ValueError, SyntaxError):
        # If evaluation fails, return the string as is
        return value_str

class FerlInit(Node):

    def __init__(self):
        super().__init__('ferl_init_node')

        self.load_params()

        self.register_callbacks()


    def load_params(self):
        """
		Loading parameters and setting up variables from the ROS environment.
		"""
        # Declare parameters
        self.declare_parameter('setup.prefix', descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('setup.model_filename', descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('setup.object_centers', descriptor=ParameterDescriptor(dynamic_typing=True))  # Declaring object_centers as a map
        self.declare_parameter('setup.feat_list', descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('setup.feat_weights', descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('setup.start', descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('setup.goal', descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('setup.goal_pose', None, descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('setup.T', descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('setup.timestep', descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('setup.save_dir', descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('setup.INTERACTION_TORQUE_THRESHOLD', descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('setup.INTERACTION_TORQUE_EPSILON', descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('setup.FEAT_RANGE', descriptor=ParameterDescriptor(dynamic_typing=True))  # Declaring FEAT_RANGE as a map
        self.declare_parameter('setup.LF_dict', descriptor=ParameterDescriptor(dynamic_typing=True))  # Declaring LF_dict as a map
        self.declare_parameter('setup.CONFIDENCE_THRESHOLD', descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('setup.N_QUERIES', descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('setup.nb_layers', descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('setup.nb_units', descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('planner.type', descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('planner.max_iter', descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('planner.num_waypts', descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('controller.type', descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('controller.p_gain', descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('controller.i_gain', descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('controller.d_gain', descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('controller.epsilon', descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('controller.max_cmd', descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('learner.type', descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('learner.step_size', descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('learner.alpha', descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('learner.n', descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('learner.P_beta', descriptor=ParameterDescriptor(dynamic_typing=True))  # Declaring P_beta as a map

        # ----- General Setup ----- #
        self.prefix = self.get_parameter('setup.prefix').value
        self.get_logger().info(f'prefix: {self.prefix}')
        pick = self.get_parameter('setup.start').value
        self.get_logger().info(f'pick: {pick}')
        self.start = np.array(pick)#*(math.pi/180.0)
        place = self.get_parameter('setup.goal').value
        self.goal = np.array(place)#*(math.pi/180.0)

        self.goal_pose = self.get_parameter('setup.goal_pose').value
        self.T = self.get_parameter('setup.T').value
        self.timestep = self.get_parameter('setup.timestep').value
        self.save_dir = self.get_parameter('setup.save_dir').value
        self.INTERACTION_TORQUE_THRESHOLD = self.get_parameter('setup.INTERACTION_TORQUE_THRESHOLD').value
        self.INTERACTION_TORQUE_EPSILON = self.get_parameter('setup.INTERACTION_TORQUE_EPSILON').value
        self.CONFIDENCE_THRESHOLD = self.get_parameter('setup.CONFIDENCE_THRESHOLD').value
        self.N_QUERIES = self.get_parameter('setup.N_QUERIES').value
        self.nb_layers = self.get_parameter('setup.nb_layers').value
        self.nb_units = self.get_parameter('setup.nb_units').value

        # Openrave parameters for the environment.
        model_filename = self.get_parameter('setup.model_filename').value
        object_centers = get_parameter_as_dict(self.get_parameter('setup.object_centers').value)
        feat_list = self.get_parameter('setup.feat_list').value
        weights = self.get_parameter('setup.feat_weights').value
        FEAT_RANGE = get_parameter_as_dict(self.get_parameter('setup.FEAT_RANGE').value)
        feat_range = [FEAT_RANGE[feat_list[feat]] for feat in range(len(feat_list))]
        LF_dict = get_parameter_as_dict(self.get_parameter('setup.LF_dict').value)
        self.environment = Environment(model_filename, self.start, object_centers, feat_list, feat_range, np.array(weights), LF_dict)
        self.num_dofs = self.environment.env.GetRobots()[0].GetActiveDOF()
        self.joint_names = np.array([self.environment.env.GetRobots()[0].GetJointFromDOFIndex(i).GetName() for i in self.environment.env.GetRobots()[0].GetManipulator('arm').GetArmIndices()])

        self.num_waypts = self.get_parameter('planner.num_waypts').value
        self.T = self.get_parameter('setup.T').value
        self.timestep = self.get_parameter('setup.timestep').value
        self.joint_names = None
        self.initial_joint_positions = None

        # Track if you have reached the start/goal of the path.
        self.reached_start = False
        self.reached_goal = False
        self.feature_learning_mode = False
        self.prev_interaction_mode = False
        self.interaction_mode = False

        # Save the intermediate target configuration.
        self.curr_pos = None
        self.interaction = False
        self.learning = False

        # Track data and keep stored.
        self.interaction_data = []
        self.interaction_time = []
        self.feature_data = []
        self.track_data = False

        # ----- Planner Setup ----- #
        # Retrieve the planner specific parameters.
        planner_type = self.get_parameter('planner.type').value
        if planner_type == "trajopt":
            max_iter = self.get_parameter('planner.max_iter').value
            num_waypts = self.get_parameter('planner.num_waypts').value

            # Initialize planner and compute trajectory to track.
            self.planner = TrajoptPlanner(max_iter, num_waypts, self.environment)
        else:
            raise Exception('Planner {} not implemented.'.format(planner_type))

        tt = time.time()
        self.traj = self.planner.replan(self.start, self.goal, self.goal_pose, self.T, self.timestep)
        self.get_logger().info(f"Planning time: {time.time() - tt}")
        self.traj_plan = self.traj.downsample(self.planner.num_waypts)

        # ----- Controller Setup ----- #
        # Retrieve controller specific parameters.
        controller_type = self.get_parameter('controller.type').value
        if controller_type == "pid":
            # P, I, D gains.
            P = self.get_parameter('controller.p_gain').value * np.ones((self.num_dofs, 1))
            I = self.get_parameter('controller.i_gain').value * np.ones((self.num_dofs, 1))
            D = self.get_parameter('controller.d_gain').value * np.ones((self.num_dofs, 1))

            # Stores proximity threshold.
            epsilon = self.get_parameter('controller.epsilon').value

            # Stores maximum COMMANDED joint torques.
            MAX_CMD = self.get_parameter('controller.max_cmd').value

            self.controller = PIDController(P, I, D, epsilon, MAX_CMD)
        else:
            raise Exception('Controller {} not implemented.'.format(controller_type))
        
        # Planner tells controller what plan to follow.
        traj = Trajectory([self.start], [0.0])
        self.controller.set_trajectory(traj)

        self.cmd = np.zeros((self.num_dofs, self.num_dofs))

        # ----- Learner Setup ----- #
        constants = {}
        constants["step_size"] = self.get_parameter("learner.step_size").value
        constants["P_beta"] = self.get_parameter("learner.P_beta").value
        constants["alpha"] = self.get_parameter("learner.alpha").value
        constants["n"] = self.get_parameter("learner.n").value
        self.feat_method = self.get_parameter("learner.type").value
        self.learner = PHRILearner(self.feat_method, self.environment, constants)

        # Compliance parameters
        self.Kp = 1.0  # Stiffness (inverse of compliance)
        self.Kd = 0.1  # Damping

        self.current_twist = TwistStamped()  # Keep track of the current twist for damping effect

        # TF Buffer and Listener to get transformations
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
    

        self.latest_wrench = None
        self.new_plan = False

        self.new_plan_timer = None # self.create_timer(0.2, self.new_plan_callback)
        self.begin_motion_timer = None
        self.can_move = True
        self.initialized = False

        self.switch_controller_client = self.create_client(SwitchController, '/controller_manager/switch_controller')
        self.deactivate_controller('scaled_joint_trajectory_controller')

        # Forward Velocity Controller
        self.activate_controller('forward_velocity_controller')


    def activate_controller(self, controller_name):
        if not self.switch_controller_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Switch control sensor service not available, waiting again...')
            return
         
        request = SwitchController.Request()
        request.activate_controllers = [controller_name]

        future = self.switch_controller_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None and future.result().ok:
            self.get_logger().info(f'Activated controller: {controller_name}')
        else:
            self.get_logger().warn(f'Could not activate controller: {controller_name}')


    def deactivate_controller(self, controller_name):
        if not self.switch_controller_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Switch control service not available, waiting again...')
            return
        
        request = SwitchController.Request()
        request.deactivate_controllers = [controller_name]

        future = self.switch_controller_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None and future.result().ok:
            self.get_logger().info(f'Deactivated controller: {controller_name}')
        else:
            self.get_logger().warn(f'Could not deactivate controller: {controller_name}')


    def register_callbacks(self):
        """
        Set up all the subscribers and publishers needed.
        """
        self.traj_timer = self.create_timer(0.1, self.publish_trajectory)
        self.joint_angles_sub = self.create_subscription(JointState, '/joint_states', self.joint_angles_callback, 10)
        self.vel_pub = self.create_publisher(Float64MultiArray, '/forward_velocity_controller/commands', 10)


    def joint_angles_callback(self, msg):
        """
        Reads the latest position of the robot and publishes an
        appropriate torque command to move the robot to the target.
        """
        if self.joint_names is None:
            self.joint_names = np.roll(np.array(msg.name), 1)
        if self.initial_joint_positions is None:
            self.initial_joint_positions = np.roll(np.array(msg.position),1)
            self.joint_positions = self.initial_joint_positions

        curr_pos = np.roll(np.array(msg.position),1).reshape(self.num_dofs,1)

        if not self.initialized:
            self.environment.env.GetRobots()[0].SetActiveDOFValues(curr_pos)
            self.initialized = True
        
        # When no in feature learning stage, update position.
        self.curr_pos = curr_pos
        self.curr_vel = np.roll(np.array(msg.velocity),1).reshape(self.num_dofs,1)
        self.environment.update_curr_pos(curr_pos)

        # Update cmd from PID based on current position.
        if not self.reached_start:
            self.cmd = self.controller.get_command(self.curr_pos, self.curr_vel)
        else:
            self.cmd = np.zeros((self.num_dofs, self.num_dofs))

        # Check if start/goal has been reached.
        if self.controller.path_start_T is not None:
            self.reached_start = True
        if self.controller.path_end_T is not None:
            self.reached_goal = True


    def publish_trajectory(self):
        if self.initial_joint_positions is None:
            return

        joint_vel = np.array([self.cmd[i][i] for i in range(len(self.joint_names))]) 

        # Float64MultiArray
        traj_msg = Float64MultiArray()
        traj_msg.data = joint_vel
        self.vel_pub.publish(traj_msg)


def main(args=None):
    rclpy.init(args=args)

    ferl_init_node = FerlInit()
    
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(ferl_init_node)

    try:
        executor.spin()
    finally:
        ferl_init_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
    