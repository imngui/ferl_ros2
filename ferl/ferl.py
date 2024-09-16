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

class Ferl(Node):

    def __init__(self):
        super().__init__('ferl_node')

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
        self.interaction_start = None

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
        self.controller.set_trajectory(self.traj)

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


    def new_plan_callback(self):
        if self.new_plan:
            self.can_move = True
            # self.cmd = np.eye(self.num_dofs)
            self.get_logger().info(f'Done Learning, Resuming Planning')
            self.publish_user_info("Done Learning, Resuming Planning")
            self.new_plan_timer = None
            traj = Trajectory([self.start], [0.0])
            self.controller.set_trajectory(traj)
            self.controller.path_start_T = None
            self.reached_start = False
            
            

    def register_callbacks(self):
        """
        Set up all the subscribers and publishers needed.
        """
        self.traj_timer = self.create_timer(0.01, self.publish_trajectory)
        self.trajectory_pub = self.create_publisher(JointTrajectory, '/scaled_joint_trajectory_controller/joint_trajectory', 10)
        self.vel_trajectory_pub = self.create_publisher(JointTrajectory, '/scaled_vel_joint_trajectory_controller/joint_trajectory', 10)
        self.vel_pub = self.create_publisher(Float64MultiArray, '/forward_velocity_controller/commands', 10)
        self.joint_angles_sub = self.create_subscription(JointState, '/joint_states', self.joint_angles_callback, 10)
        self.joint_currents_sub = self.create_subscription(JointState, '/joint_states', self.joint_currents_callback, 10)
        self.joint_current_timer = self.create_timer(0.01, self.check_interaction)

        self.info_pub = self.create_publisher(String, '/req_user_input', 10)
        self.twist_pub_ = self.create_publisher(TwistStamped, '/servo_node/delta_twist_cmds', 10)

        self.satisfied_publisher = self.create_publisher(Bool, '/req_satisfied', 10)
        # Subscribe to the force-torque sensor data
        self.force_torque_subscription = self.create_subscription(
            WrenchStamped,
            '/force_torque_sensor_broadcaster/wrench',
            self.wrench_callback,
            10)
        
        # Set the rate to 500 Hz
        self.wrench_timer = self.create_timer(1.0 / 500.0, self.timer_callback)
        
        self.ready_for_ft_pub = self.create_publisher(Bool, '/feedback_request', 10)
        self.zero_ft_client = self.create_client(Trigger, '/io_and_status_controller/zero_ftsensor')
        self.zero_ft_sensor()
        
    def joint_currents_callback(self, msg):
        # Convert joint current to torque using UR5e torque constants
        self.curr_torque = np.roll(np.array(msg.effort), 1) * [0.125, 0.125, 0.125, 0.092, 0.092, 0.092]
        
    def timer_callback(self):
        # return
        if self.latest_wrench is not None and self.initialized:
            try:
                # Look up the transformation from ft_frame to tool0 and then tool0 to base_link
                ft_to_tool0 = self.tf_buffer.lookup_transform('tool0', self.latest_wrench.header.frame_id, rclpy.time.Time())

                force = self.latest_wrench.wrench.force 
                torque = self.latest_wrench.wrench.torque 

                # Transform the force/torque from ft_frame to tool0
                force = self.transform_vector(ft_to_tool0, force)
                torque = self.transform_vector(ft_to_tool0, torque)

                # Nullify force/torque readings with magnitude < 3
                force = self.nullify_small_magnitudes(force, 10.0)
                torque = self.nullify_small_magnitudes(torque, 10.0)

                self.prev_interaction = self.interaction

                if math.sqrt(force.x ** 2 + force.y ** 2 + force.z ** 2) < 10.0:
                    self.interaction = False
                    self.can_move = True
                    return

                self.interaction = True
                self.can_move = False
                self.cmd = np.zeros((self.num_dofs, self.num_dofs))
                if self.interaction_start is None:
                    self.interaction_start = time.time()

                # Compute the twist in ee frame
                twist = TwistStamped()
                twist.header.stamp = self.get_clock().now().to_msg()
                twist.header.frame_id = 'tool0'

                twist.twist.linear.x = (1 / self.Kp) * force.x - self.Kd * self.current_twist.twist.linear.x
                twist.twist.linear.y = (1 / self.Kp) * force.y - self.Kd * self.current_twist.twist.linear.y
                twist.twist.linear.z = (1 / self.Kp) * force.z - self.Kd * self.current_twist.twist.linear.z

                twist.twist.angular.x = (1 / self.Kp) * torque.x - self.Kd * self.current_twist.twist.angular.x
                twist.twist.angular.y = (1 / self.Kp) * torque.y - self.Kd * self.current_twist.twist.angular.y
                twist.twist.angular.z = (1 / self.Kp) * torque.z - self.Kd * self.current_twist.twist.angular.z

                # Update the current twist for the next callback
                self.current_twist = twist

                # Publish the computed twist
                self.twist_pub_.publish(twist)

            except (LookupException, ConnectivityException, ExtrapolationException) as e:
                self.get_logger().warn(f"Could not transform wrench to base_link frame: {str(e)}")


    def transform_vector(self, transform, vector):
        # Extract rotation (quaternion) and translation from TransformStamped
        q = transform.transform.rotation

        # Convert quaternion to rotation matrix using scipy
        r = R.from_quat([q.x, q.y, q.z, q.w])

        # Convert Vector3 to numpy array for easy multiplication
        vector_np = np.array([vector.x, vector.y, vector.z])

        # Apply the rotation
        rotated_vector = r.apply(vector_np)

        # Return the transformed vector as a Vector3
        return Vector3(x=rotated_vector[0], y=rotated_vector[1], z=rotated_vector[2])


    def nullify_small_magnitudes(self, vector, threshold):
        magnitude = math.sqrt(vector.x ** 2 + vector.y ** 2 + vector.z ** 2)
        if magnitude < threshold or np.isnan(magnitude):
            return Vector3(x=0.0, y=0.0, z=0.0)
        else:
            return vector

    def publish_user_info(self, message):
        msg = String()
        msg.data = message
        self.info_pub.publish(msg)
        
    def wrench_callback(self, msg):
        self.latest_wrench = msg
            
    def traj_to_raw(self, traj):
        raw_data = []
        for waypt in traj.waypts:
            raw_data.append(self.environment.raw_features(waypt))
        return raw_data
    
    def traj_data_to_raw(self, traj_data):
        raw_traj_data = []
        for traj_pt in traj_data:
            raw_traj_data.append(self.environment.raw_features(traj_pt))
        return raw_traj_data

    def check_interaction(self):
        curr_torque = self.curr_torque        
        
        if self.interaction:
            self.get_logger().info(f'Interaction')
            timestamp = time.time() - self.controller.path_start_T
            self.interaction_data.append(curr_torque)
            self.interaction_time.append(timestamp)
            if self.interaction_mode == False:
                self.interaction_mode = True
        else:
            # self.interaction_mode = True
            if self.interaction_mode == True:
                self.get_logger().info(f'Learning')
                self.publish_user_info("Learning")
                # Check if betas are above CONFIDENCE_THRESHOLD.
                betas = self.learner.learn_betas(self.traj, self.interaction_data[0], self.interaction_time[0])
                # for beta in betas:
                self.get_logger().info(f'betas: {betas}')
                if max(betas) < self.CONFIDENCE_THRESHOLD:
                    # We must learn a new feature that passes CONFIDENCE_THRESHOLD before resuming.
                    self.get_logger().info("The robot does not understand the input!")
                    self.publish_user_info("The robot does not understand the input!")
                    self.feature_learning_mode = True
                    feature_learning_timestamp = time.time()
                    input_size = len(self.environment.raw_features(curr_torque))
                    self.environment.new_learned_feature(self.nb_layers, self.nb_units)
                    while True:
                        np.zeros((self.num_dofs, self.num_dofs))
                        # Keep asking for input until we are confident.
                        for i in range(self.N_QUERIES):
                            np.zeros((self.num_dofs, self.num_dofs))
                            self.get_logger().info("Need more data to learn the feature!")
                            self.publish_user_info("Need more data to learn the feature!")
                            self.feature_data = []

                            # Request the person to place the robot in a low feature value state.
                            self.get_logger().info("Place the robot in a low feature value state and press ENTER when ready.")
                            self.publish_user_info("Place the robot in a low feature value state and press ENTER when ready.")
                            rec, msg = rclpy.wait_for_message.wait_for_message(String, self, '/user_input')
                            if rec:
                                self.track_data = True
                                
                                self.get_logger().info("Move the robot to a high feature value state and press ENTER when ready.")
                                self.publish_user_info("Move the robot to a high feature value state and press ENTER when ready.")
                                rec, msg = rclpy.wait_for_message.wait_for_message(String, self, '/user_input')
                                if rec:
                                    self.track_data = False
                                                                        
                                    # Pre-process the recorded data before training.
                                    feature_data = np.squeeze(np.array(self.feature_data))
                                    lo = 0
                                    hi = feature_data.shape[0] - 1
                                    # TODO: Figure out if 1e-5 is an appropriate 
                                    while np.linalg.norm(feature_data[lo] - feature_data[lo + 1]) < 1e-5 and lo < hi:
                                        # self.get_logger().info(f'tol: {np.linalg.norm(feature_data[lo] - feature_data[lo + 1])}, pos: {feature_data[lo]}')
                                        lo += 1
                                    while np.linalg.norm(feature_data[hi] - feature_data[hi - 1]) < 1e-5 and hi > 0:
                                        hi -= 1
                                    self.get_logger().info(f'torque lo: {lo}')
                                    self.get_logger().info(f'torque hi: {hi}')
                                    feature_data = feature_data[lo:hi + 1, :][::-1]
                                    self.get_logger().info("Collected {} demonstrations out of {}.".format(i+1, self.N_QUERIES))
                                    self.publish_user_info("Collected {} demonstrations out of {}.".format(i+1, self.N_QUERIES))

                                    # Provide optional start and end labels.
                                    start_label = 0.0
                                    end_label = 1.0

                                    # Add the newly collected data.
                                    self.environment.learned_features[-1].add_data(feature_data, start_label, end_label)
                                else:
                                    self.get_logger().info("Failed to get feature trace. Retrying...")
                                    self.publish_user_info("Failed to get feature trace. Retrying...")
                            
                        # Train new feature with data of increasing "goodness".
                        self.environment.learned_features[-1].train()

                        # Check if we are happy with the input.
                        self.get_logger().info("Are you happy with the training? (y/n)")
                        self.publish_user_info("Are you happy with the training? (y/n)")
                        satisfied_msg = Bool()
                        satisfied_msg.data = True
                        self.satisfied_publisher.publish(satisfied_msg)
                        rec, msg = rclpy.wait_for_message.wait_for_message(Bool, self, '/user_input')
                        if msg.data:
                            break

                    # Compute new beta for the new feature.
                    beta_new = self.learner.learn_betas(self.traj, curr_torque, self.timestamp, [self.environment.num_features - 1])[0]
                    betas.append(beta_new)

                    # Move time forward to return to interaction position.
                    self.controller.path_start_T += (time.time() - feature_learning_timestamp)

                # We do no have misspecification now, so resume reward learning.
                self.feature_learning_mode = False
                
                # learn reward.
                self.get_logger().info('Learning weights')
                self.publish_user_info("Learning weights")
                for i in range(len(self.interaction_data)):
                    self.learner.learn_weights(self.traj, self.interaction_data[i], self.interaction_time[i], betas)

                self.get_logger().info(f'weights: {self.environment.weights}')

                self.get_logger().info('Generating new trajectory')
                self.publish_user_info("Generating new trajectory")
                self.environment.update_curr_pos(self.start)
                self.traj = self.planner.replan(self.start, self.goal, self.goal_pose, self.T, self.timestep, seed=self.traj_plan.waypts)
                self.get_logger().info('Downsampling')
                self.publish_user_info("Downsampling")
                self.traj_plan = self.traj.downsample(self.planner.num_waypts)
                self.get_logger().info('Updating Controller')
                self.publish_user_info("Updating Controller")
                self.controller.set_trajectory(self.traj)

                # Turn off interaction mode.
                self.interaction_mode = False
                self.interaction_data = []
                self.interaction_time = []
                self.learning = False
                self.get_logger().info('Done!')
                self.publish_user_info("Done!")
                self.new_plan = True
                self.new_plan_timer = self.create_timer(1.0, self.new_plan_callback)
                    


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
            traj = Trajectory([self.start], [0.0])
            self.controller.set_trajectory(traj)
            self.initialized = True
            
        if self.feature_learning_mode:
            self.cmd = np.zeros((self.num_dofs, self.num_dofs))
            
            if self.track_data == True:
                self.feature_data.append(self.environment.raw_features(curr_pos))
                
            return
        
        # When no in feature learning stage, update position.
        self.curr_pos = curr_pos
        self.curr_vel = np.roll(np.array(msg.velocity),1).reshape(self.num_dofs,1)
        self.environment.update_curr_pos(curr_pos)

        # Update cmd from PID based on current position.
        self.cmd = self.controller.get_command(self.curr_pos, self.curr_vel)

        # Check if start/goal has been reached.
        if self.controller.path_start_T is not None:
            self.reached_start = True
            self.controller.set_trajectory(self.traj)
        if self.controller.path_end_T is not None:
            self.reached_goal = True


    def publish_trajectory(self):
        if self.initial_joint_positions is None:
            return
        
        if self.interaction:
            return
        
        joint_vel = np.zeros(self.num_dofs)
        
        move_start = False
        if self.reached_start == False:
            # self.get_logger().info(f'Move to Start')
            self.cmd = self.controller.get_command(self.curr_pos, self.curr_vel)
            joint_vel = np.array([self.cmd[i][i] for i in range(len(self.joint_names))])
            move_start = True
 
            
        # self.get_logger().info(f'im: {self.interaction_mode}, cm: {self.can_move}, flm: {self.feature_learning_mode}, i: {self.interaction}')
        if not self.interaction_mode and self.can_move and not move_start and not self.feature_learning_mode and not self.interaction:
            # if self.controller.path_start_T is None:
            #     self.controller.path_start_T = time.time()
            # self.get_logger().info('Publishing trajectory')
            joint_vel = np.array([self.cmd[i][i] for i in range(len(self.joint_names))]) 

        # Float64MultiArray
        traj_msg = Float64MultiArray()
        traj_msg.data = joint_vel
        self.vel_pub.publish(traj_msg)


def main(args=None):
    rclpy.init(args=args)

    ferl_node = Ferl()
    
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(ferl_node)

    try:
        executor.spin()
    finally:
        ferl_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
    