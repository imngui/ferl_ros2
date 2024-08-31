from ament_index_python.packages import get_package_share_directory

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor # Enables the description of parameters

import math
import sys, select, os
import time
import torch

import rclpy.wait_for_message
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import String, Float64MultiArray, Bool
from std_srvs.srv import Trigger
from sensor_msgs.msg import JointState

from moveit_msgs.srv import ServoCommandType
from geometry_msgs.msg import WrenchStamped, TwistStamped, Vector3
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from scipy.spatial.transform import Rotation as R

from ferl.controllers.pid_controller import PIDController
from ferl.planners.trajopt_planner import TrajoptPlanner
from ferl.learners.phri_learner import PHRILearner
from ferl.utils import ros2_utils, openrave_utils
from ferl.utils.environment import Environment
from ferl.utils.trajectory import Trajectory

import ast
import numpy as np
import threading

def convert_string_array_to_dict(string_array):
    feat_range_dict = {}
    for item in string_array:
        key, value = item.split(':')
        feat_range_dict[key] = float(value)  # Convert the value to a float
    return feat_range_dict
    
def convert_string_array_to_dict_of_lists(string_array):
    object_centers_dict = {}
    for item in string_array:
        key, value = item.split(':')
        # Use ast.literal_eval to safely evaluate the string as a Python list
        object_centers_dict[key] = ast.literal_eval(value)
    return object_centers_dict
    
def get_parameter_as_dict(string_array):
    """
    Convert a StringArray parameter to a dictionary with the appropriate Python types.
    """
    converted_dict = {}
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

class XRFerl(Node):

    def __init__(self):
        super().__init__('xr_ferl_node')

        self.load_params()

        self.register_callbacks()

        # Run the main loop.
        # self.run_thread = threading.Thread(target=self.run)
        # self.run_thread.start()

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
        self.start = np.array(pick)*(math.pi/180.0)
        place = self.get_parameter('setup.goal').value
        self.goal = np.array(place)*(math.pi/180.0)
        # self.get_logger().info(f'start: {np.array2string(self.start)}')
        # self.get_logger().info(f'goal: {np.array2string(self.goal)}')

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
        # print("object_centers: ", object_centers)
        feat_list = self.get_parameter('setup.feat_list').value
        weights = self.get_parameter('setup.feat_weights').value
        FEAT_RANGE = get_parameter_as_dict(self.get_parameter('setup.FEAT_RANGE').value)
        # print("FEAT_RANGE: ", FEAT_RANGE)
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
        # self.i = 0

        # ----- Planner Setup ----- #
        # Retrieve the planner specific parameters.
        planner_type = self.get_parameter('planner.type').value
        if planner_type == "trajopt":
            max_iter = self.get_parameter('planner.max_iter').value
            num_waypts = self.get_parameter('planner.num_waypts').value

            # Initialize planner and compute trajectory to track.
            self.planner = TrajoptPlanner(max_iter, num_waypts, self.environment)
            # TODO: Implement TrajoptPlanner class.
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
            # TODO: Change np.eye(7) to correct arm dofs.
            P = self.get_parameter('controller.p_gain').value * np.ones((self.num_dofs, 1))
            I = self.get_parameter('controller.i_gain').value * np.ones((self.num_dofs, 1))
            D = self.get_parameter('controller.d_gain').value * np.ones((self.num_dofs, 1))

            # Stores proximity threshold.
            epsilon = self.get_parameter('controller.epsilon').value

            # Stores maximum COMMANDED joint torques.
            MAX_CMD = self.get_parameter('controller.max_cmd').value

            self.controller = PIDController(P, I, D, epsilon, MAX_CMD)
            # TODO: Implement PIDController class.
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
        

        # Set the rate to 500 Hz
        # self.wrench_timer = self.create_timer(1.0 / 500.0, self.timer_callback)

        self.latest_wrench = None

        self.new_plan_timer = None # self.create_timer(0.2, self.new_plan_callback)
        self.begin_motion_timer = None
        self.can_move = True



    def new_plan_callback(self):
        self.can_move = True
        # self.cmd = np.eye(self.num_dofs)
        self.get_logger().info(f'Done Learning, Resuming Planning')
        self.publish_user_info("Done Learning, Resuming Planning")
        self.new_plan_timer = None


    def register_callbacks(self):
        """
        Set up all the subscribers and publishers needed.
        """
        self.traj_timer = self.create_timer(0.1, self.publish_trajectory)
        self.trajectory_pub = self.create_publisher(JointTrajectory, '/scaled_joint_trajectory_controller/joint_trajectory', 10)
        self.vel_trajectory_pub = self.create_publisher(JointTrajectory, '/scaled_vel_joint_trajectory_controller/joint_trajectory', 10)
        self.vel_pub = self.create_publisher(Float64MultiArray, '/forward_velocity_controller/commands', 10)
        self.joint_angles_sub = self.create_subscription(JointState, '/joint_states', self.joint_angles_callback, 10)
        # self.joint_currents_sub = self.create_subscription(JointState, '/joint_states', self.joint_currents_callback, 10)
        self.joint_current_timer = self.create_timer(0.01, self.check_interaction)

        self.interaction_sub = self.create_subscription(Bool, '/interaction', self.interaction_callback, 10)
        self.demo_sub = self.create_subscription(JointTrajectory, '/joint_trajectory', self.demo_callback, 10)
        self.info_pub = self.create_publisher(String, '/user_info', 10)

        self.satisfied_publisher = self.create_publisher(Bool, '/req_satisfied', 10)
        # Subscribe to the force-torque sensor data
        self.force_torque_subscription = self.create_subscription(
            WrenchStamped,
            '/force_torque_sensor_broadcaster/wrench',
            self.wrench_callback,
            10)
        
        self.ready_for_ft_pub = self.create_publisher(JointTrajectory, '/feedback_request', 10)

    def publish_user_info(self, message):
        msg = String()
        msg.data = message
        self.info_pub.publish(msg)

    def interaction_callback(self, msg):
        self.interaction = msg.data
        self.can_move = not msg.data
        if self.interaction:
            self.get_logger().info('Interaction started')
            self.cmd = np.zeros((self.num_dofs, self.num_dofs))
        else:
            self.get_logger().info('Interaction not started')
    
    def demo_callback(self, msg):
        # Convert the message to interaction data in the form of torques.
        self.interaction_data = []
        self.interaction_time = []
        for point in msg.points:
            self.interaction_data.append(np.array(point.effort)) # TODO are the joints in the right order???
            self.interaction_time.append(point.time_from_start.to_sec()) #TODO is this the right time?
        self.interaction_mode = True
        self.timestamp = time.time() - self.controller.path_start_T
            
    def traj_to_raw(self, traj):
        raw_data = []
        for waypt in traj.waypts:
            raw_data.append(self.environment.raw_features(waypt))
        return raw_data

    def check_interaction(self):
        curr_torque = self.interaction_data[-1] if len(self.interaction_data) > 0 else np.zeros((self.num_dofs, self.num_dofs))
        # TODO fix logic
        # self.get_logger().info(f'Interaction')
        # if self.reached_start and not self.reached_goal:
        #     timestamp = time.time() - self.controller.path_start_T
        #     self.interaction_data.append(curr_torque)
        #     self.interaction_time.append(timestamp)
        #     if self.interaction_mode == False:
        #         self.interaction_mode = True
        #         self.can_move = False
        #         self.cmd = np.zeros((self.num_dofs, self.num_dofs))

        if self.interaction:
            return # Do nothing if interaction is currently happening

        # else:
        self.get_logger().info(f'No interaction')
        # self.interaction_mode = True
        if self.interaction_mode and not self.learning:
            self.learning = True
            self.get_logger().info(f'Learning')
            self.publish_user_info("Learning")
            # Check if betas are above CONFIDENCE_THRESHOLD.
            betas = self.learner.learn_betas(self.traj, self.interaction_data[0], self.interaction_time[0])
            for beta in betas:
                self.get_logger().info(f'beta: {beta}')
            if max(betas) < self.CONFIDENCE_THRESHOLD:
                # We must learn a new feature that passes CONFIDENCE_THRESHOLD before resuming.
                self.get_logger().info("The robot does not understand the input!")
                self.publish_user_info("The robot does not understand the input!")
                self.feature_learning_mode = True
                feature_learning_timestamp = time.time()
                input_size = len(self.environment.raw_features(curr_torque))
                self.environment.new_learned_feature(self.nb_layers, self.nb_units)
                while True:
                    # Keep asking for input until we are confident.
                    for i in range(self.N_QUERIES):
                        self.get_logger().info("Need more data to learn the feature!")
                        self.publish_user_info("Need more data to learn the feature!")
                        self.feature_data = []

                        # # Request the person to place the robot in a low feature value state.
                        self.get_logger().info("Place the robot in a low feature value state and move to a high feature value state.")
                        self.publish_user_info("Place the robot in a low feature value state and move to a high feature value state.")
                        ready_msg = Bool()
                        ready_msg.data = True
                        self.ready_for_ft_pub.publish(ready_msg)
                        
                        self.get_logger().info("Waiting for feature trace...")
                        self.publish_user_info("Waiting for feature trace...")
                        rec, msg = rclpy.wait_for_message.wait_for_message(JointTrajectory, self, '/joint_trajectory')
                        if rec:
                            # Get the trajectory from XR
                            traj_data = ros2_utils.traj_msg_to_trajectory(msg, self.joint_names)
                            
                            # Map it to raw features
                            self.feature_data = self.traj_to_raw(traj_data)
                            
                            # Pre-process the recorded data before training.
                            feature_data = np.squeeze(np.array(self.feature_data))
                            lo = 0
                            hi = feature_data.shape[0] - 1
                            while np.linalg.norm(feature_data[lo] - feature_data[lo + 1]) < 0.01 and lo < hi:
                                lo += 1
                            while np.linalg.norm(feature_data[hi] - feature_data[hi - 1]) < 0.01 and hi > 0:
                                hi -= 1
                            feature_data = feature_data[lo:hi + 1, :]
                            self.get_logger().info("Collected {} samples out of {}.".format(feature_data.shape[0], len(self.feature_data)))
                            self.publish_user_info("Collected {} samples out of {}.".format(feature_data.shape[0], len(self.feature_data)))

                            # TODO: Put this in XR
                            # Provide optional start and end labels.
                            start_label = 0.0
                            end_label = 1.0
                            # self.get_logger().info("Would you like to label your start? Press ENTER if not or enter a number from 0-10")
                            # line = sys.stdin.readline()
                            # if line in [str(i) for i in range(11)]:
                            #     start_label = int(i) / 10.0

                            # self.get_logger().info("Would you like to label your goal? Press ENTER if not or enter a number from 0-10")
                            # line = sys.stdin.readline()
                            # if line in [str(i) for i in range(11)]:
                            #     end_label = int(i) / 10.0

                            # Add the newly collected data.
                            self.environment.learned_features[-1].add_data(feature_data, start_label, end_label)
                        else:
                            i = i - 1
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
                    rec, msg = rclpy.wait_for_message.wait_for_message(Bool, self, '/feedback_response')
                    if msg.data:
                        break

                    # line = sys.stdin.readline()
                    # if line == "yes" or line == "Y" or line == "y":
                    #     break
                
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

            self.get_logger().info('Generating new trajectory')
            self.publish_user_info("Generating new trajectory")

            self.get_logger().info('Updating openrave robot state')
            self.publish_user_info("Updating openrave robot state")
            self.environment.env.GetRobots()[0].SetActiveDOFValues(self.start)
            self.get_logger().info('Replanning')
            self.publish_user_info("Replanning")
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

        # Convert to radians.
        curr_pos = curr_pos
        
        # When no in feature learning stage, update position.
        self.curr_pos = curr_pos
        self.curr_vel = np.roll(np.array(msg.velocity),1).reshape(self.num_dofs,1)

        # Update cmd from PID based on current position.
        self.cmd = self.controller.get_command(self.curr_pos, self.curr_vel)

        # Check if start/goal has been reached.
        if self.controller.path_start_T is not None:
            self.reached_start = True
        if self.controller.path_end_T is not None:
            self.reached_goal = True


    def publish_trajectory(self):
        if self.initial_joint_positions is None:
            return
        # self.get_logger().info(f'im: {self.interaction_mode}, cm: {self.can_move}, flm: {self.feature_learning_mode}')
        if not self.interaction_mode and self.can_move and not self.feature_learning_mode:
            # self.get_logger().info('Publishing trajectory')
            joint_vel = np.array([self.cmd[i][i] for i in range(len(self.joint_names))]) 
            
            # Float64MultiArray
            traj_msg = Float64MultiArray()
            traj_msg.data = joint_vel
            self.vel_pub.publish(traj_msg)


def main(args=None):
    rclpy.init(args=args)

    xr_ferl_node = XRFerl()
    
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(xr_ferl_node)

    try:
        executor.spin()
    finally:
        xr_ferl_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
    