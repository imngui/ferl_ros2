from ament_index_python.packages import get_package_share_directory

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor # Enables the description of parameters

import math
import sys, select, os
import time
import torch

import rclpy.wait_for_message
from std_msgs.msg import String, Float64MultiArray
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger

from moveit_msgs.srv import ServoCommandType
from geometry_msgs.msg import WrenchStamped, TwistStamped, Vector3
from controller_manager_msgs.srv import SwitchController

from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from scipy.spatial.transform import Rotation as R

from ferl.controllers.pid_controller import PIDController
from ferl.planners.trajopt_planner import TrajoptPlanner
from ferl.learners.phri_learner import PHRILearner
from ferl.utils import ros2_utils, openrave_utils, experiment_utils
from ferl.utils.environment import Environment
from ferl.utils.trajectory import Trajectory
from ferl.MaxEnt_Baseline.baseline_utils import map_traj_to_raw_dim

import numpy as np
import threading
import pickle
import ast

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


class DemoRecorder(Node):
    def __init__(self):
        super().__init__('demo_recorder_node')

        self.load_params()
        self.register_callbacks()
        
        self.run_thread = threading.Thread(target=self.run)
        self.run_thread.start()


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
        pick = self.get_parameter('setup.start').value
        self.start = np.array(pick)*(math.pi/180.0)
        place = self.get_parameter('setup.goal').value
        self.goal = np.array(place)*(math.pi/180.0)

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
        self.initial_joint_positions = None

        # Track if you have reached the start/goal of the path.
        self.reached_start = False
        self.reached_goal = False
        self.feature_learning_mode = False
        self.prev_interaction_mode = False
        self.interaction_mode = False

        # Save the intermediate target configuration.
        self.curr_pos = None

        # Track data and keep stored.
        self.interaction_data = []
        self.interaction_time = []
        self.feature_data = []
        self.track_data = False
        self.expUtil = None
        self.prev_interaction = False

        # ----- Planner Setup ----- #
        # Retrieve the planner specific parameters.
        # planner_type = self.get_parameter('planner.type').value
        # if planner_type == "trajopt":
        #     max_iter = self.get_parameter('planner.max_iter').value
        #     num_waypts = self.get_parameter('planner.num_waypts').value

        #     # Initialize planner and compute trajectory to track.
        #     self.planner = TrajoptPlanner(max_iter, num_waypts, self.environment)
        #     # TODO: Implement TrajoptPlanner class.
        # else:
        #     raise Exception('Planner {} not implemented.'.format(planner_type))

        # tt = time.time()
        # self.traj = self.planner.replan(self.start, self.goal, self.goal_pose, self.T, self.timestep)
        # self.get_logger().info(f"Planning time: {time.time() - tt}")
        # self.traj_plan = self.traj.downsample(self.planner.num_waypts)

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

        # # Planner tells controller what plan to follow.
        # self.controller.set_trajectory(Trajectory([self.start], [0.0]))

        
        self.cmd = np.eye(self.num_dofs)

        # Utilities for recording data.
        self.expUtil = experiment_utils.ExperimentUtils(self.save_dir)

        # ----- Learner Setup ----- #
        constants = {}
        constants["step_size"] = self.get_parameter("learner.step_size").value
        constants["P_beta"] = self.get_parameter("learner.P_beta").value
        constants["alpha"] = self.get_parameter("learner.alpha").value
        constants["n"] = self.get_parameter("learner.n").value
        self.feat_method = self.get_parameter("learner.type").value
        self.learner = PHRILearner(self.feat_method, self.environment, constants)

        # Compliance parameters
        self.Kp = 3.0  # Stiffness (inverse of compliance)
        self.Kd = 0.1  # Damping

        self.current_twist = TwistStamped()  # Keep track of the current twist for damping effect

        # TF Buffer and Listener to get transformations
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Set the rate to 500 Hz
        self.wrench_timer = self.create_timer(1.0 / 500.0, self.timer_callback)

        self.latest_wrench = None
        self.new_plan_timer = None # self.create_timer(0.2, self.new_plan_callback)
        self.begin_motion_timer = None
        self.can_move = True
        self.interaction = False
        self.initial_wrench = None
        self.interaction_start = None
        self.prev_pos = None
        self.same_pos_count = 0
        self.last_pos = None
        self.data_timer = None

        # Create a client for the ServoCommandType service
        # self.switch_input_client = self.create_client(ServoCommandType, '/servo_node/switch_command_type')
        # Call the service to enable TWIST command type
        # self.enable_twist_command()

        # self.zero_ft_client = self.create_client(Trigger, '/io_and_status_controller/zero_ftsensor')
        # self.zero_ft_sensor()

        # self.switch_controller_client = self.create_client(SwitchController, '/controller_manager/switch_controller')
        # self.deactivate_controller('scaled_joint_trajectory_controller')

        # Forward Velocity Controller
        # self.activate_controller('forward_velocity_controller')

        # Forward Position Controller
        # self.activate_controller('forward_position_controller')

        self.user_input = None
        self.initialized = False
        # TODO: Update
        self.task_id = 0


    def register_callbacks(self):
        """
        Set up all the subscribers and publishers needed.
        """
        # self.traj_timer = self.create_timer(0.1, self.publish_trajectory)
        # self.vel_pub = self.create_publisher(Float64MultiArray, '/forward_velocity_controller/commands', 10)
        self.joint_angles_sub = self.create_subscription(JointState, '/joint_states', self.joint_angles_callback, 10)
        # self.force_torque_subscription = self.create_subscription(
        #     WrenchStamped,
        #     '/force_torque_sensor_broadcaster/wrench',
        #     self.wrench_callback,
        #     10)
        self.twist_pub_ = self.create_publisher(TwistStamped, '/servo_node/delta_twist_cmds', 10)

        # self.user_input_sub = self.create_subscription(String, '/user_input', self.user_input_callback, 10)
        self.req_user_input_pub = self.create_publisher(String, '/req_user_input', 10)

        # Create a client for the ServoCommandType service
        # self.switch_input_client = self.create_client(ServoCommandType, '/servo_node/switch_command_type')
        # self.enable_twist_command()


    # def new_plan_callback(self):
    #     if not self.interaction:
    #         # self.zero_ft_sensor()
    #         # self.can_move = True
    #         self.finalize_demo_trajectory()

    #         self.new_plan_timer = None


    # def enable_twist_command(self):
    #     if not self.switch_input_client.wait_for_service(timeout_sec=1.0):
    #         self.get_logger().warn('Service not available, waiting again...')
    #         return

    #     request = ServoCommandType.Request()
    #     request.command_type = ServoCommandType.Request.TWIST

    #     future = self.switch_input_client.call_async(request)
    #     rclpy.spin_until_future_complete(self, future)

    #     if future.result() is not None and future.result().success:
    #         self.get_logger().info('Switched to input type: TWIST')
    #     else:
    #         self.get_logger().warn('Could not switch input to: TWIST')


    # def activate_controller(self, controller_name):
    #     if not self.switch_controller_client.wait_for_service(timeout_sec=1.0):
    #         self.get_logger().warn('Switch control sensor service not available, waiting again...')
    #         return
         
    #     request = SwitchController.Request()
    #     request.activate_controllers = [controller_name]

    #     future = self.switch_controller_client.call_async(request)
    #     rclpy.spin_until_future_complete(self, future)

    #     if future.result() is not None and future.result().ok:
    #         self.get_logger().info(f'Activated controller: {controller_name}')
    #     else:
    #         self.get_logger().warn(f'Could not activate controller: {controller_name}')


    # def deactivate_controller(self, controller_name):
    #     if not self.switch_controller_client.wait_for_service(timeout_sec=1.0):
    #         self.get_logger().warn('Switch control service not available, waiting again...')
    #         return
        
    #     request = SwitchController.Request()
    #     request.deactivate_controllers = [controller_name]

    #     future = self.switch_controller_client.call_async(request)
    #     rclpy.spin_until_future_complete(self, future)

    #     if future.result() is not None and future.result().ok:
    #         self.get_logger().info(f'Deactivated controller: {controller_name}')
    #     else:
    #         self.get_logger().warn(f'Could not deactivate controller: {controller_name}')


    # def zero_ft_sensor(self):
    #     if not self.zero_ft_client.wait_for_service(timeout_sec=1.0):
    #         self.get_logger().warn('Zero ft sensor service not available, waiting again...')
    #         return
        
    #     request = Trigger.Request()
    #     future = self.zero_ft_client.call_async(request)
    #     rclpy.spin_until_future_complete(self, future)

    #     if future.result() is not None and future.result().success:
    #         self.get_logger().info('Zero ft sensor complete!')
    #     else:
    #         self.get_logger().warn("Could not zero ft sensor!")


    # def wrench_callback(self, msg):
    #     self.latest_wrench = msg


    # def timer_callback(self):
    #     if self.latest_wrench is not None and self.initialized:
    #         try:
    #             # Look up the transformation from ft_frame to tool0 and then tool0 to base_link
    #             ft_to_tool0 = self.tf_buffer.lookup_transform('tool0', self.latest_wrench.header.frame_id, rclpy.time.Time())

    #             # force = self.sub_init(self.latest_wrench.wrench.force , self.initial_wrench.wrench.force)
    #             # torque = self.sub_init(self.latest_wrench.wrench.torque , self.initial_wrench.wrench.torque)

    #             force = self.latest_wrench.wrench.force 
    #             torque = self.latest_wrench.wrench.torque 

    #             # Transform the force/torque from ft_frame to tool0
    #             force = self.transform_vector(ft_to_tool0, force)
    #             torque = self.transform_vector(ft_to_tool0, torque)

    #             # Nullify force/torque readings with magnitude < 3
    #             force = self.nullify_small_magnitudes(force, 10.0)
    #             torque = self.nullify_small_magnitudes(torque, 10.0)

    #             self.prev_interaction = self.interaction

    #             if math.sqrt(force.x ** 2 + force.y ** 2 + force.z ** 2) < 10.0:
    #                 # self.interaction = False
    #                 # if self.prev_interaction != self.interaction:
    #                     # self.new_plan_timer = self.create_timer(1.0, self.new_plan_callback)
    #                 return

    #             self.interaction = True
    #             self.can_move = False
    #             if self.interaction_start is None:
    #                 self.interaction_start = time.time()

    def run(self):
        cnt = 0
        task = self.task_id
        while rclpy.ok():
            self.get_logger().info("Type [yes/y/Y] if you're ready to record a demonstration.")
            user_input_req = String()
            user_input_req.data = "Type [yes/y/Y] if you're ready to record a demonstration."
            self.req_user_input_pub.publish(user_input_req)
            
            rec, msg = rclpy.wait_for_message.wait_for_message(String, self, '/user_input')
            line = msg.data
            if (line is not "yes") and (line is not "Y") and (line is not "y"):
                self.get_logger().info("WHY?.")
            else:
                self.interaction_start = time.time()
                self.initialized = True
                
                self.get_logger().info("Type [quit/q/Q] if you're ready to stop recording.")
                user_input_req = String()
                user_input_req.data = "Type [quit/q/Q] if you're ready to stop recording."
                self.req_user_input_pub.publish(user_input_req)
                
                rec, msg = rclpy.wait_for_message.wait_for_message(String, self, '/user_input')
                line = msg.data
                if (line is not "quit") and (line is not "Q") and (line is not "q"):
                    self.get_logger().info("WHY?.")
                else:
                    self.finalize_demo_trajectory(cnt)


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
            # Planner tells controller what plan to follow.
            self.controller.set_trajectory(Trajectory([curr_pos], [0.0]))
            self.initialized = True

        # When no in feature learning stage, update position.
        self.curr_pos = curr_pos

        if self.interactioin:
            timestamp = time.time() - self.interaction_start
            self.expUtil.update_tracked_traj(timestamp, curr_pos)


    def data_collected(self):
        self.get_logger().info("HERE")
        if np.linalg.norm(self.curr_pos - self.last_pos) < 1e-2:
            self.interaction = False
            self.finalize_demo_trajectory()
        self.data_timer = None

    def finalize_demo_trajectory(self, demo_number):
        self.get_logger().info("Final")

        # Process and save the recording.
        raw_demo = self.expUtil.tracked_traj[:,1:7]

        # Trim ends of waypoints and create Trajectory.
        lo = 0
        hi = raw_demo.shape[0] - 1
        while np.linalg.norm(raw_demo[lo] - raw_demo[lo + 1]) < 1e-6 and lo < hi:
            # self.get_logger().info(f'lo-{lo}: {np.linalg.norm(raw_demo[lo] - raw_demo[lo + 1])}')
            lo += 1
        while np.linalg.norm(raw_demo[hi] - raw_demo[hi - 1]) < 1e-6 and hi > 0:
            hi -= 1
        waypts = raw_demo[lo:hi+1, :]
        waypts_time = np.linspace(0.0, self.T, waypts.shape[0])
        traj = Trajectory(waypts, waypts_time)

        # Downsample/Upsample trajectory to fit desired timestep and T.
        num_waypts = int(self.T / self.timestep) + 1
        if num_waypts < len(traj.waypts):
            demo = traj.downsample(int(self.T / self.timestep) + 1)
        else:
            demo = traj.upsample(int(self.T / self.timestep) + 1)

        # Decide whether to save trajectory
        openrave_utils.plotTraj(self.environment.env, self.environment.robot,
        				self.environment.bodies, demo.waypts, color=[0, 0, 1])

        demo_feat_trace = map_traj_to_raw_dim(self.environment, traj.waypts)

        self.ready_for_input = True
        self.get_logger().info("Type [yes/y/Y] if you're happy with the demonstration.")
        user_input_req = String()
        user_input_req.data = "Type [yes/y/Y] if you're happy with the demonstration."
        self.req_user_input_pub.publish(user_input_req)
        # line = input()
        rec, msg = rclpy.wait_for_message.wait_for_message(String, self, '/user_input')
        line = msg.data
        if (line is not "yes") and (line is not "Y") and (line is not "y"):
            self.get_logger().info("Not happy with demonstration. Terminating experiment.")
        else:
            # self.get_logger().info("Please type in the ID number (e.g. [0/1/2/...]).")
            # user_input_req.data = "Please type in the ID number (e.g. [0/1/2/...])."
            # self.req_user_input_pub.publish(user_input_req)
            # # ID = input()
            # rec, msg = rclpy.wait_for_message.wait_for_message(String, self, '/user_input')
            # ID = msg.data
            ID = demo_number
            # self.get_logger().info("Please type in the task number (e.g. [0/1/2/...]).")
            # user_input_req.data = "Please type in the task number (e.g. [0/1/2/...])."
            # self.req_user_input_pub.publish(user_input_req)
            # rec, msg = rclpy.wait_for_message.wait_for_message(String, self, '/user_input')
            # task = msg.data
            task = self.task_id
            filename = "demo" + "_ID" + ID + "_task" + task + ".p"
            savefile = os.path.join(get_package_share_directory('ferl'), 'data', 'demonstrations', 'demos', filename)
            ft_filename = "ft_" + filename
            ft_savefile = os.path.join(get_package_share_directory('ferl'), 'data', 'demonstrations', 'demos', ft_filename)
            
            with open(savefile, "wb") as f:
                pickle.dump(demo, f)
            with open(ft_savefile, "wb") as f:
                pickle.dump(demo_feat_trace, f)
            self.get_logger().info("Saved demonstration in {}.".format(savefile))
        

def main(args=None):
    rclpy.init(args=args)

    demo_recorder_node = DemoRecorder()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(demo_recorder_node)

    try:
        executor.spin()
    finally:
        # demo_recorder_node.finalize_demo_trajectory()
        demo_recorder_node.destroy_node()
        rclpy.shutdown()

    print("test: ", test)

if __name__ == '__main__':
    main()
