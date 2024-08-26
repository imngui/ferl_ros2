from ament_index_python.packages import get_package_share_directory

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor # Enables the description of parameters

import math
import sys, select, os
import time

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import String, Float64MultiArray
from sensor_msgs.msg import JointState

from moveit_msgs.srv import ServoCommandType
from geometry_msgs.msg import WrenchStamped, TwistStamped, Vector3
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from scipy.spatial.transform import Rotation as R

from ferl.controllers.pid_controller import PIDController
from ferl.planners.trajopt_planner import TrajoptPlanner
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

class TestVel(Node):

    def __init__(self):
        super().__init__('test_vel_node')

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
        pick = self.get_parameter('setup.start').value
        self.start = np.array(pick)*(math.pi/180.0)
        place = self.get_parameter('setup.goal').value
        self.goal = np.array(place)*(math.pi/180.0)
        self.get_logger().info(f'start: {np.array2string(self.start)}')
        self.get_logger().info(f'goal: {np.array2string(self.goal)}')

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
        print("object_centers: ", object_centers)
        feat_list = self.get_parameter('setup.feat_list').value
        weights = self.get_parameter('setup.feat_weights').value
        FEAT_RANGE = get_parameter_as_dict(self.get_parameter('setup.FEAT_RANGE').value)
        print("FEAT_RANGE: ", FEAT_RANGE)
        feat_range = [FEAT_RANGE[feat_list[feat]] for feat in range(len(feat_list))]
        LF_dict = get_parameter_as_dict(self.get_parameter('setup.LF_dict').value)
        self.environment = Environment(model_filename, self.start, object_centers, feat_list, feat_range, np.array(weights), LF_dict)
        self.num_dofs = self.environment.env.GetRobots()[0].GetActiveDOF()
        self.joint_names = np.array([self.environment.env.GetRobots()[0].GetJointFromDOFIndex(i).GetName() for i in self.environment.env.GetRobots()[0].GetManipulator('arm').GetArmIndices()])
        # dof_values = np.array([self.environment.env.GetRobots()[0].GetJointFromDOFIndex(i).GetValue() for i in self.environment.env.GetRobots()[0].GetManipulator('arm').GetArmIndices()])
        # ds = np.array(dir(self.environment.env.GetRobots()[0]))
        # self.get_logger().info(f'dof: {np.array2string(ds)}')

        # dsv = np.array(self.environment.env.GetRobots()[0].GetActiveDOFValues())
        # self.get_logger().info(f'dof: {np.array2string(dsv)}')

        # self.get_logger().info(f'jn: {np.array2string(self.joint_names)}')
        

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

        print("Planning trajectory...")
        print("Start: ", self.start)
        print("Goal: ", self.goal)
        print("Goal Pose: ", self.goal_pose)
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

        self.cmd = np.eye(self.num_dofs)

        # Subscribe to the force-torque sensor data

        # Compliance parameters
        self.Kp = 1.0  # Stiffness (inverse of compliance)
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

        # Create a client for the ServoCommandType service
        self.switch_input_client = self.create_client(ServoCommandType, '/servo_node/switch_command_type')
        # Call the service to enable TWIST command type
        self.enable_twist_command()

    def new_plan_callback(self):
        # Update openrave model
        start = self.curr_pos
        self.environment.env.GetRobots()[0].SetActiveDOFValues(start)
        self.traj = self.planner.replan(start.squeeze(), self.goal, self.goal_pose, self.T, self.timestep)
        self.controller.set_trajectory(self.traj)

        # self.begin_motion_timer = self.create_timer(1.0, self.begin_motion_callback)
        self.can_move = True
        self.new_plan_timer = None


    def begin_motion_callback(self):
        self.begin_motion_callback = None
        self.can_move = True


    def enable_twist_command(self):
        if not self.switch_input_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Service not available, waiting again...')
            return

        request = ServoCommandType.Request()
        request.command_type = ServoCommandType.Request.TWIST

        future = self.switch_input_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None and future.result().success:
            self.get_logger().info('Switched to input type: TWIST')
        else:
            self.get_logger().warn('Could not switch input to: TWIST')

    def wrench_callback(self, msg):
        self.latest_wrench = msg

    def timer_callback(self):
        if self.latest_wrench is not None:
            try:
                # Look up the transformation from ft_frame to tool0 and then tool0 to base_link
                ft_to_tool0 = self.tf_buffer.lookup_transform('tool0', self.latest_wrench.header.frame_id, rclpy.time.Time())
                # tool0_to_base_link = self.tf_buffer.lookup_transform('base_link', 'tool0', rclpy.time.Time())

                # Transform the force/torque from ft_frame to tool0
                force = self.transform_vector(ft_to_tool0, self.latest_wrench.wrench.force)
                torque = self.transform_vector(ft_to_tool0, self.latest_wrench.wrench.torque)

                # Transform the force/torque from tool0 to base_link
                # force = self.transform_vector(tool0_to_base_link, force)
                # torque = self.transform_vector(tool0_to_base_link, torque)

                # Nullify force/torque readings with magnitude < 3
                force = self.nullify_small_magnitudes(force, 3.0)
                torque = self.nullify_small_magnitudes(torque, 3.0)

                self.prev_interaction_mode = self.interaction_mode
                if math.sqrt(force.x ** 2 + force.y ** 2 + force.z ** 2) < 3.0:
                    self.interaction_mode = False
                    if self.new_plan_timer is None and self.prev_interaction_mode != self.interaction_mode:
                        self.new_plan_timer = self.create_timer(1.0, self.new_plan_callback)
                    return
                self.interaction_mode = True
                self.can_move = False
                self.cmd = np.zeros((self.num_dofs, self.num_dofs))

                # Compute the twist in base_link frame
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

    def register_callbacks(self):
        """
        Set up all the subscribers and publishers needed.
        """
        self.traj_timer = self.create_timer(0.1, self.publish_trajectory)
        self.trajectory_pub = self.create_publisher(JointTrajectory, '/scaled_joint_trajectory_controller/joint_trajectory', 10)
        self.vel_trajectory_pub = self.create_publisher(JointTrajectory, '/scaled_vel_joint_trajectory_controller/joint_trajectory', 10)
        self.vel_pub = self.create_publisher(Float64MultiArray, '/forward_velocity_controller/commands', 10)
        self.joint_angles_sub = self.create_subscription(JointState, '/joint_states', self.joint_angles_callback, 10)
        self.force_torque_subscription = self.create_subscription(
            WrenchStamped,
            '/force_torque_sensor_broadcaster/wrench',
            self.wrench_callback,
            10)
        self.twist_pub_ = self.create_publisher(TwistStamped, '/servo_node/delta_twist_cmds', 10)

        # # Create a client for the ServoCommandType service
        # self.switch_input_client = self.create_client(ServoCommandType, '/servo_node/switch_command_type')
        # # Call the service to enable TWIST command type
        # self.enable_twist_command()

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
            # self.start = np.array(self.initial_joint_positions)
            # self.goal = np.array(self.initial_joint_positions)
            # self.goal[0] += 1.57
            # self.goal[2] += 1.57
            # self.get_logger().info(f'names: {np.array2string(self.joint_names)}')
            # self.get_logger().info(f'start: {np.array2string(self.start)}')
            # # self.get_logger().info(f'goal: {np.array2string(self.goal)}')
            # # traj_msg = Float64MultiArray()
            # # joint_vel = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0]
            # # traj_msg.data = joint_vel
            # # self.vel_pub.publish(traj_msg)
            # # assert(1 == 2)
            # # waypt_time = np.linspace(0.0, self.T, self.num_waypts)
            # waypt_time = [0.0, self.T]
            # self.traj = Trajectory([self.start, self.goal], waypt_time)
            # self.traj = self.traj.upsample(int(self.T/self.timestep)+1)
            # self.traj_plan = self.traj.downsample(self.num_waypts)
            # # for point in self.traj.waypts:
            # #     pt_str = np.array2string(point.reshape((1,-1)))
            # #     self.get_logger().info(f'p: {pt_str}')
            # # assert(1 == 2)
            # self.controller.set_trajectory(self.traj)

        curr_pos = np.roll(np.array(msg.position),1).reshape(self.num_dofs,1)
        curr_pos_str = np.array2string(curr_pos)
        # self.get_logger().info(f"curr_pos: {curr_pos_str}")

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
            # self.cmd = np.eye(self.num_dofs) * 0.0


    def publish_trajectory(self):
        if self.initial_joint_positions is None:
            return

        if not self.interaction_mode and self.can_move:
            joint_vel = np.array([self.cmd[i][i] for i in range(len(self.joint_names))]) 
            # self.get_logger().info(f'joint_vel: {np.array2string(joint_vel)}')

            # [shoulder_pan_joint, shoulder_lift_joint, elbow_joint, wrist_1_joint, wrist_2_joint, wrist_3_joint]
            # [shoulder_lift_joint, elbow_joint, wrist_1_joint, wrist_2_joint, wrist_3_joint, shoulder_pan_joint]
            # joint_vel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            # traj_msg = JointTrajectory()
            # traj_msg.joint_names = self.joint_names
            # point = JointTrajectoryPoint()
            # point.time_from_start.sec = 0
            # point.time_from_start.nanosec = int(1e8)  # 0.1 seconds
            

            # # Position Control
            # point.positions = self.curr_pos + (joint_vel * self.timestep).reshape(self.num_dofs,1)
            # traj_msg.points.append(point)
            # self.trajectory_pub.publish(traj_msg)

            # Velocity Control
            # point.velocities = joint_vel
            # point.positions = self.curr_pos
            # traj_msg.points.append(point)
            # self.vel_trajectory_pub.publish(traj_msg)

            
            # Float64MultiArray
            traj_msg = Float64MultiArray()
            traj_msg.data = joint_vel
            self.vel_pub.publish(traj_msg)

        

    # def run(self):
    #     rate = self.create_rate(100)

    #     while rclpy.ok():
    #         if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
    #             line = sys.stdin.readline()
    #             if line[0] == 'q' or line == 'Q' or line == 'quit':
    #                 break
    #         self.publish_trajectory()
    #         rate.sleep()

def main(args=None):
    rclpy.init(args=args)

    test_vel_node = TestVel()
    
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(test_vel_node)

    try:
        executor.spin()
    finally:
        test_vel_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
    