import rclpy
import kinova_msgs_ros2.msg
import geometry_msgs.msg
import kinova_msgs_ros2.srv
from kinova_msgs_ros2.srv import *
import time

def cmd_to_JointTorqueMsg(cmd):
	"""
	Returns a JointTorque Kinova msg from an array of torques
	"""
	jointCmd = kinova_msgs_ros2.msg.JointTorque()
	jointCmd.joint1 = cmd[0][0];
	jointCmd.joint2 = cmd[1][1];
	jointCmd.joint3 = cmd[2][2];
	jointCmd.joint4 = cmd[3][3];
	jointCmd.joint5 = cmd[4][4];
	jointCmd.joint6 = cmd[5][5];
	jointCmd.joint7 = cmd[6][6];
	
	return jointCmd

def cmd_to_JointVelocityMsg(cmd):
	"""
	Returns a JointVelocity Kinova msg from an array of velocities
	"""
	jointCmd = kinova_msgs_ros2.msg.JointVelocity()
	jointCmd.joint1 = cmd[0][0];
	jointCmd.joint2 = cmd[1][1];
	jointCmd.joint3 = cmd[2][2];
	jointCmd.joint4 = cmd[3][3];
	jointCmd.joint5 = cmd[4][4];
	jointCmd.joint6 = cmd[5][5];
	jointCmd.joint7 = cmd[6][6];

	return jointCmd

def waypts_to_PoseArrayMsg(cart_waypts):
	"""
	Returns a PoseArray msg from an array of 3D carteian waypoints
	"""
	poseArray = geometry_msgs.msg.PoseArray()
	# TODO: figure out what to use instead of rospy.Time.now()
	# poseArray.header.stamp = rospy.Time.now()
	poseArray.header.stamp = time.time()
	poseArray.header.frame_id = "/root"

	for i in range(len(cart_waypts)):
		somePose = geometry_msgs.msg.Pose()
		somePose.position.x = cart_waypts[i][0]
		somePose.position.y = cart_waypts[i][1]
		somePose.position.z = cart_waypts[i][2]

		somePose.orientation.x = 0.0
		somePose.orientation.y = 0.0
		somePose.orientation.z = 0.0
		somePose.orientation.w = 1.0
		poseArray.poses.append(somePose)

	return poseArray

def start_admittance_mode(prefix, node):
	"""
	Switches Kinova arm to admittance-control mode using ROS services.
	"""
	service_address = prefix+'/in/start_force_control'
	startForceControl = node.create_client(kinova_msgs_ros2.srv.Start, service_address)
	while not startForceControl.wait_for_service(timeout_sec=1.0):
		node.get_logger().info('service not available, waiting again...')
	resp = startForceControl.call_async(kinova_msgs_ros2.srv.Start.Request())
	rclpy.spin_until_future_complete(node, resp)
        
	# rospy.wait_for_service(service_address)
	# try:
	# 	startForceControl = rospy.ServiceProxy(service_address, Start)
	# 	startForceControl()
	# except rospy.ServiceException e:
	# 	print("Service call failed: %s"%e)
	# 	return None

def stop_admittance_mode(prefix, node):
    """
    Switches Kinova arm to position-control mode using ROS services.
    """
    service_address = prefix+'/in/stop_force_control'
    stopForceControl = node.create_client(kinova_msgs_ros2.srv.Stop, service_address)
    while not stopForceControl.wait_for_service(timeout_sec=1.0):
        node.get_logger().info('service not available, waiting again...')
    resp = stopForceControl.call_async(kinova_msgs_ros2.srv.Stop.Request())
    rclpy.spin_until_future_complete(node, resp)

    # rospy.wait_for_service(service_address)
    # try:
    #     stopForceControl = rospy.ServiceProxy(service_address, Stop)
    #     stopForceControl()
    # except rospy.ServiceException e:
    #     print("Service call failed: %s"%e)
    #     return None

