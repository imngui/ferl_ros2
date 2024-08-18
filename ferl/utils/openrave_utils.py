import openravepy
from openravepy import *
from openravepy import RaveCreateModule, RaveLoadPlugin

# from ferl.utils.bind import bind_subclass
# from prpy.bind import bind_subclass
# from archierobot import ArchieRobot
# from catkin.find_in_workspaces import find_in_workspaces
from ament_index_python.packages import get_package_share_directory, get_package_prefix
import numpy as np
from numpy import *
import logging
import math
import os

# Silence the planning logger to prevent spam.
logger = logging.getLogger('openrave_utils')
logging.basicConfig(filename='logs.log', level=logging.INFO)


# robot_starting_dofs = np.array([-1, 2, 0, 2, 0, 4, 0, 1.11022302e-16,  -1.11022302e-16, 3.33066907e-16])
robot_starting_dofs = np.array([0, 0, 0, 0, 0, 0, 0, 0])

def initialize(model_filename='gen3', envXML=None, viewer=True):
	'''
	Load and configure the GEN3 robot. If envXML is not None, loads environment.
	Returns robot and environment.
	-----
	NOTE:
	IF YOU JUST WANT TO DO COMPUTATIONS THROUGH OPENRAVE
	AND WANT MULTPILE INSTANCES TO OPEN, THEN HAVE TO TURN OFF
	QTCOIN AND ALL VIEWER FUNCTIONALITY OR IT WILL CRASH.
	'''
	env = openravepy.Environment()
	if envXML is not None:
		env.LoadURI(envXML)

	# Assumes the robot files are located in the data folder of the
	# kinova_description package in the catkin workspace.
	urdf_uri = os.path.join(get_package_share_directory('kortex_description'), 'robots', 'gen3.urdf')
	srdf_uri = os.path.join(get_package_share_directory('kinova_gen3_7dof_robotiq_2f_85_moveit_config'), 'config', 'gen3.srdf')

	found = RaveLoadPlugin(os.path.join(get_package_prefix('or_urdf'), 'lib', 'openrave-', 'or_urdf_plugin.so'))
	print("Found plugin: " + str(found))

	or_urdf = RaveCreateModule(env,'urdf_')

	created = or_urdf.SendCommand('LoadURI {:s} {:s}'.format(urdf_uri, srdf_uri))
	# print("Created robot:", created)
	if not created:
		raise Exception('Failed to load URDF and SRDF files.')

	robot = env.GetRobots()[0]
	robot.SetActiveManipulator('manipulator')
	manip = robot.GetManipulator('manipulator')
	active_dofs = np.arange(0, len(robot.GetJoints()))
	
	# Generate the ik solver
	# ikmodel = databases.inversekinematics.InverseKinematicsModel(robot, iktype=IkParameterization.Type.Transform6D)
	# if not ikmodel.load():
	# 	ikmodel.autogenerate()
  
	# print("Generated IK")
	# Generate a controller
	multicontroller = openravepy.RaveCreateMultiController(env, '')
	robot.SetController(multicontroller)
	controller = openravepy.RaveCreateController(env, 'IdealController')
	if controller is None:
		raise Exception('Controller creation failed.')
    
	multicontroller.AttachController(controller, manip.GetArmIndices(), 0)
 
	robot.SetActiveDOFs(active_dofs)
	robot.SetDOFValues(robot_starting_dofs)
 

	if viewer:
		env.SetViewer('qtosg')
		viewer = env.GetViewer()
		viewer.SetSize(700,500)
		cam_params = np.array([[-0.99885711, -0.01248719, -0.0461361 , -0.18887213],
                               [ 0.02495645,  0.68697757, -0.72624996,  2.04733515],
                               [ 0.04076329, -0.72657133, -0.68588079,  1.67818344],
                               [ 0.        ,  0.        ,  0.        ,  1.        ]])
		viewer.SetCamera(cam_params)

	return env, robot

# ------- Conversion Utils ------- #

def robotToCartesian(robot):
	"""
	Converts robot configuration into a list of cartesian
	(x,y,z) coordinates for each of the robot's links.
	------
	Returns: 7-dimensional list of 3 xyz values
	"""
	links = robot.GetLinks()
	cartesian = [None]*7
	i = 0
	for i in range(1,8):
		link = links[i]
		tf = link.GetTransform()
		cartesian[i-1] = tf[0:3,3]

	return cartesian

def robotToOrientation(robot):
	"""
	Converts robot configuration into a list of z-axis
	orientations for each of the robot's links.
	------
	Returns: 7-dimensional list of 3 xyz values
	"""
	links = robot.GetLinks()
	orientation = [None]*7
	i = 0
	for i in range(1,8):
		link = links[i]
		tf = link.GetTransform()
		orientation[i-1] = tf[:3,:3]

	return orientation

def manipToCartesian(robot, offset_z):
	"""
	Gets center of robot's manipulator in cartesian space
	------
	Params: robot object
			offset in m from the base of the manip to the center
	Returns: xyz of center of robot manipulator
	"""
	links = robot.GetLinks()
	manipTf = links[7].GetTransform()
	rot = manipTf[0:3,0:3]
	xyz = manipTf[0:3,3]
	offset = np.array([0,0,offset_z]).T
	return xyz

def poseToRobot(robot, pose):
	"""
	Returns Inverse Kinematics solution for given pose.
	------
	Params: robot object
			pose in xyz coordinates
	Returns: robot configuration space IK solution
	"""
	manip = robot.GetActiveManipulator()
	ikmodel = openravepy.databases.inversekinematics.InverseKinematicsModel(robot,iktype=openravepy.IkParameterization.Type.Translation3D)
	if not ikmodel.load():
		ikmodel.autogenerate()

	with robot: # lock environment and save robot state
		ikparam = openravepy.IkParameterization(pose,ikmodel.iktype) # build up the translation3d ik query
		sols = manip.FindIKSolutions(ikparam, openravepy.IkFilterOptions.CheckEnvCollisions) # get all solutions
		return sols[0]

def executePathSim(env,robot,waypts):
	"""
	Executes in the planned trajectory in simulation
	"""
	traj = openravepy.RaveCreateTrajectory(env,'')
	traj.Init(robot.GetActiveConfigurationSpecification())
	for i in range(len(waypts)):
		traj.Insert(i, waypts[i])
	robot.ExecutePath(traj)

# ------- Plotting Utils ------- #

def plotCupTraj(env,robot,bodies,waypts,color=[0,1,0], increment=1):
	"""
	Plots trajectory of the cup.
	"""
	for i in range(0,len(waypts),increment):
		waypoint = waypts[i]
		print("waypt: " +str(waypoint))
		dof = np.append(waypoint, np.array([1]))
		dof[2] += math.pi
		robot.SetDOFValues(dof)

		links = robot.GetLinks()
		manipTf = links[7].GetTransform() 

		# load mug into environment
		# objects_path = os.path.join(get_package_share_directory('iact_control'),'/data')
		objects_path = os.path.join(get_package_share_directory('ferl'), 'data')
		env.Load('{:s}/mug.xml'.format(objects_path))
		mug = env.GetKinBody('mug')
		mug.GetLinks()[0].GetGeometries()[0].SetDiffuseColor(np.array(color))
		angle = -np.pi/2

		rot_x = np.array([[1,0,0,0],[0,np.cos(angle),-np.sin(angle),0],[0,np.sin(angle),np.cos(angle),0],[0,0,0,1]]) 

		rot_y = np.array([[np.cos(angle),0,np.sin(angle),0],[0,1,0,0],[-np.sin(angle),0,np.cos(angle),0],[0,0,0,1]]) 

		rot_z = np.array([[np.cos(angle),-np.sin(angle),0,0],[np.sin(angle),0,np.cos(angle),0],[0,0,1,0],[0,0,0,1]]) 

		trans = np.array([[0,0,0,-0.02],[0,0,0,0.02],[0,0,0,-0.02],[0,0,0,1]]) 

		rotated = np.dot(manipTf+trans,rot_x)
		mug.SetTransform(rotated)

		body = mug
		body.SetName("pt"+str(len(bodies)))
		env.Add(body, True)
		bodies.append(body)

def plotMug(env, bodies, transform, color=[1,0,0]):
	"""
	Plots mug at specific transform.
	"""
	# objects_path = os.path.join(get_package_share_directory('iact_control'),'/data')
	objects_path = os.path.join(get_package_share_directory('ferl'), 'data')
	env.Load('{:s}/mug.xml'.format(objects_path))
	mug = env.GetKinBody('mug')
	mug.GetLinks()[0].GetGeometries()[0].SetDiffuseColor(np.array(color))
	mug.SetTransform(transform)
	body = mug
	body.SetName("pt"+str(len(bodies)))
	env.Add(body, True)
	bodies.append(body)
	return mug

def plotPoints(env,robot,bodies,waypts,colors):
	"""
	Plots a bunch of points with different colors.
	"""
	for i in range(len(waypts)):
		waypoint = waypts[i]
		dof = np.append(waypoint, np.array([1, 1, 1]))
		dof[2] += math.pi
		robot.SetDOFValues(dof)
		coord = robotToCartesian(robot)
		if np.linalg.norm(coord[6][:2]) < 0.45 and coord[6][2]<0.45:
			size = 0.005
		else:
			size = 0.015
		plotSphere(env, bodies, coord[6], size, colors[i])

def plotTraj(env,robot,bodies,waypts, size=10, color=[0, 1, 0]):
	"""
	Plots the best trajectory found or planned.
	"""
	for i in range(len(waypts)):
		waypoint = waypts[i]
		dof = np.append(waypoint, np.array([1]))
		dof[2] += math.pi
		robot.SetDOFValues(dof)
		coord = robotToCartesian(robot)
		plotSphere(env, bodies, coord[6], size, color)

def plotSphere(env, bodies, coords, size=10, color=[0, 0, 1]):
	"""
	Plots a single sphere in OpenRAVE center at coords(x,y,z) location.
	"""
	bodies.append(env.plot3(points=np.array((coords[0],coords[1],coords[2])), pointsize=size, colors=color, drawstyle=1))

def plotTable(env):
	"""
	Plots the robot table in OpenRAVE.
	"""
	# Load table into environment
	# objects_path = os.path.join(get_package_share_directory('iact_control'),'/data')
	objects_path = os.path.join(get_package_share_directory('ferl'), 'data')
	# table_path = os.path.join(get_package_share_directory('ferl'), 'data', 'objects', 'table.xml')
	# print("objects_path: ", objects_path)
	env.Load('{:s}/table.xml'.format(objects_path))
	# env.Load('{:s}'.format(table_path))
	table = env.GetKinBody('table')
	table.SetTransform(np.array([[0.0, 1.0,  0.0, -0.8128/2], #should be negative 1?
  								 [1.0, 0.0,  0.0, 0],
			                     [0.0, 0.0,  1.0, -0.1143], #-0.7874
			                     [0.0, 0.0,  0.0, 1.0]]))
	color = np.array([0.9, 0.75, 0.75])
	table.GetLinks()[0].GetGeometries()[0].SetDiffuseColor(color)

def plotCabinet(env):
	"""
	Plots the cabinet in OpenRAVE.
	"""
	# Load table into environment
	# objects_path = os.path.join(get_package_share_directory('iact_control'),'/data')
	objects_path = os.path.join(get_package_share_directory('ferl'), 'data')
	env.Load('{:s}/cabinet.xml'.format(objects_path))
	cabinet = env.GetKinBody('cabinet')
	cabinet.SetTransform(np.array([[0.0, -1.0,  0.0, 0.6],
  								 [1.0, 0.0,  0.0, 0],
			                     [0.0, 0.0,  1.0, 0],
			                     [0.0, 0.0,  0.0, 1.0]]))
	color = np.array([0.63,0.32,0.18])
	cabinet.GetLinks()[0].GetGeometries()[0].SetDiffuseColor(color)

def plotMug(env):
	# Load table into environment
    # objects_path = os.path.join(get_package_share_directory('iact_control'),'/data')
    objects_path = os.path.join(get_package_share_directory('ferl'), 'data')
    env.Load('{:s}/mug.xml'.format(objects_path))

def plotMan(env):
    """
    Plots a human in OpenRAVE.
    """
    # Load table into environment
    # objects_path = os.path.join(get_package_share_directory('iact_control'),'/data')
    objects_path = os.path.join(get_package_share_directory('ferl'), 'data')
    env.Load('{:s}/manifest.xml'.format(objects_path))

def plotTableMount(env,bodies):
	"""
	Plots the robot table mount in OpenRAVE.
	"""
	# Create robot base attachment
	body = openravepy.RaveCreateKinBody(env, '')
	body.InitFromBoxes(np.array([[0,0,0, 0.3048/2,0.8128/2,0.1016/2]]))
	body.SetTransform(np.array([[1.0, 0.0,  0.0, 0],
			                     [0.0, 1.0,  0.0, 0],
			                     [0.0, 0.0,  1.0, -0.1016/2],
			                     [0.0, 0.0,  0.0, 1.0]]))
	body.SetName("robot_mount")
	env.Add(body)

	color = np.array([0.9, 0.58, 0])
	body.GetLinks()[0].GetGeometries()[0].SetDiffuseColor(color)
	bodies.append(body)

def plotLaptop(env,bodies,pos):
	"""
	Plots the robot table mount in OpenRAVE.
	"""
	# Create robot base attachment
	body = openravepy.RaveCreateKinBody(env, '')
	#12 x 9 x 1 in, 0.3048 x 0.2286 x 0.0254 m
	# divide by 2: 0.1524 x 0.1143 x 0.0127
	#20 in from robot base
	body.InitFromBoxes(np.array([[0,0,0,0.1143,0.1524,0.0127]]))
	body.SetTransform(np.array([[1.0, 0.0,  0.0, pos[0]],
			                     [0.0, 1.0,  0.0, pos[1]],
			                     [0.0, 0.0,  1.0, pos[2]-0.1016],
			                     [0.0, 0.0,  0.0, 1.0]]))
	body.SetName("laptop")
	env.Add(body)
	color = np.array([0, 0, 0])
	body.GetLinks()[0].GetGeometries()[0].SetDiffuseColor(color)
	bodies.append(body)


