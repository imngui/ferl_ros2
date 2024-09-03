import torch
import os
from enum import Enum

import openravepy
from openravepy import *

from ferl.utils.openrave_utils import *
from ferl.utils.learned_feature import LearnedFeature

from rclpy.impl import rcutils_logger
logger = rcutils_logger.RcutilsLogger(name="env")


# TODO ! Figure out EE_Link stuff

class Environment(object):
    """
    This class creates an OpenRave environment and contains all the
    functionality needed for custom features and constraints.
    """
    def __init__(self, model_filename, start_dofs, object_centers, feat_list=None, feat_range=None, feat_weights=None, LF_dict=None, viewer=True):
        # ---- Create environment ---- #
        self.env, self.robot = initialize(model_filename, start_dofs, viewer=viewer)
        self.num_dofs = self.robot.GetActiveDOF()

        # Insert any objects you want into environment.
        self.bodies = []
        self.object_centers = object_centers
        plotTable(self.env)
        plotTableMount(self.env, self.bodies)
        plotCabinet(self.env)
        for center in object_centers.keys():
            if center == "LAPTOP_CENTER":
                plotLaptop(self.env, self.bodies, object_centers[center])
            else:
                plotSphere(self.env, self.bodies, object_centers[center], 0.015)

        # Create the initial feature function list.
        self.feature_func_list = []
        self.feature_list = feat_list
        self.num_features = len(self.feature_list)
        self.feat_range = feat_range
        for feat in self.feature_list:
            if feat == 'table':
                self.feature_func_list.append(self.table_features)
            elif feat == 'coffee':
                self.feature_func_list.append(self.coffee_features)
            elif feat == 'human':
                self.feature_func_list.append(self.human_features)
            elif feat == 'laptop':
                self.feature_func_list.append(self.laptop_features)
            elif feat == 'origin':
                self.feature_func_list.append(self.origin_features)
            elif feat == 'efficiency':
                self.feature_func_list.append(self.efficiency_features)
            elif feat == 'proxemics':
                self.feature_func_list.append(self.proxemics_features)
            elif feat == 'betweenobjects':
                self.feature_func_list.append(self.betweenobjects_features)

        # Create a list of learned features.
        self.learned_features = []

        # Initialize the utility function weight vector.
        self.weights = feat_weights

        # Initialize LF_dict optionally for learned features.
        self.LF_dict = LF_dict


    # -- Compute features for all waypoints in trajectory. -- #
    def featurize(self, waypts, feat_idx=None):
        """
        Computes the features for a given trajectory.
        ---
        Params:
            waypts -- trajectory waypoints
            feat_idx -- list of feature indices (optional)
        Returns:
            features -- list of feature values (T x num_features)
        """
        # if no list of idx is provided use all of them
        if feat_idx is None:
            feat_idx = list(np.arange(self.num_features))

        features = [[0.0 for _ in range(len(waypts)-1)] for _ in range(0, len(feat_idx))]
        for index in range(len(waypts)-1):
            for feat in range(len(feat_idx)):
                waypt = waypts[index+1]
                if self.feature_list[feat_idx[feat]] == 'efficiency':
                    waypt = np.concatenate((waypts[index+1],waypts[index]))
                features[feat][index] = self.featurize_single(waypt, feat_idx[feat])
        return features

    # -- Compute single feature for single waypoint -- #
    def featurize_single(self, waypt, feat_idx):
        """
        Computes given feature value for a given waypoint.
        ---
        Params:
            waypt -- single waypoint
            feat_idx -- feature index
        Returns:
            featval -- feature value
        """
        # If it's a learned feature, feed in raw_features to the NN.
        if self.feature_list[feat_idx] == 'learned_feature':
            waypt = self.raw_features(waypt)
        # Compute feature value.
        featval = self.feature_func_list[feat_idx](waypt)
        if self.feature_list[feat_idx] == 'learned_feature':
            featval = featval[0][0]
        else:
            if self.feat_range is not None:
                featval /= self.feat_range[feat_idx]
        return featval

    # -- Return raw features -- #
    def raw_features(self, waypt):
        """
        Computes raw state space features for a given waypoint.
        ---
        Params:
            waypt -- single waypoint
        Returns:
            raw_features -- list of raw feature values
        """
        object_coords = np.array([self.object_centers[x] for x in self.object_centers.keys()])
        if torch.is_tensor(waypt):
            Tall = self.get_torch_transforms(waypt)
            coords = Tall[:,:3,3]
            orientations = Tall[:,:3,:3]
            object_coords = torch.from_numpy(object_coords)
            # logger.info(f't waypt: {waypt.squeeze().shape}')
            # logger.info(f't orien: {orientations.flatten().shape}')
            # logger.info(f't coord: {coords.flatten().shape}')
            # logger.info(f't ocoor: {object_coords.flatten().shape}')
            return torch.reshape(torch.cat((waypt.squeeze(), orientations.flatten(), coords.flatten(), object_coords.flatten())), (-1,))
        else:
            # if len(waypt) < 10:
            #     waypt_openrave = np.append(waypt.reshape(self.num_dofs), np.array([0]))
            #     waypt_openrave[2] += math.pi

            waypt_openrave = waypt
            self.robot.SetActiveDOFValues(waypt_openrave)
            # self.robot.SetDOFValues(waypt_openrave)
            coords = np.array(robotToCartesian(self.robot))
            orientations = np.array(robotToOrientation(self.robot))
            # logger.info(f'waypt: {waypt.squeeze().shape}')
            # logger.info(f'orien: {orientations.flatten().shape}')
            # logger.info(f'coord: {coords.flatten().shape}')
            # logger.info(f'ocoor: {object_coords.flatten().shape}')
            # temp = np.reshape(np.concatenate((waypt.squeeze(), orientations.flatten(), coords.flatten(), object_coords.flatten())), (-1,))
            # logger.info(f'raw len: {temp.shape}')
            return np.reshape(np.concatenate((waypt.squeeze(), orientations.flatten(), coords.flatten(), object_coords.flatten())), (-1,))

    def get_torch_transforms(self, waypt):
        """
        Computes torch transforms for given waypoint.
        ---
        Params:
            waypt -- single waypoint
        Returns:
            Tall -- Transform in torch for every joint (7D)
        """
        def transform(theta, alpha, a, d):
            T = torch.eye(4)
            T[0][0] = torch.cos(theta)
            T[0][1] = -torch.sin(theta) * torch.cos(alpha)
            T[0][2] = torch.sin(theta) * torch.sin(alpha)
            T[0][3] = a * torch.cos(theta)
            T[1][0] = torch.sin(theta)
            T[1][1] = torch.cos(theta) * torch.cos(alpha)
            T[1][2] = -torch.cos(theta) * torch.sin(alpha)
            T[1][3] = a * torch.sin(theta)
            T[2][1] = torch.sin(alpha)
            T[2][2] = torch.cos(alpha)
            T[2][3] = d
            return T

        # DH parameters for UR5e as given in the image
        a = torch.tensor([0, -0.425, -0.3922, 0, 0.0, 0], requires_grad=True)
        d = torch.tensor([0.1625, 0, 0, 0.1333, 0.0997, 0.0996], requires_grad=True)
        alpha = torch.tensor([torch.pi/2, 0, 0, torch.pi/2, -torch.pi/2, 0], requires_grad=True)

        Tall = torch.eye(4).unsqueeze(0)  # Start with identity matrix for the base frame

        T_prev = Tall[0]

        for i in range(6):
            Ti = transform(waypt[i], alpha[i], a[i], d[i])
            T_curr = torch.matmul(T_prev, Ti)
            Tall = torch.cat((Tall, T_curr.unsqueeze(0)))
            T_prev = T_curr

        return Tall

    # -- Instantiate a new learned feature -- #

    def new_learned_feature(self, nb_layers, nb_units, checkpoint_name=None):
        """
        Adds a new learned feature to the environment.
        --
        Params:
            nb_layers -- number of NN layers
            nb_units -- number of NN units per layer
            checkpoint_name -- name of NN model to load (optional)
        """
        self.learned_features.append(LearnedFeature(nb_layers, nb_units, self.LF_dict))
        self.feature_list.append('learned_feature')
        self.num_features += 1
        # initialize new feature weight with zero
        self.weights = np.hstack((self.weights, np.zeros((1, ))))

        # If we can, load a model instead of a blank feature.
        if checkpoint_name is not None:
            here = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../../'))
            self.learned_features[-1] = torch.load(here+'/data/final_models/' + checkpoint_name)

        self.feature_func_list.append(self.learned_features[-1].function)

    # -- Efficiency -- #

    def efficiency_features(self, waypt):
        """
        Computes efficiency feature for waypoint, confirmed to match trajopt.
        ---
        Params:
            waypt -- single waypoint
        Returns:
            dist -- scalar feature
        """

        return np.linalg.norm(waypt[:self.num_dofs] - waypt[self.num_dofs:])**2

    # -- Distance to Robot Base (origin of world) -- #

    def origin_features(self, waypt):
        """
        Computes the total feature value over waypoints based on 
        y-axis distance to table.
        ---
        Params:
            waypt -- single waypoint
        Returns:
            dist -- scalar feature
        """
        # if len(waypt) < 8:
        #     waypt = np.append(waypt.reshape(self.num_dofs), np.array([0]))
        #     waypt[2] += math.pi
        # self.robot.SetDOFValues(waypt)
        self.robot.SetActiveDOFValues(waypt)
        coords = robotToCartesian(self.robot)
        # EEcoord_y = coords[6][1]
        # EEcoord_y = np.linalg.norm(coords[6])
        EEcoord_y = coords[-1][1]
        EEcoord_y = np.linalg.norm(coords[-1])
        return EEcoord_y

    # -- Distance to Table -- #

    def table_features(self, waypt, prev_waypt=None):
        """
        Computes the total feature value over waypoints based on 
        z-axis distance to table.
        ---
        Params:
            waypt -- single waypoint
        Returns:
            dist -- scalar feature
        """
        # if len(waypt) < 8:
        #     waypt = np.append(waypt.reshape(self.num_dofs), np.array([0]))
        #     waypt[2] += math.pi
        # self.robot.SetDOFValues(waypt)
        self.robot.SetActiveDOFValues(waypt)
        coords = robotToCartesian(self.robot)
        # EEcoord_z = coords[6][2]
        EEcoord_z = coords[-1][2]
        return EEcoord_z

    # -- Coffee (or z-orientation of end-effector) -- #

    def coffee_features(self, waypt):
        """
        Computes the coffee orientation feature value for waypoint
        by checking if the EE is oriented vertically.
        ---
        Params:
            waypt -- single waypoint
        Returns:
            dist -- scalar feature
        """
        # if len(waypt) < 8:
        #     waypt = np.append(waypt.reshape(self.num_dofs), np.array([0]))
        #     waypt[2] += math.pi

        # self.robot.SetDOFValues(waypt)
        self.robot.SetActiveDOFValues(waypt)
        # TODO How do we know/get the ee link?
        # EE_link = self.robot.GetLinks()[7]
        EE_link = self.robot.GetLink('tool0')
        Rx = EE_link.GetTransform()[:3,0]
        return 1 - EE_link.GetTransform()[:3,0].dot([0,0,1])

    # -- Distance to Laptop -- #

    def laptop_features(self, waypt):
        """
        Computes distance from end-effector to laptop in xy coords
        Params:
            waypt -- single waypoint
        Returns:
            dist -- scalar distance where
                0: EE is at more than 0.3 meters away from laptop
                +: EE is closer than 0.3 meters to laptop
        """
        # if len(waypt) < 8:
        #     waypt = np.append(waypt.reshape(self.num_dofs), np.array([0]))
        #     waypt[2] += math.pi
        # self.robot.SetDOFValues(waypt)
        self.robot.SetActiveDOFValues(waypt)
        coords = robotToCartesian(self.robot)
        # EE_coord_xy = coords[6][0:2]
        EE_coord_xy = coords[-1][0:2]
        laptop_xy = np.array(self.object_centers['LAPTOP_CENTER'][0:2])
        dist = np.linalg.norm(EE_coord_xy - laptop_xy) - 0.3
        if dist > 0:
            return 0
        return -dist

    # -- Distance to Human -- #

    def human_features(self, waypt):
        """
        Computes distance from end-effector to human in xy coords
        Params:
            waypt -- single waypoint
        Returns:
            dist -- scalar distance where
                0: EE is at more than 0.4 meters away from human
                +: EE is closer than 0.4 meters to human
        """
        # if len(waypt) < 8:
        #     waypt = np.append(waypt.reshape(self.num_dofs), np.array([0]))
        #     waypt[2] += math.pi
        # self.robot.SetDOFValues(waypt)
        self.robot.SetActiveDOFValues(waypt)
        coords = robotToCartesian(self.robot)
        # EE_coord_xy = coords[6][0:2]
        EE_coord_xy = coords[-1][0:2]
        human_xy = np.array(self.object_centers['HUMAN_CENTER'][0:2])
        dist = np.linalg.norm(EE_coord_xy - human_xy) - 0.4
        if dist > 0:
            return 0
        return -dist

    # -- Human Proxemics -- #

    def proxemics_features(self, waypt):
        """
        Computes distance from end-effector to human proxemics in xy coords
        Params:
            waypt -- single waypoint
        Returns:
            dist -- scalar distance where
                0: EE is at more than 0.3 meters away from human
                +: EE is closer than 0.3 meters to human
        """
        # if len(waypt) < 8:
        #     waypt = np.append(waypt.reshape(self.num_dofs), np.array([0]))
        #     waypt[2] += math.pi
        # self.robot.SetDOFValues(waypt)
        self.robot.SetActiveDOFValues(waypt)
        coords = robotToCartesian(self.robot)
        # EE_coord_xy = coords[6][0:2]
        EE_coord_xy = coords[-1][0:2]
        human_xy = np.array(self.object_centers['HUMAN_CENTER'][0:2])
        # Modify ellipsis distance.
        EE_coord_xy[1] /= 3
        human_xy[1] /= 3
        dist = np.linalg.norm(EE_coord_xy - human_xy) - 0.3
        if dist > 0:
            return 0
        return -dist

    # -- Between 2-objects -- #

    def betweenobjects_features(self, waypt):
        """
        Computes distance from end-effector to 2 objects in xy coords.
        Params:
            waypt -- single waypoint
        Returns:
            dist -- scalar distance where
                0: EE is at more than 0.2 meters away from the objects and between
                +: EE is closer than 0.2 meters to the objects and between
        """
        # if len(waypt) < 8:
        #     waypt = np.append(waypt.reshape(self.num_dofs), np.array([0]))
        #     waypt[2] += math.pi
        # self.robot.SetDOFValues(waypt)
        self.robot.SetActiveDOFValues(waypt)
        coords = robotToCartesian(self.robot)
        # EE_coord_xy = coords[6][0:2]
        EE_coord_xy = coords[-1][0:2]
        object1_xy = np.array(self.object_centers['OBJECT1'][0:2])
        object2_xy = np.array(self.object_centers['OBJECT2'][0:2])

        # Determine where the point lies with respect to the segment between the two objects.
        o1EE = np.linalg.norm(object1_xy - EE_coord_xy)
        o2EE = np.linalg.norm(object2_xy - EE_coord_xy)
        o1o2 = np.linalg.norm(object1_xy - object2_xy)
        o1angle = np.arccos((o1EE**2 + o1o2**2 - o2EE**2) / (2*o1o2*o1EE))
        o2angle = np.arccos((o2EE**2 + o1o2**2 - o1EE**2) / (2*o1o2*o2EE))

        dist1 = 0
        if o1angle < np.pi/2 and o2angle < np.pi/2:
            dist1 = np.linalg.norm(np.cross(object2_xy - object1_xy, object1_xy - EE_coord_xy)) / o1o2 - 0.2
        dist1 = 0.8*dist1 # control how much less it is to go between the objects versus on top of them
        dist2 = min(np.linalg.norm(object1_xy - EE_coord_xy), np.linalg.norm(object2_xy - EE_coord_xy)) - 0.2

        if dist1 > 0 and dist2 > 0:
            return 0
        elif dist2 > 0:
            return -dist1
        elif dist1 > 0:
            return -dist2
        return -min(dist1, dist2)

    # ---- Custom environmental constraints --- #

    def table_constraint(self, waypt):
        """
        Constrains z-axis of robot's end-effector to always be above the table.
        """
        # if len(waypt) < 10:
        #     waypt = np.append(waypt.reshape(self.num_dofs), np.array([0]))
        #     waypt[2] += math.pi
        # self.robot.SetDOFValues(waypt)
        self.robot.SetActiveDOFValues(waypt)
        # EE_link = self.robot.GetLinks()[8]
        EE_link = self.robot.GetLink('tool0')
        EE_coord_z = EE_link.GetTransform()[2][3]
        if EE_coord_z > -0.1016:
            return 0
        return 10000

    def coffee_constraint(self, waypt):
        """
        Constrains orientation of robot's end-effector to be holding coffee mug upright.
        """
        # if len(waypt) < 10:
        #     waypt = np.append(waypt.reshape(self.num_dofs), np.array([0]))
        #     waypt[2] += math.pi
        # self.robot.SetDOFValues(waypt)
        self.robot.SetActiveDOFValues(waypt)
        # TODO How do we get/know the ee link?
        # EE_link = self.robot.GetLinks()[7]
        EE_link = self.robot.GetLink('tool0')
        return EE_link.GetTransform()[:2,:3].dot([1,0,0])

    def coffee_constraint_derivative(self, waypt):
        """
        Analytic derivative for coffee constraint.
        """
        # if len(waypt) < 10:
        #     waypt = np.append(waypt.reshape(self.num_dofs), np.array([0]))
        #     waypt[2] += math.pi
        # self.robot.SetDOFValues(waypt)
        self.robot.SetActiveDOFValues(waypt)
        # world_dir = self.robot.GetLinks()[7].GetTransform()[:3,:3].dot([1,0,0])
        world_dir = EE_link = self.robot.GetLink('tool0').GetTransform()[:3,:3].dot([1,0,0])
        return np.array([np.cross(self.robot.GetJoints()[i].GetAxis(), world_dir)[:2] for i in range(self.num_dofs)]).T.copy()

    # ---- Helper functions ---- #

    def update_curr_pos(self, curr_pos):
        """
        Updates DOF values in OpenRAVE simulation based on curr_pos.
        ----
        curr_pos - 7x1 vector of current joint angles (degrees)
        """
        # TODO Convert to num_dofs dim
        # pos = np.array([curr_pos[0][0],curr_pos[1][0],curr_pos[2][0]+math.pi,curr_pos[3][0],curr_pos[4][0],curr_pos[5][0],curr_pos[6][0],0,0,0])
        pos = np.array([curr_pos[0][0],curr_pos[1][0],curr_pos[2][0]+math.pi,curr_pos[3][0],curr_pos[4][0],curr_pos[5][0]])

        # self.robot.SetDOFValues(pos)
        self.robot.SetActiveDOFValues(pos)

    def kill_environment(self):
        """
        Destroys openrave thread and environment for clean shutdown.
        """
        self.env.Destroy()
        RaveDestroy() # destroy the runtime
