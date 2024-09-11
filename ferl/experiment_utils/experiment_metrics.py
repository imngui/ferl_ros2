import sys
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import re
import math
import os
import glob
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.append(os.path.join(os.path.abspath(os.getcwd()),".."))
sys.path.append(os.path.join(os.path.abspath(os.getcwd()),"../.."))
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '../..'))
import torch
import matplotlib.pyplot as plt

# import FERL Modules
from utils.learned_feature import LearnedFeature
from utils.environment import *
from utils.plot_utils import *

# Settings for the different cases
feat_list_cases = [["coffee","table", "laptop"],["coffee","laptop", "table"], ["coffee","table", "proxemics"]]
weights_cases = [[0.0, 10.0, 0.0], [0.0, 10.0, 0.0], [0.0, 10.0, 0.0]]
# known_features_cases = [["coffee", "table"], ["coffee", "laptop"], ["coffee", "table"]]
known_weights = [0., 0.]

p1 = [4.03901256, 5.51417794]
p2 = [0.0, 4.35964768, 4.88110989]
p3 = [0.0, 3.09983027, 5.1572305 ]

learned_weights_from_pushes_cases = [p1, p2, p3]

# traces_file_cases = ["laptop", "table", "proxemics"]
feature = "table"

# some settings for TrajOpt
FEAT_RANGE = {'table':0.98, 'coffee':1.0, 'laptop':0.3, 'human':0.3, 'efficiency':0.22, 'proxemics': 0.3, 'betweenobjects': 0.2, 'learned_feature':1.0}
obj_center_dict = {'HUMAN_CENTER': [-0.2, -0.5, 0.6], 'LAPTOP_CENTER': [-0.65, 0.0, 0.0]}
T = 20.0
timestep=0.5

# settings for the learned feature (27D Euclidean Feature Space)
# LF_dict = {'bet_data':5, 'sin':False, 'cos':False, 'rpy':False, 'lowdim':False, 'norot':True,
#            'noangles':True, '6D_laptop':False, '6D_human':False, '9D_coffee':False, 'EErot':False,
#            'noxyz':False, 'subspace_heuristic':False}
LF_dict = {'bet_data':5, 'sin':False, 'cos':False, 'rpy':False, 'lowdim':False, 'norot':True,
           'noangles':True, '6D_laptop':False, '6D_human':False, '9D_coffee':False, 'EErot':False,
           'noxyz':False, 'subspace_heuristic':False}

# Step 0: Create environment with known feature & initialize learned feature
env = Environment("ur5e", np.array([0.0, -1.5708, 0, -1.5708, 0, 0]), obj_center_dict, [feature],
                    [1.0], [0.0], viewer=False)

gt_env = Environment("ur5e", np.array([0.0, -1.5708, 0, -1.5708, 0, 0]), obj_center_dict, [feature],
                    [1.0], [0.0], viewer=False)

# env = Environment("gen3", np.array([0.0, -1.5708, 0, -1.5708, 0, 0]), obj_center_dict, known_features_cases[case-1],
#                     known_features_cases[case-1], known_weights, viewer=False)

# Step 1: load feature traces & initialize a learnable feature
unknown_feature = LearnedFeature(2, 64, LF_dict)

# for data_file in glob.glob(parent_dir + '/data/FERL_traces/traces_{}.p'.format(traces_file_cases[case-1])):
sim_trajectory_list = []
for data_file in glob.glob(parent_dir + '/data/demonstrations/sim_demo_{}.p'.format(feature)):
    sim_trajectory_list = pickle.load(open( data_file, "rb" ))

phys_trajectory_list = []
for data_file in glob.glob(parent_dir + '/data/demonstrations/phys_demo_{}.p'.format(feature)):
    phys_trajectory_list = pickle.load(open( data_file, "rb" ))

# all_trace_data = np.empty((0, 84), float)
# for idx in range(0, len(sim_trajectory_list)):
#     all_trace_data = np.vstack((all_trace_data, sim_trajectory_list[idx]))


sim_traj_list = np.array(sim_trajectory_list, dtype=object)
phys_traj_list = np.array(phys_trajectory_list, dtype=object)

mse_df = []
num_trials = 10
for n in range(2, 11):
    mses = []
    for trial in range(0, num_trials):
        sim_env = Environment("ur5e", np.array([0.0, -1.5708, 0, -1.5708, 0, 0]), obj_center_dict, [feature],
                    [1.0], [0.0], viewer=False)
        phys_env = Environment("ur5e", np.array([0.0, -1.5708, 0, -1.5708, 0, 0]), obj_center_dict, [feature],
                    [1.0], [0.0], viewer=False)
        sim_unknown_feature = LearnedFeature(2, 64, LF_dict)
        phys_unknown_feature = LearnedFeature(2, 64, LF_dict)
        np.random.shuffle(sim_traj_list)
        np.random.shuffle(phys_traj_list)
        sim_trace_data = np.empty((0, 84), float)
        phys_trace_data = np.empty((0, 84), float)
        for idx in range(0, n):
            sim_unknown_feature.add_data(sim_traj_list[idx])
            sim_trace_data = np.vstack((sim_trace_data, sim_trajectory_list[idx]))

            phys_unknown_feature.add_data(phys_traj_list[idx])
            phys_trace_data = np.vstack((phys_trace_data, phys_trajectory_list[idx]))
        
        # _ = unknown_feature.train(epochs=100, batch_size=32, learning_rate=1e-3, weight_decay=0.001, s_g_weight=10.)

        # Add learned feature to the environment
        sim_env.learned_features.append(sim_unknown_feature)
        sim_env.feature_list.append('learned_feature')
        sim_env.num_features += 1
        sim_env.feature_func_list.append(sim_unknown_feature.function)
        sim_env.weights = p1

        phys_env.learned_features.append(phys_unknown_feature)
        phys_env.feature_list.append('learned_feature')
        phys_env.num_features += 1
        phys_env.feature_func_list.append(phys_unknown_feature.function)
        phys_env.weights = p1

        raw_waypts, gt_cost = get_coords_gt_cost(gt_env, parent_dir)

        sim_feat_idx = list(np.arange(sim_env.num_features))
        sim_features = [[0.0 for _ in range(len(raw_waypts))] for _ in range(0, len(sim_env.feature_list))]
        for index in range(len(raw_waypts)):
            for feat in range(len(sim_feat_idx)):
                sim_features[feat][index] = sim_env.featurize_single(raw_waypts[index,:6], sim_feat_idx[feat])
        sim_features = np.array(sim_features).T
        sim_learned_cost = np.matmul(sim_features, np.array(sim_env.weights).reshape(-1, 1))

        sim_mse = (np.linalg.norm(sim_learned_cost - gt_cost)**2)/gt_cost.squeeze().shape[0]

        mse_df.append({"Demonstration Type": "RADER", "Num Features": n, "MSE": sim_mse})


        phys_feat_idx = list(np.arange(phys_env.num_features))
        phys_features = [[0.0 for _ in range(len(raw_waypts))] for _ in range(0, len(phys_env.feature_list))]
        for index in range(len(raw_waypts)):
            for feat in range(len(phys_feat_idx)):
                phys_features[feat][index] = phys_env.featurize_single(raw_waypts[index,:6], phys_feat_idx[feat])
        phys_features = np.array(phys_features).T
        phys_learned_cost = np.matmul(phys_features, np.array(phys_env.weights).reshape(-1, 1))

        phys_mse = (np.linalg.norm(phys_learned_cost - gt_cost)**2)/gt_cost.squeeze().shape[0]

        mse_df.append({"Demonstration Type": "Physical", "Num Features": n, "MSE": phys_mse})


df = pd.DataFrame(mse_df)

sns.barplot(df, x="Num Features", y="MSE", hue="Demonstration Type")
plt.show()


# 1.1 Visualize the Traces labeled at random with the initialized Network
# plot_learned_traj(unknown_feature.function, all_trace_data, env, feat=traces_file_cases[case-1])

# Step 2: train the feature on the set of traces
# torch.set_num_threads(4)
# _ = unknown_feature.train(epochs=100, batch_size=32, learning_rate=1e-3, weight_decay=0.001, s_g_weight=10.)


# # Step 3: Analyze the learned Feature

# # 3.1 Visualize the labeled Traces
# plot_learned_traj(unknown_feature.function, all_trace_data, env, feat=traces_file_cases[case-1])

# # 3.2 Visualize the learned function over the 3D Reachable Set
# plot_learned3D(parent_dir, unknown_feature.function, env, feat=traces_file_cases[case-1])

# # # Merge learned feature to others for overall cost/reward function

# # Add learned feature to the environment
# env.learned_features.append(unknown_feature)
# env.feature_list.append('learned_feature')
# env.num_features += 1
# env.feature_func_list.append(unknown_feature.function)

# # update with pushes data
# env.weights = learned_weights_from_pushes_cases[case-1]

# # plot GT
# gt_env = Environment("jaco_dynamics", np.array([0.0, -1.5708, 0, -1.5708, 0, 0]), obj_center_dict, feat_list_cases[case-1],
#                   gt_feat_range, weights_cases[case-1], viewer=False)
# # gt_env = Environment("jaco_dynamics", np.array([0.0, -1.5708, 0, -1.5708, 0, 0]), obj_center_dict, feat_list_cases[case-1],
# #                   feat_list_cases[case-1], weights_cases[case-1], viewer=False)
# plot_gt3D(parent_dir, gt_env)

# # Plot learned 3D cost function
# plot_gt3D(parent_dir, env, title='FERL Learned Cost over 3D Reachable Set')
