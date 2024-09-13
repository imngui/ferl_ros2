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
from utils.trajectory import *

# Settings for the different cases
# feat_list_cases = [["coffee","table", "laptop"],["coffee","laptop", "table"], ["coffee","table", "proxemics"]]
# weights_cases = [[0.0, 10.0, 0.0], [0.0, 10.0, 0.0], [0.0, 10.0, 0.0]]
# known_features_cases = [["coffee", "table"], ["coffee", "laptop"], ["coffee", "table"]]
known_weights = [0., 0.]

p1 = [4.03901256, 5.51417794]
p2 = [4.35964768, 4.88110989]
p3 = [3.09983027, 5.1572305 ]

learned_weights_from_pushes_cases = [p1, p2, p3]

# traces_file_cases = ["laptop", "table", "proxemics"]
feature = "coffee"

# some settings for TrajOpt
FEAT_RANGE = {'table':0.98, 'coffee':1.0, 'laptop':0.3, 'human':0.3, 'efficiency':0.22, 'proxemics': 0.3, 'betweenobjects': 0.2, 'learned_feature':1.0}
obj_center_dict = {'HUMAN_CENTER': [-0.2, -0.5, 0.6], 'LAPTOP_CENTER': [-0.65, 0.0, 0.0]}
T = 20.0
timestep=0.5

# settings for the learned feature (27D Euclidean Feature Space)
# LF_dict = {'bet_data':5, 'sin':False, 'cos':False, 'rpy':False, 'lowdim':False, 'norot':True,
#            'noangles':True, '6D_laptop':False, '6D_human':False, '9D_coffee':False, 'EErot':False,
#            'noxyz':False, 'subspace_heuristic':False}
LF_dict = {'bet_data':5, 'sin':False, 'cos':False, 'rpy':False, 'lowdim':False, 'norot':False,
           'noangles':True, '6D_laptop':False, '6D_human':False, '9D_coffee':True, 'EErot':False,
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
for data_file in glob.glob(parent_dir + '/data/demonstrations/sim_demo_{}_1.p'.format(feature)):
    sim_trajectory_list = pickle.load(open( data_file, "rb" ))

phys_trajectory_list = []
for data_file in glob.glob(parent_dir + '/data/demonstrations/phys_demo_{}.p'.format(feature)):
    phys_trajectory_list = pickle.load(open( data_file, "rb" ))

num_waypts = 40
ds_phys_trace_data = []
for idx in range(0, len(phys_trajectory_list)):
    ds_traj = phys_trajectory_list[idx][0::3]
    ds_phys_trace_data.append(ds_traj)
    

sim_traj_list = np.array(sim_trajectory_list, dtype=object)
phys_traj_list = np.array(ds_phys_trace_data, dtype=object)


################## GPT #######################################################
mse_df = []
num_trials = 10
for n in range(2, 11):
    mses = []
    for trial in range(0, num_trials):
        # Initialize environments
        sim_env = Environment("ur5e", np.array([0.0, -1.5708, 0, -1.5708, 0, 0]), obj_center_dict, [feature],
                              [1.0], [0.0], viewer=False)
        phys_env = Environment("ur5e", np.array([0.0, -1.5708, 0, -1.5708, 0, 0]), obj_center_dict, [feature],
                               [1.0], [0.0], viewer=False)
        
        # Initialize learned features
        sim_unknown_feature = LearnedFeature(2, 64, LF_dict)
        phys_unknown_feature = LearnedFeature(2, 64, LF_dict)
        np.random.shuffle(sim_traj_list)
        np.random.shuffle(phys_traj_list)
        
        # Combine traces
        sim_trace_data = np.empty((0, 84), float)
        phys_trace_data = np.empty((0, 84), float)
        for idx in range(0, n):
            sim_unknown_feature.add_data(sim_traj_list[idx])
            sim_trace_data = np.vstack((sim_trace_data, sim_trajectory_list[idx]))
            
            phys_unknown_feature.add_data(phys_traj_list[idx])
            phys_trace_data = np.vstack((phys_trace_data, ds_phys_trace_data[idx]))

        raw_waypts, gt_cost = get_coords_gt_cost(gt_env, parent_dir)

        # Train learned feature functions
        _ = sim_unknown_feature.train(epochs=100, batch_size=32, learning_rate=1e-3, weight_decay=0.001, s_g_weight=10.)
        _ = phys_unknown_feature.train(epochs=100, batch_size=32, learning_rate=1e-3, weight_decay=0.001, s_g_weight=10.)

        # Add learned feature to the environments
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

        # Compute learned feature MSE for simulated environment
        sim_features = np.array([[sim_unknown_feature.function(raw_waypts[i], norm=True) 
                                  for _ in range(len(sim_env.feature_list))] for i in range(len(raw_waypts))]).squeeze()
        print(sim_features.shape)
        sim_learned_cost = np.matmul(sim_features, np.array(sim_env.weights).reshape(-1, 1))
        sim_mse = (np.linalg.norm(sim_learned_cost - gt_cost)**2) / gt_cost.squeeze().shape[0]
        print('sim_mse: ', sim_mse)

        # Compute learned feature MSE for physical environment
        phys_features = np.array([[phys_unknown_feature.function(raw_waypts[i], norm=True) 
                                   for _ in range(len(phys_env.feature_list))] for i in range(len(raw_waypts))]).squeeze()
        phys_learned_cost = np.matmul(phys_features, np.array(phys_env.weights).reshape(-1, 1))
        phys_mse = (np.linalg.norm(phys_learned_cost - gt_cost)**2) / gt_cost.squeeze().shape[0]
        print('phys_mse: ', phys_mse)

        # Compute random feature function (MSE_random)
        # random_weights = np.random.rand(*np.array(phys_env.weights).shape)
        # random_learned_cost = np.matmul(phys_features, random_weights.reshape(-1, 1))
        # mse_random = (np.linalg.norm(random_learned_cost - gt_cost)**2) / gt_cost.squeeze().shape[0]
        # print('mse_random: ', mse_random)
        mse_random = 1.0

        # Compute MSE_norm for both simulations and physical demonstrations
        sim_mse_norm = sim_mse / mse_random
        phys_mse_norm = phys_mse / mse_random
        print('sim_mse_norm: ', sim_mse_norm)
        print('phys_mse_norm: ', phys_mse_norm)

        # Store the results in the dataframe
        mse_df.append({"Demonstration Type": "RADER", "Num Features": n, "MSE": sim_mse_norm})
        mse_df.append({"Demonstration Type": "Physical", "Num Features": n, "MSE": phys_mse_norm})

# Convert results to a DataFrame
df = pd.DataFrame(mse_df)

df.to_csv('laptop.csv')

# Plot results using Seaborn
sns.barplot(df, x="Num Features", y="MSE", hue="Demonstration Type")
plt.show()

############# OURS ###########################################################

# mse_df = []
# num_trials = 10
# for n in range(2, 11):
#     mses = []
#     for trial in range(0, num_trials):
#         sim_env = Environment("ur5e", np.array([0.0, -1.5708, 0, -1.5708, 0, 0]), obj_center_dict, [feature],
#                     [1.0], [0.0], viewer=False)
#         phys_env = Environment("ur5e", np.array([0.0, -1.5708, 0, -1.5708, 0, 0]), obj_center_dict, [feature],
#                     [1.0], [0.0], viewer=False)
#         sim_unknown_feature = LearnedFeature(2, 64, LF_dict)
#         phys_unknown_feature = LearnedFeature(2, 64, LF_dict)
#         np.random.shuffle(sim_traj_list)
#         np.random.shuffle(phys_traj_list)
#         sim_trace_data = np.empty((0, 84), float)
#         phys_trace_data = np.empty((0, 84), float)
#         for idx in range(0, n):
#             sim_unknown_feature.add_data(sim_traj_list[idx])
#             sim_trace_data = np.vstack((sim_trace_data, sim_trajectory_list[idx]))

            
#             phys_unknown_feature.add_data(phys_traj_list[idx])
#             phys_trace_data = np.vstack((phys_trace_data, ds_phys_trace_data[idx]))

#         raw_waypts, gt_cost = get_coords_gt_cost(gt_env, parent_dir)

#         # Add learned feature to the environment
#         # sim_env.learned_features = [sim_unknown_feature]
#         # sim_env.feature_func_list = [sim_unknown_feature.function]

#         # phys_env.learned_features = [phys_unknown_feature]
#         # phys_env.feature_func_list = [phys_unknown_feature.function]

#         # sim_feat_idx = list(np.arange(sim_env.num_features))
#         # sim_features = [[0.0 for _ in range(len(raw_waypts))] for _ in range(0, len(sim_env.feature_list))]
#         # for index in range(len(raw_waypts)):
#         #     for feat in range(len(sim_feat_idx)):
#         #         sim_features[feat][index] = sim_unknown_feature.function(raw_waypts[index])
#         #         # sim_features[feat][index] = sim_env.featurize_single(raw_waypts[index,:6], sim_feat_idx[feat])
#         # sim_features = np.array(sim_features).T
#         # sim_learned_cost = np.matmul(sim_features, np.array(sim_env.weights).reshape(-1, 1))

#         # sim_mse_rand = (np.linalg.norm(sim_learned_cost - gt_cost)**2)/gt_cost.squeeze().shape[0]
#         # print('sim_mse_rand: ', sim_mse_rand)
#         # sim_mse_rand = 1.0

#         # phys_feat_idx = list(np.arange(phys_env.num_features))
#         # phys_features = [[0.0 for _ in range(len(raw_waypts))] for _ in range(0, len(phys_env.feature_list))]
#         # for index in range(len(raw_waypts)):
#         #     for feat in range(len(phys_feat_idx)):
#         #         phys_features[feat][index] = phys_unknown_feature.function(raw_waypts[index])
#         #         # phys_features[feat][index] = phys_env.featurize_single(raw_waypts[index,:6], phys_feat_idx[feat])
#         # phys_features = np.array(phys_features).T
#         # phys_learned_cost = np.matmul(phys_features, np.array(phys_env.weights).reshape(-1, 1))

#         # phys_mse_rand = (np.linalg.norm(phys_learned_cost - gt_cost)**2)/gt_cost.squeeze().shape[0]
#         # print('phys_mse_rand: ', phys_mse_rand)
#         # phys_mse_rand = 1.0

#         _ = sim_unknown_feature.train(epochs=100, batch_size=32, learning_rate=1e-3, weight_decay=0.001, s_g_weight=10.)
#         _ = phys_unknown_feature.train(epochs=100, batch_size=32, learning_rate=1e-3, weight_decay=0.001, s_g_weight=10.)

#         # Add learned feature to the environment
#         sim_env.learned_features.append(sim_unknown_feature)
#         sim_env.feature_list.append('learned_feature')
#         sim_env.num_features += 1
#         sim_env.feature_func_list.append(sim_unknown_feature.function)
#         sim_env.weights = p1

#         phys_env.learned_features.append(phys_unknown_feature)
#         phys_env.feature_list.append('learned_feature')
#         phys_env.num_features += 1
#         phys_env.feature_func_list.append(phys_unknown_feature.function)
#         phys_env.weights = p1

#         sim_feat_idx = list(np.arange(sim_env.num_features))
#         sim_features = [[0.0 for _ in range(len(raw_waypts))] for _ in range(0, len(sim_env.feature_list))]
#         for index in range(len(raw_waypts)):
#             for feat in range(len(sim_feat_idx)):
#                 sim_features[feat][index] = sim_unknown_feature.function(raw_waypts[index], norm=True)
#                 # sim_features[feat][index] = sim_env.featurize_single(raw_waypts[index,:6], sim_feat_idx[feat])
#         sim_features = np.array(sim_features).T
#         sim_learned_cost = np.matmul(sim_features, np.array(sim_env.weights).reshape(-1, 1))
#         sim_mse = (np.linalg.norm(sim_learned_cost - gt_cost)**2)/gt_cost.squeeze().shape[0]

#         phys_feat_idx = list(np.arange(phys_env.num_features))
#         phys_features = [[0.0 for _ in range(len(raw_waypts))] for _ in range(0, len(phys_env.feature_list))]
#         for index in range(len(raw_waypts)):
#             for feat in range(len(phys_feat_idx)):
#                 phys_features[feat][index] = phys_unknown_feature.function(raw_waypts[index], norm=True)
#                 # phys_features[feat][index] = phys_env.featurize_single(raw_waypts[index,:6], phys_feat_idx[feat])
#         phys_features = np.array(phys_features).T
#         phys_learned_cost = np.matmul(phys_features, np.array(phys_env.weights).reshape(-1, 1))
#         phys_mse = (np.linalg.norm(phys_learned_cost - gt_cost)**2)/gt_cost.squeeze().shape[0]

#         mse_df.append({"Demonstration Type": "RADER", "Num Features": n, "MSE": (sim_mse)})
#         mse_df.append({"Demonstration Type": "Physical", "Num Features": n, "MSE": (phys_mse)})


# df = pd.DataFrame(mse_df)

# sns.barplot(df, x="Num Features", y="MSE", hue="Demonstration Type")
# plt.show()
