# %%
import sys
import numpy as np
import pandas as pd
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

# %%
# import FERL Modules
from utils.learned_feature import LearnedFeature
from utils.environment import *
from utils.plot_utils import *

# %%
# Settings for the different cases
feat_list_cases = [["coffee","table", "laptop"],["coffee","laptop", "table"], ["coffee","table", "proxemics"]]
weights_cases = [[0.0, 10.0, 0.0], [0.0, 10.0, 0.0], [0.0, 10.0, 0.0]]
known_features_cases = [["coffee", "table"], ["coffee", "laptop"], ["coffee", "table"]]
# known_features_cases = [["coffee", "proxemics"], ["coffee", "proxemics"], ["coffee", "proxemics"]]
known_weights = [0., 0.]

traces_file_cases = ["laptop", "table", "proxemics"]
# traces_file_cases = ["table"]
traces_idx = np.arange(10).tolist()

# learned weights from pushes
p1 = [0.0, 4.03901256, 5.51417794]
p2 = [0.0, 4.35964768, 4.88110989]
p3 = [0.0, 3.09983027, 5.1572305 ]

learned_weights_from_pushes_cases = [p1, p2, p3]

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

# %% [markdown]
# # Learn Laptop Feature from collected feature traces

# %%
# Setting for which Case (see paper)
case = 1

# some derivative data
feat_range = [FEAT_RANGE[known_features_cases[case-1][feat]] for feat in range(len(known_features_cases[case-1]))]
gt_feat_range = [FEAT_RANGE[feat_list_cases[case-1][feat]] for feat in range(len(feat_list_cases[case-1]))]

# %%
# Step 0: Create environment with known feature & initialize learned feature
env = Environment("gen3", np.array([0.0, -1.5708, 0, -1.5708, 0, 0]), obj_center_dict, known_features_cases[case-1],
                    feat_range, known_weights, viewer=False)

# env = Environment("gen3", np.array([0.0, -1.5708, 0, -1.5708, 0, 0]), obj_center_dict, known_features_cases[case-1],
#                     known_features_cases[case-1], known_weights, viewer=False)

# Step 1: load feature traces & initialize a learnable feature
unknown_feature = LearnedFeature(2, 64, LF_dict)

# for data_file in glob.glob(parent_dir + '/data/FERL_traces/traces_{}.p'.format(traces_file_cases[case-1])):
trajectory_list = []
for data_file in glob.glob(parent_dir + '/data/demonstrations/sim_demo_{}.p'.format(traces_file_cases[case-1])):
    # trajectory_list.extend(pickle.load(open( data_file, "rb" )))
    trajectory_list = pickle.load(open( data_file, "rb" ))

# for data_file in glob.glob(parent_dir + '/data/demonstrations/demo_0_table.p'):
#     trajectory_list.extend(pickle.load(open( data_file, "rb" )))

# for data_file in glob.glob(parent_dir + '/data/demonstrations/demo_1_table.p'):
#     trajectory_list.extend(pickle.load(open( data_file, "rb" )))

# for data_file in glob.glob(parent_dir + '/data/demonstrations/demo_0_laptop.p'):
#     trajectory_list.extend(pickle.load(open( data_file, "rb" )))

# for data_file in glob.glob(parent_dir + '/data/demonstrations/demo_1_laptop.p'):
#     trajectory_list.extend(pickle.load(open( data_file, "rb" )))

# for data_file in glob.glob(parent_dir + '/data/demonstrations/demo_2_laptop.p'):
#     trajectory_list.extend(pickle.load(open( data_file, "rb" )))

# for data_file in glob.glob(parent_dir + '/data/demonstrations/demo_7_laptop.p'):
#     trajectory_list.extend(pickle.load(open( data_file, "rb" )))

all_trace_data = np.empty((0, 84), float)
for idx in traces_idx:
# for idx in range(0, len(trajectory_list)):
#     print(trajectory_list[idx].shape)

    # Reverse the order of the data (IF PROXEMICS or MUG DO NOT REVERSE)
    # trajectory_list[idx] = trajectory_list[idx][::-1]
    unknown_feature.add_data(trajectory_list[idx])
    all_trace_data = np.vstack((all_trace_data, trajectory_list[idx]))
    # all_trace_data = np.append(all_trace_data, trajectory_list[idx])

# print(all_trace_data.shape)
# all_trace_data = np.empty((0,), float)
# for idx in traces_idx:
#     print(len(trajectory_list[idx]))
#     # print(trajectory_list[idx])
#     for traj in trajectory_list[idx]:
#         print(len(traj))
#         print()
#     break
#     unknown_feature.add_data(trajectory_list[idx])
#     all_trace_data = np.vstack((all_trace_data, trajectory_list[idx]))

# %%
# 1.1 Visualize the Traces labeled at random with the initialized Network
plot_learned_traj(unknown_feature.function, all_trace_data, env, feat=traces_file_cases[case-1])

# %%
# Step 2: train the feature on the set of traces
# torch.set_num_threads(4)
_ = unknown_feature.train(epochs=100, batch_size=32, learning_rate=1e-3, weight_decay=0.001, s_g_weight=10.)

# %%
# Step 3: Analyze the learned Feature

# %%
# 3.1 Visualize the labeled Traces
plot_learned_traj(unknown_feature.function, all_trace_data, env, feat=traces_file_cases[case-1])

# %%
# 3.2 Visualize the learned function over the 3D Reachable Set
plot_learned3D(parent_dir, unknown_feature.function, env, feat=traces_file_cases[case-1])

# %% [markdown]
# # Merge learned feature to others for overall cost/reward function

# %%
# Add learned feature to the environment
env.learned_features.append(unknown_feature)
env.feature_list.append('learned_feature')
env.num_features += 1
env.feature_func_list.append(unknown_feature.function)

# update with pushes data
env.weights = learned_weights_from_pushes_cases[case-1]

# %%
# plot GT
gt_env = Environment("jaco_dynamics", np.array([0.0, -1.5708, 0, -1.5708, 0, 0]), obj_center_dict, feat_list_cases[case-1],
                  gt_feat_range, weights_cases[case-1], viewer=False)
# gt_env = Environment("jaco_dynamics", np.array([0.0, -1.5708, 0, -1.5708, 0, 0]), obj_center_dict, feat_list_cases[case-1],
#                   feat_list_cases[case-1], weights_cases[case-1], viewer=False)
plot_gt3D(parent_dir, gt_env)

# %%
# Plot learned 3D cost function
plot_gt3D(parent_dir, env, title='FERL Learned Cost over 3D Reachable Set')

# %%



