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

# %%
# import FERL Modules
from utils.learned_feature import LearnedFeature
from utils.environment import *
from utils.plot_utils import *

# %%
# Settings for the different cases
feat_list_cases = [["coffee","table", "laptop"],["coffee","laptop", "table"], ["coffee","table", "proxemics"]]
weights_cases = [[0.0, 10.0, 10.0], [0.0, 10.0, 10.0], [0.0, 10.0, 10.0]]
known_features_cases = [["coffee", "table"], ["coffee", "laptop"], ["coffee", "table"]]
known_weights = [0., 0.]

traces_file_cases = ["laptop", "table", "proxemics"]
# traces_file_cases = ["laptop"]
traces_idx = np.arange(10).tolist()

# learned weights from pushes
p1 = [0.0, 4.03901256, 5.51417794]
p2 = [0.0, 4.35964768, 4.88110989]
p3 = [0.0, 3.09983027, 5.1572305 ]
# p1 = [0.0, 0.0, 0.0]
# p2 = [0.0, 0.1, 0.0]
# p3 = [0.0, 0.0, 0.0]

learned_weights_from_pushes_cases = [p1, p2, p3]

# some settings for TrajOpt
FEAT_RANGE = {'table':0.98, 'coffee':1.0, 'laptop':0.3, 'human':0.3, 'efficiency':0.22, 'proxemics': 0.3, 'betweenobjects': 0.2, 'learned_feature':1.0}
obj_center_dict = {'HUMAN_CENTER': [-0.2, -0.5, 0.6], 'LAPTOP_CENTER': [-0.5, 0.0, 0.0]}
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

# for data_file in glob.glob(parent_dir + '/data/FERL_traces/traces_{}.p'.format(traces_file_cases[case-1])):
trajectory_list = []
for data_file in glob.glob(parent_dir + '/data/demonstrations/test.p'):
    trajectory_list = pickle.load(open( data_file, "rb" ))

all_trace_data = np.empty((0, 84), float)
ee_pose = []
for idx in range(len(trajectory_list)):
    # unknown_feature.add_data(trajectory_list[idx])
    all_trace_data = np.vstack((all_trace_data, trajectory_list[idx]))

for idx in range(0, all_trace_data.shape[0]):
    env.robot.SetActiveDOFValues(all_trace_data[idx, :6])
    ee_pose.append(robotToCartesian(env.robot)[-1])

color = range(0, all_trace_data.shape[0])
fig = px.scatter_3d(x=ee_pose[:, 0], y=ee_pose[:, 1], z=ee_pose[:, 2], color = color)

# laptop = np.zeros((1, 84))
# laptop[:, 75] = -0.8
# all_trace_data = np.vstack((all_trace_data, laptop))
# color = range(0, all_trace_data.shape[0])
# print(color)
# fig = px.scatter_3d(x=all_trace_data[:,75], y=all_trace_data[:,76], z=all_trace_data[:,77], color=color)



# fig = px.scatter_3d(x=all_trace_data[:,81], y=all_trace_data[:,82], z=all_trace_data[:,83])

fig.show()


