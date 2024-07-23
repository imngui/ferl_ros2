import os
import time
import numpy as np
import casadi
import pinocchio as pin
from pinocchio import casadi as cpin
import pathlib
import torch

class TrajOptPlanner(object):
    """
    """
    def __init__(self, max_iter, num_waypts, env):
        self.MAX_ITER = max_iter
        self.num_waypts = num_waypts
        
        self.env = env
        
        # Pinocchio setup
        self.robot = pin.robot_wrapper.RobotWrapper.BuildFromURDF(self.env.urdf_file)
        self.model = self.robot.model
        self.data = self.model.createData()
        self.ee_frame = self.env.ee_frame
        self.ee_frame_ID = self.model.getFrameId(self.ee_frame)
        self.opti = casadi.Opti()
    
        
    # -- Interpolate feature value between neighboring waypoints to help planner optimization. -- #

    def interpolate_features(self, waypt, prev_waypt, feat_idx, NUM_STEPS=4):
        """
        Computes feature value over waypoints, interpolating and
        sampling between each pair to check for intermediate collisions.
        ---
        input neighboring waypoints and feature function, output scalar feature
        """
        feat_val = 0.0
        for step in range(NUM_STEPS):
            inter_waypt = prev_waypt + ((1.0 + step)/NUM_STEPS)*(waypt - prev_waypt)
            feat_val += self.env.featurize_single(inter_waypt, feat_idx)
        return feat_val / NUM_STEPS

	# ---- Costs ---- #
 
    def efficiency_cost(self, waypt):
        """
        Computes the total efficiency cost
        ---
        input waypoint, output scalar cost
        """
        feature_idx = self.env.feature_list.index('efficiency')
        feature = self.interpolate_features(waypt, waypt, feature_idx, NUM_STEPS=1)
        return feature*self.env.weights[feature_idx]
    
    def origin_cost(self, prev_waypt, curr_waypt):
        """
        Computes the total distance from EE to base of robot cost.
        ---

        input waypoint, output scalar cost
        """
        # prev_waypt = waypt[0:7]
        # curr_waypt = waypt[7:14]
        feature_idx = self.env.feature_list.index('origin')
        feature = self.interpolate_features(curr_waypt, prev_waypt, feature_idx)
        return feature*self.env.weights[feature_idx]*np.linalg.norm(curr_waypt - prev_waypt)

    def table_cost(self, prev_waypt, curr_waypt):
        """
        Computes the total distance to table cost.
        ---
        input waypoint, output scalar cost
        """
        # prev_waypt = waypt[0:7]
        # curr_waypt = waypt[7:14]
        feature_idx = self.env.feature_list.index('table')
        feature = self.interpolate_features(curr_waypt, prev_waypt, feature_idx)
        return feature*self.env.weights[feature_idx]*np.linalg.norm(curr_waypt - prev_waypt)

    def coffee_cost(self, prev_waypt, curr_waypt):
        """
        Computes the total coffee (EE orientation) cost.
        ---
        input waypoint, output scalar cost
        """
        # prev_waypt = waypt[0:7]
        # curr_waypt = waypt[7:14]
        feature_idx = self.env.feature_list.index('coffee')
        feature = self.interpolate_features(curr_waypt, prev_waypt, feature_idx)
        return feature*self.env.weights[feature_idx]*np.linalg.norm(curr_waypt - prev_waypt)

    def laptop_cost(self, prev_waypt, curr_waypt):
        """
        Computes the total distance to laptop cost
        ---
        input waypoint, output scalar cost
        """
        # prev_waypt = waypt[0:7]
        # curr_waypt = waypt[7:14]
        feature_idx = self.env.feature_list.index('laptop')
        feature = self.interpolate_features(curr_waypt, prev_waypt, feature_idx)
        return feature*self.env.weights[feature_idx]*np.linalg.norm(curr_waypt - prev_waypt)

    def human_cost(self, prev_waypt, curr_waypt):
        """
        Computes the total distance to human cost.
        ---
        input waypoint, output scalar cost
        """
        # prev_waypt = waypt[0:7]
        # curr_waypt = waypt[7:14]
        feature_idx = self.env.feature_list.index('human')
        feature = self.interpolate_features(curr_waypt, prev_waypt, feature_idx)
        return feature*self.env.weights[feature_idx]*np.linalg.norm(curr_waypt - prev_waypt)

    def learned_feature_costs(self, prev_waypt, curr_waypt):
        """
        Computes the cost for all the learned features.
        ---
        input waypoint, output scalar cost
        """
        # prev_waypt = waypt[0:7]
        # curr_waypt = waypt[7:14]
        # get the number of learned features
        n_learned = self.env.feature_list.count('learned_feature')

        feature_values = []
        for i, feature in enumerate(self.env.learned_features):
            # get the value of the feature
            feat_idx = self.env.num_features - n_learned + i
            feature_values.append(self.interpolate_features(curr_waypt, prev_waypt, feat_idx))
        # calculate the cost
        return np.matmul(self.env.weights[-n_learned:], np.array(feature_values))*np.linalg.norm(curr_waypt - prev_waypt)
	
    def learned_feature_cost_derivatives(self, waypt):
        """
        Computes the cost derivatives for all the learned features.
        ---
        input waypoint, output scalar cost
        """
        # get the number of learned features
        n_learned = self.env.feature_list.count('learned_feature')

        J = []
        sols = []
        for i, feature in enumerate(self.env.learned_features):
            # Setup for computing Jacobian.
            x = torch.tensor(waypt, requires_grad=True)

            # Get the value of the feature
            feat_idx = self.env.num_features - n_learned + i
            feat_val = torch.tensor(0.0, requires_grad=True)
            NUM_STEPS = 4
            for step in range(NUM_STEPS):
                delta = torch.tensor((1.0 + step)/NUM_STEPS, requires_grad=True)
                inter_waypt = x[:7] + delta * (x[7:] - x[:7])
                # Compute feature value.
                z = self.env.feature_func_list[feat_idx](self.env.raw_features(inter_waypt).float(), torchify=True)
                feat_val = feat_val + z
            y = feat_val / torch.tensor(float(NUM_STEPS), requires_grad=True)
            y = y * torch.tensor(self.env.weights[-n_learned+i:], requires_grad=True) * torch.norm(x[7:] - x[:7])
            y.backward()
            J.append(x.grad.data.numpy())
        return np.sum(np.array(J), axis = 0).reshape((1,-1))

	# ---- Here's TrajOpt --- #

    def trajOpt(self, start, goal, goal_pose, traj_seed=None):
        """
        Computes a plan from start to goal using trajectory optimizer.
        Reference: http://joschu.net/docs/trajopt-paper.pdf
        ---
        Paramters:
            start -- The start position.
            goal -- The goal position.
            goal_pose -- The goal pose (optional: can be None).
            traj_seed [optiona] -- An optional initial trajectory seed.

        Returns:
            waypts_plan -- A downsampled trajectory resulted from the TrajOpt
            optimization problem solution.
        """

        # --- Initialization --- #
        self.robot.q0 = np.array(start)
        
        cmodel = cpin.Model(self.model)
        cdata = cmodel.createData()
        
        nq = self.model.nq
        nv = self.model.nv
        nx = nq + nv
        ndx = 2 * nv
        cx = casadi.SX.sym("x", nx, 1)
        cdx = casadi.SX.sym("dx", nv * 2, 1)
        cq = cx[:nq]
        cv = cx[nq:]
        caq = casadi.SX.sym("a", nv, 1)

        # Compute kinematics casadi graphs
        cpin.forwardKinematics(cmodel, cdata, cq, cv, caq)
        cpin.updateFramePlacements(cmodel, cdata)
        
        # Sym graph for the integration operation x' = [ q+vDT+aDT**2, v+aDT ]
        cnext = casadi.Function(
            "next",
            [cx, caq],
            [
                casadi.vertcat(
                    cpin.integrate(cmodel, cx[:nq], cx[nq:] * self.env.DT + caq * self.env.DT**2),
                    cx[nq:] + caq * self.env.DT,
                )
            ],
        )
        
        w_vel = 0.1
        w_conf = 5
        
        self.opti = casadi.Opti()
        var_xs = [self.opti.variable(nx) for t in range(self.env.T + 1)]
        var_as = [self.opti.variable(nv) for t in range(self.env.T)]
        totalcost = 0

        # --- Linear interpolation seed --- #
        if traj_seed is None:
            # print("Using straight line initialization!")
            init_waypts = np.zeros((self.num_waypts, nq))
            for count in range(self.num_waypts):
                init_waypts[count, :] = start + count/(self.num_waypts - 1.0)*(goal - start)
        else:
            print("Using trajectory seed initialization!")
            init_waypts = traj_seed
                    
        # Convert traj_seed to casadi DM
        for t in range(self.env.T+1):
            init_waypts[t] = casadi.DM(init_waypts[t])
            
        # --- Running Cost --- #
        for t in range(self.env.T):
            
            totalcost += 1e-3 * self.env.DT * casadi.sumsqr(var_xs[t][nq:])
            totalcost += 1e-3 * self.env.DT * casadi.sumsqr(var_xs[t][:nq])
            totalcost += 1e-4 * self.env.DT * casadi.sumsqr(var_as[t])
        
        # TODO: Add velocity limits(?)
        
        # --- Goal cost --- #
        # If pose is given, must include pose constraint.
        if goal_pose is not None:
            # print("Using goal pose for trajopt computation.")
            # xyz_target = goal_pose
            # quat_target = [1, 0, 0, 0]  #wxyz
            # constraint = [
            #     {
            #         "type": "pose",
            #         "params": {"xyz" : xyz_target,
            #                     "wxyz" : quat_target,
            #                     "link": "j2s7s300_link_7",
            #                     "rot_coeffs" : [0, 0, 0],
            #                     "pos_coeffs" : [35, 35, 35],
            #                     }
            #     }
            # ]
            Mtarget = pin.SE3(pin)
            error3_tool = casadi.Function(
                "etool3", [cq], [cdata.oMf[self.ee_frame_ID].translation - Mtarget.translation]
            )
            error6_tool = casadi.Function(
                "etool6",
                [cq],
                [cpin.log6(cdata.oMf[self.ee_frame_ID].inverse() * cpin.SE3(Mtarget)).vector],
            )
            error_tool = error3_tool
            
            totalcost += 1e4 * casadi.sumsqr(error_tool(var_xs[self.env.T]))
        else:
            joint_goal = np.array(goal.toList())
            
            totalcost += 1e4 * casadi.sumsqr(var_xs[self.env.T][:nq] - joint_goal)
            
        # Initial Constraints
        self.opti.subject_to(var_xs[0][:nq] == self.robot.q0)
        self.opti.subject_to(var_xs[0][nq:] == 0)
        
        # Integration Constraints
        for t in range(self.env.T):
            self.opti.subject_to(cnext(var_xs[t], var_as[t]) == var_xs[t + 1])


        # --- Setup additional costs --- #
        # Feature Costs
        for t in range(1, self.num_waypts):
                if 'coffee' in self.env.feature_list:
                    totalcost += self.coffee_cost(var_xs[t-1][:nq], var_xs[t][:nq])
                if 'table' in self.env.feature_list:
                    totalcost += self.table_cost(var_xs[t-1][:nq], var_xs[t][:nq])
                if 'laptop' in self.env.feature_list:
                    totalcost += self.table_cost(var_xs[t-1][:nq], var_xs[t][:nq])
                if 'origin' in self.env.feature_list:
                    totalcost += self.origin_cost(var_xs[t-1][:nq], var_xs[t][:nq])
                if 'human' in self.env.feature_list:
                    totalcost += self.human_cost(var_xs[t-1][:nq], var_xs[t][:nq])
                if 'efficiency' in self.env.feature_list:
                    totalcost += self.efficiency_cost(var_xs[t-1][:nq], var_xs[t][:nq])
                if 'learned_feature' in self.env.feature_list:
                    # TODO: Determine how trajoptpy uses the cost derivatives and if they should be included here
                    # prob.AddErrorCost(self.learned_feature_costs, self.learned_feature_cost_derivatives, [(t-1, j) for j in range(7)]+[(t, j) for j in range(7)], "ABS", "learned_features%i"%t)
                    totalcost += self.learned_feature_cost(var_xs[t-1][:nq], var_xs[t][:nq])
        
        for t in range(1, self.num_waypts - 1):
            # TODO: Figure out how to represent table constraints and other obstacle constraints
            # prob.AddConstraint(self.env.table_constraint, [(t, j) for j in range(7)], "INEQ", "up%i"%t)
            continue

        
        # Set cost minimization objective
        self.opti.minimize(totalcost)
        
        # Set solver parameters
        p_opts = {"expand": True}
        s_opts = {"max_iter": 100}
        self.opti.solver("ipopt", p_opts, s_opts)  # set numerical backend
        
        # Set trajectory seed
        for t in range(self.env.T+1):
            self.opti.set_initial(var_xs[t][:nq], init_waypts[t])
                
        # Solve problem
        try:
            sol = self.opti.solve_limited()
            sol_xs = [self.opti.value(var_x) for var_x in var_xs]
            sol_as = [self.opti.value(var_a) for var_a in var_as]
        except:
            print("ERROR in convergence, plotting debug info.")
            sol_xs = [self.opti.debug.value(var_x) for var_x in var_xs]
            sol_as = [self.opti.debug.value(var_a) for var_a in var_as]
        
        # --- Get Trajectory --- #
        q_opt = [x[:nq] for x in sol_xs]
        # v_opt = traj[:, self.state.nq:]
        return q_opt
        

    def replan(self, start, goal, goal_pose, T, timestep, seed=None):
        """
        Replan the trajectory from start to goal given weights.
        ---
        Parameters:
            start -- Start position
            goal -- Goal position.
            goal_pose -- Goal pose (optional: can be None).
            T [float] -- Time horizon for the desired trajectory.
            timestep [float] -- Frequency of waypoints in desired trajectory.
        Returns:
            traj [Trajectory] -- The optimal trajectory satisfying the arguments.
        """
        waypts = self.trajOpt(start, goal, goal_pose, traj_seed=seed)
        waypts_time = np.linspace(0.0, T, self.num_waypts)
        traj = Trajectory(waypts, waypts_time)
        return traj.upsample(int(T/timestep) + 1)