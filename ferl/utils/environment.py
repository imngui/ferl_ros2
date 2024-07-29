import os
import open3d as o3d
import pinocchio as pin
import numpy as np
import hppfcl as fcl

class Obstacle:
    def __init__(self, collision_geometry, rotation = pin.SE3.Identity(), translation = np.zeros(3), label = ""):
        self.collision_geometry = collision_geometry
        self.rotation = rotation
        self.translation = translation
        self.label = label

class Environment:
    def __init__(self, obstacle_files = [], resolution = 0.01):
        self.resolution = resolution
        self.obstacle_files = obstacle_files
        self.obstacles = []
        self.load_obstacles()
    
    def load_obstacles(self):
        for obstacle_file in self.obstacle_files:
            if not os.path.exists(obstacle_file):
                raise FileNotFoundError(f"Obstacle file {obstacle_file} does not exist.")
            
            if obstacle_file.endswith(".pcd"):
                # Load point cloud
                points = parse_point_cloud(obstacle_file)
                verts = fcl.StdVec_Vec3f()
                verts.extend(points)

                # Create convex hull
                convex = fcl.Convex.convexHull(verts, False, "")
                
                # Create obstacle
                label = os.path.basename(obstacle_file).split(".")[0]
                obstacle = Obstacle(convex, label=label)
                self.obstacles.append(obstacle)
            else:
                raise ValueError(f"Unknown file format for obstacle file {obstacle_file}")
            
def parse_point_cloud(pcd_file):
    # Extract points from the point cloud (.pcd) file
    pcd = o3d.io.read_point_cloud(pcd_file)
    points = np.asarray(pcd.points)

    return points
