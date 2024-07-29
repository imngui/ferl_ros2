from ferl.utils.environment import Environment

def test_environment():
    test_files = ["test/data/map.pcd"]
    env = Environment(test_files)
    assert len(env.obstacles) == 1
    assert env.obstacles[0].label == "map"

    print("Convex Hull:")
    print("Num Points:", env.obstacles[0].collision_geometry.num_points)
    print("Center:", env.obstacles[0].collision_geometry.center)
    print("Points:", env.obstacles[0].collision_geometry.points())


if __name__ == "__main__":
    test_environment()