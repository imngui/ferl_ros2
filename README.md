# ferl_ros2
ROS 2 implementation of Feature Expansive Reward Learning

install ros-humble-srdfdom

unzip the stls and place into the kortex_description meshes directory where the urdf specifies, replace dae with stl in the urdf

# XR Instructions
1. Launch simulated ur_robot_driver
```bash
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5e robot_ip:=yyy.yyy.yyy.yyy use_mock_hardware:=true launch_rviz:=true
```

2. Launch xr_ferl
```bash
ros2 launch ferl xr_ferl.launch.py ur_type:=ur5e launch_rviz:=false
```
3. Do VR Stuff




# Steps to run
1. Launch ur_robot_driver
```bash 
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5e robot_ip:=192.168.1.5 launch_rviz:=true
```

2. Start robot on teach pendant

3. Clear force/torque data
```bash
ros2 service call /io_and_status_controller/zero_ftsensor std_srvs/srv/Trigger
```

4. Setup controllers
```bash
ros2 control switch_controllers --deactivate scaled_joint_trajectory_controller
```
```bash
ros2 control switch_controllers --activate forward_velocity_controller
```

5. Launch ferl node
```bash
ros2 launch ferl test_vel.launch.py ur_type:=ur5e launch_rviz:=false launch_servo:=true
```