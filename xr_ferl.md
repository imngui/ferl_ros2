# Instructions

1. Start simulated robot driver
```bash
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5e robot_ip:=yyy.yyy.yyy.yyy use_mock_hardware:=true launch_rviz:=true
```

2. Start application on XR device

3. Start Unity-ROS TCP Endpoint with the ip address set to the host machines ip
```bash
ros2 run ros_tcp_endpoint default_server_endpoint --ros-args -p ROS_IP:=<ip_address>
```

4. Start XR FERL node
```bash
ros2 launch ferl xr_ferl.launch.py ur_type:=ur5e launch_rviz:=false
```