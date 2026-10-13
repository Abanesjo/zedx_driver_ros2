# zedx_driver_ros2

This repository allows bringup and body tracking of a ZEDX Camera. The cameras are attached to a ZED Box and the primary compute is on a more powerful networked computer. 

USB cameras such as the ZED Mini are also supported, see [Camera Type](#camera-type).

## Usage
The instructions assume you will be splitting compute between the ZED Box and a networked workstation computer with an RTX GPU

### ZED Box Setup
On the ZED Box: 
1. Clone the repository
```
git clone --recursive https://github.com/Abanesjo/zedx_driver_ros2
```
2. ROS 2 communication uses `rmw_zenoh`. Leave `src/RMW_ZENOH_ROUTER_CONFIG.json5` as it is on the ZED Box
3. On the `docker/Dockerfile`, ensure the first line is: 
```
FROM stereolabs/zed:5.3.0-py-devel-l4t-r36.4
```
4. Create the docker container
```
cd docker
docker compose up
```
5. Start the Zenoh router (within the docker container) and leave it running. Nodes cannot discover each other without it
```
docker exec -it zedx_driver_ros2 bash
ros2 run rmw_zenoh_cpp rmw_zenohd
```
6. Start the ZED server (within the docker container)
```
docker exec -it zedx_driver_ros2 bash
ros2 launch zed_launcher server.launch.xml
```
7. (Optional) You can run body tracking directly on the ZED Box
```
docker exec -it zedx_driver_ros2 bash
ros2 launch zed_launcher body_tracking.launch.xml debug:=true rviz:=true
```

Now that the jetson/ZED Box is setup, we now launch on the separate compute.
### Networked Workstation Setup
On the Networked Workstation: 
1. Clone the repository
```
git clone --recursive https://github.com/Abanesjo/zedx_driver_ros2
```
2. In `src/RMW_ZENOH_ROUTER_CONFIG.json5`, add the ZED Box router to `connect.endpoints` so the two routers find each other, for example `endpoints: ["tcp/192.168.0.133:7447"]`
3. On the `docker/Dockerfile`, ensure the first line is: 
```
FROM stereolabs/zed:5.3-gl-devel-cuda12.8-ubuntu22.04
```
4. Create the docker container
```
cd docker
docker compose up
```
5. Start the Zenoh router (within the docker container) and leave it running
```
docker exec -it zedx_driver_ros2 bash
ros2 run rmw_zenoh_cpp rmw_zenohd
```
6. Start the Body Tracking (within the docker container)
```
docker exec -it zedx_driver_ros2 bash
ros2 launch zed_launcher body_tracking.launch.xml debug:=true rviz:=true
```

Setting `debug:=true` publishes as images the annotated 2d skeleton on each camera image, the 3d skeleton per image, and the fused 3d skeleton. Disabling it is recommended for improving performance. 

7. For the human angle calculation and collision capsules, run:
```
docker exec -it zedx_driver_ros2 bash
ros2 launch human_mapping human_mapping.launch.xml
```

You should see a visualization like below

![rviz](docs/rviz.png)

The axes also show the position of the cameras in the world frame. 

## Camera Type
Both `server.launch.xml` and `body_tracking.launch.xml` take a `camera_type` argument. Use the same value on both.

| `camera_type` | Cameras | Default stream mode |
|---|---|---|
| `zedx` (default) | ZED X (GMSL2) | HD1080 at 60 fps |
| `zedm` | ZED Mini (USB) | HD720 at 60 fps |

A ZED Mini cannot do HD1080 at 60 fps. The valid resolution and frame rate combinations for each type are listed at the top of `src/zed_launcher/launch/server.launch.xml`; override them with `grab_resolution:=` and `grab_frame_rate:=`.

### Number of cameras
The `zedm` setup currently uses a single ZED Mini, so `num_cameras` in `server.launch.xml` defaults to 1 for `zedm` (2 for `zedx`). With one camera only the left camera is started:
```
ros2 launch zed_launcher server.launch.xml camera_type:=zedm
ros2 launch zed_launcher body_tracking.launch.xml camera_type:=zedm
```
Body tracking takes the number of cameras from the calibration file, which may hold one or two cameras. If the server runs on the same machine, add `stream_address:=127.0.0.1`.

For two USB cameras, pass `num_cameras:=2` and the serial numbers (shown by `ZED_Explorer -a`), because USB camera IDs depend on the order the cameras are detected:
```
ros2 launch zed_launcher server.launch.xml camera_type:=zedm num_cameras:=2 left_serial:=<serial> right_serial:=<serial>
ros2 launch zed_launcher body_tracking.launch.xml camera_type:=zedm left_serial:=<serial> right_serial:=<serial>
```

## Calibration
Each camera type has its own calibration file: `src/zed_launcher/calibration/<camera_type>/calibration.json`. To calibrate the cameras, update that file with the output obtained from the [ZED360](https://www.stereolabs.com/docs/fusion/zed360) tool, then rebuild `zed_launcher`.

The `zedm` calibration is written by hand for a single camera: an identity pose, which puts the world origin at the camera with gravity-aligned axes.
