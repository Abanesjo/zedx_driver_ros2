# zedx_driver_ros2

This repository allows bringup and body tracking of a ZEDX Camera. The cameras are attached to a ZED Box and the primary compute is on a more powerful networked computer. 

## Usage
The instructions assume you will be splitting compute between the ZED Box and a networked workstation computer with an RTX GPU

### ZED Box Setup
On the ZED Box: 
1. Clone the repository
```
git clone --recursive https://github.com/Abanesjo/zedx_driver_ros2
```
2. Modify the `cyclonedds.xml` file with the network / ethernet interface of the ZED Box
3. On the `Dockerfile`, ensure the first line is: 
```
FROM stereolabs/zed:5.3.0-py-devel-l4t-r36.4
```
4. Create the docker container
```
cd docker
docker compose up
```
5. Start the ZED server (within the docker container)
```
docker exec -it zedx_driver_ros2 bash
ros2 launch zed_launcher server.launch.xml
```
6. (Optional) You can run body tracking directly on the ZED Box
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
2. Modify the `cyclonedds.xml` file with the network / ethernet interface of the Workstation
3. On the `Dockerfile`, ensure the first line is: 
```
FROM stereolabs/zed:5.3-gl-devel-cuda12.8-ubuntu22.04
```
4. Create the docker container
```
cd docker
docker compose up
```
5. Start the Body Tracking (within the docker container)
```
docker exec -it zedx_driver_ros2 bash
ros2 launch zed_launcher body_tracking.launch.xml debug:=true rviz:=true
```

Setting `debug:=true` publishes as images the annotated 2d skeleton on each camera image, the 3d skeleton per image, and the fused 3d skeleton. Disabling it is recommended for improving performance. 

6. For the human angle calculation and collision capsules, run:
```
docker exec -it zedx_driver_ros2 bash
ros2 launch human_mapping human_mapping.launch.xml
```

You should see a visualization like below

![rviz](docs/rviz.png)

The axes also show the position of the cameras in the world frame. 

## Calibration
To calibrate the cameras, update `calibration.json` with the output obtained from the [ZED360](https://www.stereolabs.com/docs/fusion/zed360) tool.
