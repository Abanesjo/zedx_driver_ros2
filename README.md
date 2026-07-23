# zedx_driver_ros2

This repository provides bringup and fused body tracking for three ZED X cameras. The cameras are attached to a ZED Box, while the primary compute runs on a more powerful networked computer.

## Usage
The instructions assume you will be splitting compute between the ZED Box and a networked workstation computer with an RTX GPU

The default camera-to-stream mapping matches `src/zed_launcher/calibration/calibration.json`:

| Camera | Serial number | Stream port |
| --- | ---: | ---: |
| `zed_left` | `41235597` | `30000` |
| `zed_center` | `46229474` | `30004` |
| `zed_right` | `49967328` | `30002` |

The server binds each camera by serial number, so USB device enumeration does not affect this mapping.

`body_tracking.launch.xml` enables three-camera, two-tag fusion by default.
Front tag ID 0 and back tag ID 1 are both 12 cm `APRILTAG_36h11` markers.
For each tag, detections from the largest agreeing camera set are fused using
reprojection quality. When both tags are synchronized, their positions are
averaged and `tag_frame` uses tag 1's orientation. With one visible tag, the
last learned front-to-back separation is used. With neither visible, the last
higher-quality tag estimate is held and rebroadcast.

The resulting frame ownership is:

```text
pelvis --static--> tag_frame --dynamic--> fusion_world --static--> cameras
```

The `pelvis -> tag_frame` mounting edge is optional and must be published by
the robot as a static transform. The AprilTag node publishes only the dynamic
`tag_frame -> fusion_world` edge; the three camera edges come from
`calibration.json`. Skeletons and colliders remain message data with
`header.frame_id=fusion_world`; no joint TF tree is generated. The equivalent
pose topic is `/fusion_world_pose_in_tag_frame`.

Tag 1 is assumed to be rotated 180 degrees about tag 0's local y axis.
`tag_frame` matches tag 1's orientation. Each tag's local +z points inward,
toward the initial center estimate 3 cm away. The node learns the tag-to-tag
z separation from synchronized pairs at runtime; that estimate resets on every
node startup.

Generate actual-size print files for the default tags with:

```bash
ros2 run zed_launcher generate_apriltag_prints --output-dir ./apriltag_prints
```

This creates one 600-DPI A4 PNG per tag plus a two-page PDF. Print the PDF at
100% / Actual Size with all fit-to-page scaling disabled, then verify the
included 10 cm scale.

### ZED Box Setup
On the ZED Box: 
1. Clone the repository
```
git clone --recursive https://github.com/Abanesjo/zedx_driver_ros2
```
2. Modify the `src/cyclonedds.xml` file with the network / ethernet interface of the ZED Box
3. On the `docker/Dockerfile`, ensure the first line is: 
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
2. Modify the `src/cyclonedds.xml` file with the network / ethernet interface of the Workstation
3. On the `docker/Dockerfile`, ensure the first line is: 
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

Setting `debug:=true` publishes separate per-camera image topics for the
annotated 2D skeleton and AprilTag axes, plus the 3D skeleton per image and the
fused 3D skeleton. The AprilTag images use
`/zed_{left,center,right}/zed_node/rgb/color/rect/apriltag_overlay`; skeleton
images use the corresponding `skeleton_overlay` suffix.
Disabling debug is recommended for improving performance.

6. For the human angle calculation and collision capsules, run:
```
docker exec -it zedx_driver_ros2 bash
ros2 launch human_mapping human_mapping.launch.xml
```

This launch publishes the tracked G1 upper-body joint commands on
`/human/joint_commands` and the tracked body/hand capsules on
`/human/colliders`. It replaces the role of `g1_human_manual.launch.xml` for
live tracking; do not run both launches at the same time because they publish
the same topics.

You should see a visualization like below

![rviz](docs/rviz.png)

The axes also show the position of the cameras in the world frame. 

## Calibration
To calibrate the cameras, update `src/zed_launcher/calibration/calibration.json` with the output obtained from the [ZED360](https://www.stereolabs.com/docs/fusion/zed360) tool.
