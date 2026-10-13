# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

ROS 2 Humble (C++17, ament_cmake) packages that fuse body tracking from two ZED cameras (ZED X over GMSL2, or ZED Mini over USB) and turn the fused skeleton into upper-body joint targets and collision capsules for a Unitree G1 humanoid. Compute is normally split across two machines: a ZED Box (Jetson) that only grabs and H.264-streams the cameras, and a workstation with an RTX GPU that runs body tracking and fusion on the streams.

## Build and run

Everything builds and runs inside the Docker container; there is no host-side build. The ZED SDK and CUDA come from the base image.

```bash
git submodule update --init --recursive   # dependencies/ are submodules
cd docker && docker compose up            # builds image, then entrypoint runs rosdep + colcon build
docker exec -it zedx_driver_ros2 bash     # .bashrc already sources ROS + the workspace
```

Inside the container the workspace is `/workspace/ros2_ws`. `../src` is bind-mounted at `src/zedx_driver`, so edits on the host are live; rebuild with:

```bash
cd /workspace/ros2_ws
colcon build --symlink-install --packages-select human_mapping   # or zed_launcher, g1_cbf_msg
source install/setup.bash
```

`dependencies/` (the Stereolabs `zed-ros2-wrapper` and `zed-ros2-examples` submodules) is `COPY`'d and built at image build time, not mounted. Changing a submodule requires `docker compose build`.

Launch:

```bash
ros2 launch zed_launcher server.launch.xml                              # ZED Box: camera stream servers
ros2 launch zed_launcher body_tracking.launch.xml debug:=true rviz:=true  # fusion (workstation, or ZED Box)
ros2 launch human_mapping human_mapping.launch.xml                      # joint targets + colliders
```

The middleware is `rmw_zenoh` (CycloneDDS is no longer installed). Nothing starts the Zenoh router: run `ros2 run rmw_zenoh_cpp rmw_zenohd` in its own shell first, or nodes will not discover each other. `rmw_zenoh` on Humble does not exchange messages with Jazzy or newer nodes.

There are no tests and no lint configuration in this repo.

## Per-machine files

Three things differ between machines and are edited in place, so expect them to show up as local modifications. Do not "fix" or commit them without being asked:

- `docker/Dockerfile` line 1: `stereolabs/zed:5.3.0-py-devel-l4t-r36.4` on the Jetson, `stereolabs/zed:5.3-gl-devel-cuda12.8-ubuntu22.04` on the workstation (the other is kept as a comment). On the AGX Thor host (L4T r38.4) the Jetson image builds, runs ROS and opens cameras, but its ZED SDK cannot do inference or hardware encoding there: TensorRT model optimisation fails with "no kernel image is available for execution on the device" and the stream encoder cannot open `/dev/v4l2-nvenc`. Body tracking and the stream server therefore do not work on the Thor with this image. The matching r38.4 ZED images are Ubuntu 24.04, which has no Humble apt packages.
- `src/RMW_ZENOH_ROUTER_CONFIG.json5`: on one machine of a pair, `connect.endpoints` must list the other machine's router (`tcp/<ip>:7447`). It is loaded via `ZENOH_ROUTER_CONFIG_URI`, set in the `.bashrc` block that `docker/entrypoint.sh` writes.
- `ZED_BOX_IP`, also exported from that `.bashrc` block in `docker/entrypoint.sh`. It is the default `stream_address` for `body_tracking.launch.xml`. The block is only appended once per container, so changing it means recreating the container or editing `~/.bashrc` inside it.

## Architecture

Data flow:

```
ZED Box:      zed_wrapper x1 or x2 (server.launch.xml)  -- H.264 streams, ports 30000 / 30002
                   |
Workstation:  zed_body_fusion_node   -> /zed_fusion/body_trk/skeletons   (zed_msgs/ObjectsStamped, frame fusion_world)
                   |
              human_mapping_node     -> /upper_body_targets_unsafe       (sensor_msgs/JointState, 8 joints)
                                     -> /human/colliders                 (g1_cbf_msg/CapsuleArray)
                   |
              human_collider_viz_node -> /human/collider_markers         (MarkerArray, RViz only)
```

### `zed_launcher`

`server.launch.xml` includes the upstream `zed_camera.launch.py` twice and uses `param_overrides` to turn off every wrapper feature (depth, tracking, all publishers) except the stream server. The overrides shared by both cameras live in the `common_overrides` variable.

Both launch files take `camera_type` (`zedx` or `zedm`), which must match on the two sides. Its value is passed straight to the wrapper as `camera_model`, and it also selects the default resolution (ZED Mini only reaches 60 fps at HD720; the full mode table is in the comment at the top of `server.launch.xml`), the default serials, the calibration folder and `num_cameras` in `server.launch.xml` (2 for `zedx`, 1 for `zedm`; with 1 only the left camera is started). Derived defaults use `$(eval ...)` on `$(var camera_type)`.

`zed_body_fusion_node` does not consume ROS topics from the wrapper. It uses the ZED SDK directly: for each camera in the calibration file (one or two) it opens an `sl::Camera` (from the network stream when `input_mode:=stream`, or from the local device when `live`), enables per-camera body tracking, and calls `startPublishing` over shared memory. Each camera has its own grab thread (`runCameraWorker`). An in-process `sl::Fusion` subscribes to both, and a wall timer (`processFusion`) retrieves fused bodies and publishes them. Points to know:

- The calibration file (`calibration/<camera_type>/calibration.json`, ZED360 output) must contain one or two cameras; `fusion_minimum_allowed_cameras` is lowered to that count. `calibration/zedm/` is hand-written for a single ZED Mini with an identity pose, so `fusion_world` is at the camera, gravity-aligned. Cameras are matched to stream ports and names by serial number via `left_serial` / `right_serial`; the defaults are the serials of the current rigs, and a lone camera is the left one (`right_serial` 0). New cameras need both the calibration and those args updated.
- Calibration files are installed to the package share directory, so adding or renaming one needs a `colcon build` of `zed_launcher` even with `--symlink-install`.
- The SDK is configured for ROS conventions (`RIGHT_HANDED_Z_UP_X_FWD`, metres), so no axis conversion happens downstream. Camera poses from the calibration are published as static TFs `fusion_world -> <camera_name>_left_camera_optical_frame`.
- `single_body_enabled` publishes only one skeleton. Selection is sticky: it switches to a different body only if that body beats the current one by `single_body_switch_margin` confidence for `single_body_switch_frames` consecutive frames.
- `debug:=true` only sets the defaults of `publish_overlay_images` and `publish_per_camera_skeletons`. Image retrieval is skipped when nothing subscribes.
- `role` accepts only `local` (`remote` / `stream_client` are aliases for local + stream).
- Launch-file defaults differ from the `declare_parameter` defaults in the C++ for several parameters (e.g. `body_model`, `fusion_reference_frame`, `fusion_skeleton_smoothing`, `fusion_minimum_allowed_cameras`). The launch file is what actually runs.

`zed_skeleton_overlay_node` is a standalone topic-based overlay drawer. It is built and installed but not started by any launch file; the fusion node draws its own overlays.

### `human_mapping`

`human_mapping_node` takes the highest-confidence body from the fused skeleton topic and assumes BODY_38 keypoint indices (constants at the top of `human_mapping_node.hpp`). Pipeline per message: per-keypoint EMA with jump rejection, then geometric angle estimation in a torso frame built from pelvis and shoulders, then angle EMA with rate limiting, then neutral-pose subtraction, then `q_home + sign * gain * delta + bias` clamped to `[q_min, q_max]`. A separate timer republishes the latest result at `publish_rate_hz`.

- Neutral calibration: for `startup_delay_sec` after node start the output delta is zero, then angles are averaged for `neutral_calibration_duration` and used as the zero pose. The person must hold a neutral pose during that window; until it completes the node outputs `q_home`.
- If no valid skeleton arrives within `stale_timeout_sec`, it publishes `q_home` and an empty capsule array rather than holding the last pose.
- The joint order is fixed at 8 joints (waist roll/pitch, then shoulder pitch/roll and elbow per arm). `joint_names`, `q_home`, `signs`, `gains`, `bias`, `q_min`, `q_max` in `config/human_mapping.yaml` must all have exactly 8 entries in that order. `AngleFilter` also has its own hard-coded per-joint limits in the header, applied before the YAML limits.
- Capsules are built from the same filtered keypoints, with radii multiplied by `human_radius_scale`, and forearm capsules extended past the wrist by `hand_extension_length` to cover the hand.
- Launch arguments override the YAML for topic names; the viz node's `collider_marker_rate` launch default (5.0) overrides the YAML value (60.0).

### `g1_cbf_msg`

Defines `Capsule` and `CapsuleArray`. The package name and message layout are shared with the G1 control-barrier-function consumer outside this repo, so treat them as an external interface.

## Code conventions

- Each node is split into three files: `include/<pkg>/<name>_node.hpp` (class declaration), `src/<name>.cpp` (implementation), and `src/<name>_node.cpp` (`main`). The `_node.cpp` file `#include`s the implementation `.cpp` directly, and CMake lists only the `_node.cpp`. A new source file must be included the same way or added to `add_executable`.
- Launch files are XML, not Python.
- Skeleton, joint-target and collider topics are published best-effort (`SensorDataQoS` in the fusion node, best-effort keep-last-1 in `human_mapping_node`). A reliable subscriber will not receive them.
