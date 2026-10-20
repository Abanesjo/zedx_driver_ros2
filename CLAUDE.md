# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

ROS 2 Jazzy (C++17, ament_cmake) packages that fuse body tracking from several ZED cameras (three ZED X over GMSL2, or two ZED Minis over USB), localize the robot against AprilTags, and turn the fused skeleton into upper-body joint targets and collision capsules for a Unitree G1 humanoid. With ZED X the compute is split across two machines: a ZED Box (Jetson) that only grabs and H.264-streams the cameras, and a workstation with an RTX GPU that runs body tracking, fusion and AprilTags on the streams. ZED Minis are plugged into the machine that runs fusion, with no stream.

## Build and run

Everything builds and runs inside the Docker container; there is no host-side build. The ZED SDK and CUDA come from the base image.

```bash
git submodule update --init --recursive   # dependencies/ are submodules
cd docker && docker compose up            # builds image, then entrypoint runs rosdep + colcon build
docker exec -it zedx_driver_ros2 bash     # .bashrc already sources ROS + the workspace
```

The container runs with `pid: host`, so `pkill -f` inside it also matches host processes; use specific patterns.

Inside the container the workspace is `/workspace/ros2_ws`. `../src` is bind-mounted at `src/zedx_driver`, so edits on the host are live; rebuild with:

```bash
cd /workspace/ros2_ws
colcon build --symlink-install --packages-select human_mapping   # or zed_launcher, g1_cbf_msg
source install/setup.bash
```

`dependencies/` (the Stereolabs `zed-ros2-wrapper` and `zed-ros2-examples` submodules) is `COPY`'d and built at image build time, not mounted. Changing a submodule requires `docker compose build`. Anything added under `dependencies/` ends up in the colcon workspace, so a submodule with plain CMake projects (such as the `zed-sdk` samples) would be built too.

Tests (gtest + pytest, no lint):

```bash
colcon test --packages-select human_mapping zed_launcher --event-handlers console_direct+
colcon test-result --verbose
colcon test --packages-select zed_launcher --ctest-args -R body_tracking_launch_contract   # one test
```

The pytest files under `src/zed_launcher/test/` read the launch files from the source tree and can also run on any host with ROS 2 sourced: `python3 -m pytest src/zed_launcher/test/test_body_tracking_launch_contract.py`.

Launch:

```bash
ros2 launch zed_launcher server.launch.xml                              # ZED Box: three ZED X stream servers
ros2 launch zed_launcher body_tracking.launch.xml debug:=true rviz:=true  # ZED X fusion + AprilTags (workstation, or ZED Box)
ros2 launch zed_launcher body_tracking_single_camera.launch.py camera_id:=1   # one ZED X from the calibration, for testing
ros2 launch zed_launcher zed_mini.launch.xml debug:=true rviz:=true       # two local ZED Minis: fusion only, no server, no AprilTags
ros2 launch human_mapping human_mapping.launch.xml                      # joint targets + colliders
```

The middleware is `rmw_zenoh` (CycloneDDS is no longer installed). Nothing starts the Zenoh router: run `ros2 run rmw_zenoh_cpp rmw_zenohd` in its own shell first, or nodes will not discover each other. `rmw_zenoh` does not exchange messages between Humble and Jazzy, so every machine sharing topics must be on the same distro (the camera stream is not a ROS topic and is unaffected). With `rmw_zenoh`, `ros2 topic hz` / `echo` can ignore SIGTERM, so wrap them in `timeout -s KILL` in scripts.

## Per-machine files

Three things differ between machines and are edited in place, so expect them to show up as local modifications. Do not "fix" or commit them without being asked:

- `docker/Dockerfile`: the `FROM` line and `ARG ROS_DISTRO` at the top. The image's Ubuntu version decides the distro: `stereolabs/zed:5.3.0-py-devel-l4t-r38.4` (Jetson AGX Thor) and `stereolabs/zed:5.3-gl-devel-cuda12.8-ubuntu24.04` (workstation) are Ubuntu 24.04 and use `jazzy`; `stereolabs/zed:5.3.0-py-devel-l4t-r36.4` (JetPack 6 Jetson / ZED Box) is Ubuntu 22.04 and needs `humble`. On the Thor only the r38.4 image works: with the r36.4 image the SDK can open cameras but TensorRT inference and hardware encoding both fail.
- `src/RMW_ZENOH_ROUTER_CONFIG.json5`: on one machine of a pair, `connect.endpoints` must list the other machine's router (`tcp/<ip>:7447`). It is loaded via `ZENOH_ROUTER_CONFIG_URI`, set in the `.bashrc` block that `docker/entrypoint.sh` writes.
- `ZED_BOX_IP`, also exported from that `.bashrc` block in `docker/entrypoint.sh`. It is the default `stream_address` for `body_tracking.launch.xml`. The block is only appended once per container, so changing it means recreating the container or editing `~/.bashrc` inside it.

## Architecture

Data flow:

```
ZED Box:      zed_wrapper x3 (server.launch.xml)       -- H.264 streams, ports 30000 / 30004 / 30002
                   |                      (ZED Mini: no server; the fusion node opens the USB cameras itself)
Workstation:  body_tracking_node     -> /zed_fusion/body_trk/skeletons   (zed_msgs/ObjectsStamped, frame fusion_world)
              (fusion + AprilTags)   -> TF tag_frame -> fusion_world, /zed_fusion/diagnostics
                   |
              human_mapping_node     -> /human/joint_commands            (sensor_msgs/JointState, 11 joints)
                                     -> /human/body_colliders            (g1_cbf_msg/CapsuleArray)
              [zed_hand_mapping_node -> /human/hand_colliders, /hand_finger_angles; off by default]
              human_collider_merge_node -> /human/colliders              (g1_cbf_msg/CapsuleArray, body + hand)
                   |
              human_collider_viz_node -> /human/collider_markers         (MarkerArray, RViz only)
```

### `zed_launcher`

`server.launch.xml` includes the upstream `zed_camera.launch.py` once per camera, selected by serial number, and uses `param_overrides` to turn off every wrapper feature (depth, tracking, all publishers) except the stream server. A running stream keeps its resolution and frame rate, so the server must be restarted after changing them.

`body_tracking.launch.xml` starts `body_tracking_node`, which runs `ZedBodyFusionNode` and `ApriltagFusionNode` in one process (multi-threaded executor, `/zed_fusion` namespace). `zed_body_fusion_node` and `apriltag_fusion_node` are still built as standalone executables but no launch file starts them. `zed_mini.launch.xml` and `body_tracking_single_camera.launch.py` both include `body_tracking.launch.xml` and only override arguments; `zed_mini.launch.xml` sets `input_mode:=live`, `enable_apriltag_fusion:=false`, the ZED Mini serials, `calibration/zedm/`, VGA at 100 fps, `HUMAN_BODY_FAST` / `BODY_18` with body fitting off, and a 60 Hz Fusion loop. An XML `<include>` silently ignores arguments the included file does not declare, so a renamed argument there fails at runtime, not at launch.

`test/test_body_tracking_launch_contract.py` pins about 50 argument defaults of `body_tracking.launch.xml` and `server.launch.xml` (and the RViz overlay topics). Changing one of those defaults means updating the test. Per-setup values belong in a wrapper launch file such as `zed_mini.launch.xml`.

The fusion node does not consume ROS topics from the wrapper. It uses the ZED SDK directly: for each configured camera it opens an `sl::Camera` (from the network stream when `input_mode:=stream`, or from the local device by the calibration's serial when `live`), enables per-camera body tracking, and calls `startPublishing` over shared memory. Each camera has its own grab thread (`runCameraWorker`). An in-process `sl::Fusion` subscribes to all of them, and a wall timer (`processFusion`) retrieves fused bodies and publishes them. Points to know:

- Cameras are configured by three parallel lists: `camera_names`, `camera_serials`, `stream_ports` (same length, unique, even ports, validated even in `live` mode). Only calibration entries whose serial is in `camera_serials` are used, in that order, and every listed serial must be in the calibration. `fusion_minimum_allowed_cameras` and `camera_body_fallback_minimum_cameras` must not exceed the camera count, or the node throws.
- Calibrations are ZED360 output: `calibration/zedx/calibration.json` (three ZED X) and `calibration/zedm/calibration.json` (two ZED Minis). New cameras need both the calibration and the serial list updated. `body_tracking_single_camera.launch.py` requires exactly three cameras in its calibration.
- Launch and calibration files are installed to the package share directory, so adding or renaming one needs a `colcon build` of `zed_launcher` even with `--symlink-install`.
- The SDK is configured for ROS conventions (`RIGHT_HANDED_Z_UP_X_FWD`, metres), so no axis conversion happens downstream. Camera poses (calibration pose plus live SDK gravity) are published as static TFs `fusion_world -> <camera_name>_left_camera_optical_frame` once Fusion is running.
- The TF chain is `pelvis -> tag_frame -> fusion_world -> cameras`. The robot publishes `pelvis -> tag_frame`; `ApriltagFusionNode` publishes only `tag_frame -> fusion_world`, from two 12 cm `APRILTAG_36h11` tags (IDs 0 and 1) detected with OpenCV ArUco (contrib `aruco` on OpenCV < 4.7, `objdetect` otherwise; chosen in CMake). Tag detection runs on its own thread at up to 20 Hz on the newest frame per camera, optionally refined with same-grab depth. `ros2 run zed_launcher generate_apriltag_prints` makes printable tags.
- With body fitting on, Fusion returns BODY_34 for BODY_18 input; the published `body_format` is whatever Fusion returns (`body_format_utils.hpp`).
- `single_body_enabled` publishes only one skeleton under a stable `single_body_logical_id`. Selection is sticky (`single_body_switch_margin` for `single_body_switch_frames` frames), short gaps are bridged, and when Fusion drops the body it can fall back to agreeing per-camera bodies (`camera_body_fallback_*`).
- `debug:=true` publishes one combined skeleton + AprilTag overlay image per camera (`<camera>/zed_node/rgb/color/rect/body_tracking_overlay`) and per-camera skeletons. No raw image or depth topics are published.
- `role` accepts only `local` (`remote` / `stream_client` are aliases for local + stream).
- The launch file is what actually runs; do not assume its defaults match the `declare_parameter` defaults in the C++.

`zed_skeleton_overlay_node` is a standalone topic-based overlay drawer. It is built and installed but not started by any launch file.

### `human_mapping`

`human_mapping_node` takes the highest-confidence body from the fused skeleton topic. BODY_18, BODY_34 and BODY_38 keypoints are remapped onto the BODY_38 indices (constants at the top of `human_mapping_node.hpp`; BODY_18's pelvis is the hip midpoint). Points go through a root-relative filter (pelvis filtered as the root, other points relative to it, velocity prediction while the SDK reports SEARCHING). Capsules come from those points; joint angles are estimated **only for BODY_38**, otherwise the joint targets stay at `q_home`. Angles then go through EMA with rate limiting, neutral-pose subtraction, and `q_home + sign * gain * delta + bias` clamped to `[q_min, q_max]`. A timer republishes the latest result at `publish_rate_hz`.

- Neutral calibration: for `startup_delay_sec` after node start the output delta is zero, then angles are averaged (circular mean for yaw joints) for `neutral_calibration_duration` and used as the zero pose. The person must hold a neutral pose during that window.
- If no valid skeleton arrives within `stale_timeout_sec`, it publishes `q_home` and an empty capsule array rather than holding the last pose.
- The joint order is fixed at 11 joints: waist yaw/roll/pitch, then shoulder pitch/roll/yaw and elbow per arm (left first). `joint_names`, `q_home`, `signs`, `gains`, `bias`, `q_min`, `q_max` in `config/human_mapping.yaml` must all have exactly 11 entries in that order. `AngleFilter` also has its own hard-coded per-joint limits in the header, applied before the YAML limits.
- `human_collider_merge_node` republishes the body capsules on `/human/colliders`, appending hand capsules only when `enable_hand_mapping:=true` (which also starts `zed_hand_mapping_node`; it requires BODY_38).
- Launch arguments override the YAML for topic names, `publish_rate_hz` and every viz-node parameter.
- `collision_geometry` (`capsules`, `boxes`, `spheres`) and the `sphere_*` parameters only change how `human_collider_viz_node` draws the markers. `/human/colliders` is always capsules.

### `g1_cbf_msg`

Defines `Capsule` and `CapsuleArray`. The package name and message layout, and the `/human/joint_commands` / `/human/colliders` topics, are shared with the G1 control-barrier-function consumer outside this repo, so treat them as an external interface.

## Code conventions

- Each node is split into `include/<pkg>/<name>_node.hpp` (class declaration), `src/<name>.cpp` (implementation), and `src/<name>_node.cpp` (`main`). Older nodes `#include` the implementation `.cpp` from `_node.cpp` and list only that file in CMake; newer ones (`apriltag_fusion_node`, `body_tracking_node`, the tests) list the implementation `.cpp` files in `add_executable` / `ament_add_gtest`. Follow whichever pattern the target already uses.
- Launch files are XML, except `body_tracking_single_camera.launch.py`, which needs Python to read the calibration.
- Skeleton, joint-target and collider topics are published best-effort (`SensorDataQoS` in the fusion node, best-effort keep-last-1 in `human_mapping`). A reliable subscriber will not receive them.
