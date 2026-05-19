#!/bin/bash
set -e

source /opt/ros/humble/setup.bash
if [ -f /workspace/ros2_ws/install/setup.bash ]; then
    source /workspace/ros2_ws/install/setup.bash || \
        echo "[entrypoint] Warning: failed to source existing workspace setup; continuing." >&2
fi

cd /workspace/ros2_ws

apt update
rosdep install --from-paths src --ignore-src -r -y --skip-keys scout_description
if ! colcon build --symlink-install --parallel-workers $(( $(nproc) / 2 )); then
    echo "[entrypoint] Warning: colcon build failed; continuing so the container stays available." >&2
fi
if [ -f /workspace/ros2_ws/install/setup.bash ]; then
    source /workspace/ros2_ws/install/setup.bash || \
        echo "[entrypoint] Warning: failed to source workspace setup after build; continuing." >&2
fi

if ! grep -qxF "#Entrypoint Setup" ~/.bashrc; then
    cat <<'EOF' >> ~/.bashrc

#Entrypoint Setup
source /opt/ros/humble/setup.bash
source /workspace/ros2_ws/install/setup.bash
export RMW_IMPLEMENTATION="rmw_cyclonedds_cpp"
export CYCLONEDDS_URI=file:///workspace/ros2_ws/src/zedx_driver/cyclonedds.xml
alias mujoco='export LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/x86_64-linux-gnu'
export XLA_PYTHON_CLIENT_MEM_FRACTION=".50"
export CAM_MODEL="zedx"
export ZED_BOX_IP="192.168.123.200"
EOF
fi

exec bash
