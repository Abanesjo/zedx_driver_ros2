#!/bin/bash
set -e

source /opt/ros/${ROS_DISTRO}/setup.bash
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
source /opt/ros/${ROS_DISTRO}/setup.bash
source /workspace/ros2_ws/install/setup.bash
export RMW_IMPLEMENTATION="rmw_zenoh_cpp"
export ZENOH_ROUTER_CONFIG_URI=/workspace/ros2_ws/src/zedx_driver/RMW_ZENOH_ROUTER_CONFIG.json5
alias mujoco='export LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/x86_64-linux-gnu'
export XLA_PYTHON_CLIENT_MEM_FRACTION=".50"
export ZED_BOX_IP="192.168.0.133"
EOF
fi

exec bash
