#!/bin/bash
set -e

source /opt/ros/humble/setup.bash

cd /workspace/ros2_ws

rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --parallel-workers $(( $(nproc) / 2 ))
source /workspace/ros2_ws/install/setup.bash

if ! grep -qxF "#Entrypoint Setup" ~/.bashrc; then
    cat <<'EOF' >> ~/.bashrc

#Entrypoint Setup
source /opt/ros/humble/setup.bash
source /workspace/ros2_ws/install/setup.bash
export RMW_IMPLEMENTATION="rmw_cyclonedds_cpp"
export CYCLONEDDS_URI=file:///workspace/dependencies/cyclonedds.xml
alias mujoco='export LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/x86_64-linux-gnu'
export XLA_PYTHON_CLIENT_MEM_FRACTION=".50"
EOF
fi

exec bash
