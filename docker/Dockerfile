FROM stereolabs/zed:5.3.0-py-devel-l4t-r36.4

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV PIP_ROOT_USER_ACTION=ignore
ENV ROS_DISTRO=humble

RUN apt update && apt upgrade -y

# Basic tools + ROS 2 apt repository
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg2 \
    lsb-release \
    ca-certificates \
    software-properties-common \
    && add-apt-repository universe \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo ${UBUNTU_CODENAME}) main" \
        > /etc/apt/sources.list.d/ros2.list \
    && apt-get update

# ROS 2 + dependencies
RUN apt-get install -y --no-install-recommends \
    ros-humble-desktop \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-rosidl-generator-dds-idl \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    python3-pip \
    libyaml-cpp-dev \
    libeigen3-dev \
    libboost-all-dev \
    libspdlog-dev \
    libfmt-dev \
    sudo \
    tmux

COPY dependencies/ /workspace/ros2_ws/src/dependencies

WORKDIR /workspace/ros2_ws

SHELL ["/bin/bash", "-c"]

RUN rosdep init
RUN rosdep update
RUN rosdep install --from-paths src --ignore-src -r -y

RUN source /opt/ros/humble/setup.bash && colcon build --symlink-install --parallel-workers $(( $(nproc) / 2 ))

COPY docker/entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]

