"""Launch body tracking and AprilTag processing for one calibrated ZED camera."""

import json
from pathlib import Path
from typing import NamedTuple


class CameraSelection(NamedTuple):
    """The physical camera selected by its ordered calibration index."""

    camera_id: int
    name: str
    serial_number: int
    stream_port: int


_CAMERA_NAMES = ("zed_left", "zed_center", "zed_right")
_STREAM_PORTS = (30000, 30004, 30002)
_FUSED_OVERLAY_TOPIC = (
    "/zed_fusion/zed_node/rgb/color/rect/body_tracking_overlay"
)


def load_camera_selection(calibration_path, camera_id):
    """Load one camera serial from the ordered ZED Fusion calibration."""

    try:
        selected_id = int(camera_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("camera_id must be one of 0, 1, or 2") from exc

    if str(camera_id).strip() != str(selected_id) or selected_id not in range(3):
        raise ValueError("camera_id must be one of 0, 1, or 2")

    path = Path(calibration_path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(
            f"Unable to read Fusion calibration '{path}': {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Fusion calibration '{path}' is not valid JSON: {exc}"
        ) from exc

    if not isinstance(document, dict) or len(document) != 3:
        raise RuntimeError(
            "Fusion calibration must contain exactly three ordered cameras"
        )

    ordered_serials = []
    seen_serials = set()
    for calibration_id, (calibration_key, entry) in enumerate(document.items()):
        try:
            fusion_configuration = entry["FusionConfiguration"]
            serial_number = fusion_configuration["serial_number"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                "Calibration camera "
                f"{calibration_id} is missing "
                "FusionConfiguration.serial_number"
            ) from exc

        if (
            isinstance(serial_number, bool)
            or not isinstance(serial_number, int)
            or serial_number <= 0
        ):
            raise RuntimeError(
                f"Calibration camera {calibration_id} has an invalid serial "
                "number"
            )

        try:
            key_serial = int(calibration_key)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Calibration camera {calibration_id} has a non-serial "
                "object key"
            ) from exc
        if key_serial != serial_number:
            raise RuntimeError(
                "Calibration camera "
                f"{calibration_id} key {calibration_key} does not match "
                f"serial {serial_number}"
            )
        if serial_number in seen_serials:
            raise RuntimeError(
                f"Fusion calibration contains duplicate serial {serial_number}"
            )
        seen_serials.add(serial_number)
        ordered_serials.append(serial_number)

    return CameraSelection(
        camera_id=selected_id,
        name=_CAMERA_NAMES[selected_id],
        serial_number=ordered_serials[selected_id],
        stream_port=_STREAM_PORTS[selected_id],
    )


def _configure_camera(context):
    from launch.actions import SetLaunchConfiguration
    from launch.substitutions import LaunchConfiguration

    calibration_path = LaunchConfiguration("fusion_config_path").perform(context)
    selection = load_camera_selection(
        calibration_path, LaunchConfiguration("camera_id").perform(context)
    )

    physical_overlay_topic = (
        f"/{selection.name}/zed_node/rgb/color/rect/body_tracking_overlay"
    )

    return [
        SetLaunchConfiguration(
            "selected_camera_names", f"['{selection.name}']"
        ),
        SetLaunchConfiguration(
            "selected_camera_serials", f"[{selection.serial_number}]"
        ),
        SetLaunchConfiguration(
            "selected_stream_ports", f"[{selection.stream_port}]"
        ),
        SetLaunchConfiguration(
            "selected_physical_overlay_topic", physical_overlay_topic
        ),
    ]


def generate_launch_description():
    from ament_index_python.packages import get_package_share_directory
    from launch import LaunchDescription
    from launch.actions import (
        DeclareLaunchArgument,
        GroupAction,
        IncludeLaunchDescription,
        OpaqueFunction,
    )
    from launch.substitutions import LaunchConfiguration
    from launch_ros.actions import SetRemap
    from launch_xml.launch_description_sources import XMLLaunchDescriptionSource

    package_share = Path(get_package_share_directory("zed_launcher"))
    calibration_path = package_share / "calibration" / "calibration.json"
    rviz_path = package_share / "rviz" / "body_fusion_single_camera.rviz"
    body_tracking_launch = (
        package_share / "launch" / "body_tracking.launch.xml"
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "camera_id",
                default_value="0",
                description=(
                    "Ordered calibration camera: 0=left, 1=center, 2=right"
                ),
            ),
            DeclareLaunchArgument(
                "fusion_config_path",
                default_value=str(calibration_path),
                description="ZED Fusion calibration JSON",
            ),
            DeclareLaunchArgument(
                "rviz_config",
                default_value=str(rviz_path),
                description="RViz configuration used when rviz:=true",
            ),
            DeclareLaunchArgument(
                "body_model",
                default_value="HUMAN_BODY_ACCURATE",
                description="Single-camera body detector quality profile",
            ),
            DeclareLaunchArgument(
                "depth_mode",
                default_value="NEURAL",
                description="Single-camera stereo depth quality profile",
            ),
            DeclareLaunchArgument(
                "confidence_threshold",
                default_value="40.0",
                description="Minimum SDK body detection confidence",
            ),
            DeclareLaunchArgument(
                "fusion_skeleton_smoothing",
                default_value="0.1",
                description="SDK Fusion fitted-skeleton smoothing",
            ),
            OpaqueFunction(function=_configure_camera),
            GroupAction(
                scoped=True,
                actions=[
                    SetRemap(
                        src=LaunchConfiguration(
                            "selected_physical_overlay_topic"
                        ),
                        dst=_FUSED_OVERLAY_TOPIC,
                    ),
                    IncludeLaunchDescription(
                        XMLLaunchDescriptionSource(str(body_tracking_launch)),
                        launch_arguments={
                            "camera_names": LaunchConfiguration(
                                "selected_camera_names"
                            ),
                            "camera_serials": LaunchConfiguration(
                                "selected_camera_serials"
                            ),
                            "stream_ports": LaunchConfiguration(
                                "selected_stream_ports"
                            ),
                            "fusion_config_path": LaunchConfiguration(
                                "fusion_config_path"
                            ),
                            "fusion_minimum_allowed_cameras": "1",
                            "camera_body_fallback_minimum_cameras": "1",
                            "publish_per_camera_skeletons": "false",
                            "sender_tracking_enabled": "false",
                            "fusion_tracking_enabled": "true",
                            "body_fitting_enabled": "true",
                            "rviz_config": LaunchConfiguration("rviz_config"),
                            "body_model": LaunchConfiguration("body_model"),
                            "depth_mode": LaunchConfiguration("depth_mode"),
                            "confidence_threshold": LaunchConfiguration(
                                "confidence_threshold"
                            ),
                            "fusion_skeleton_smoothing": LaunchConfiguration(
                                "fusion_skeleton_smoothing"
                            ),
                        }.items(),
                    ),
                ],
            ),
        ]
    )
