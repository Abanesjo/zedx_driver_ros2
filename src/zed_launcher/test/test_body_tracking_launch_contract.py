from pathlib import Path
import re
import xml.etree.ElementTree as ET


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
LAUNCH_PATH = PACKAGE_ROOT / "launch" / "body_tracking.launch.xml"
SERVER_LAUNCH_PATH = PACKAGE_ROOT / "launch" / "server.launch.xml"
RVIZ_PATH = PACKAGE_ROOT / "rviz" / "body_fusion.rviz"


def test_body_tracking_launch_uses_combined_in_process_pipeline():
    launch = ET.parse(LAUNCH_PATH).getroot()
    executables = [node.attrib.get("exec") for node in launch.findall("node")]

    assert executables.count("body_tracking_node") == 1
    assert "zed_body_fusion_node" not in executables
    assert "apriltag_fusion_node" not in executables

    argument_names = {arg.attrib.get("name") for arg in launch.findall("arg")}
    parameter_names = {
        param.attrib.get("name")
        for node in launch.findall("node")
        for param in node.findall("param")
    }
    assert "publish_images" not in argument_names
    assert "publish_images" not in parameter_names

    overlay_parameters = [
        param
        for node in launch.findall("node")
        for param in node.findall("param")
        if param.attrib.get("name") == "publish_overlay_images"
    ]
    assert len(overlay_parameters) == 1
    assert overlay_parameters[0].attrib.get("value") == "$(var debug)"


def test_continuity_and_apriltag_defaults_and_parameter_plumbing():
    launch = ET.parse(LAUNCH_PATH).getroot()
    arguments = {
        arg.attrib["name"]: arg.attrib.get("default")
        for arg in launch.findall("arg")
    }
    body_tracking_node = next(
        node
        for node in launch.findall("node")
        if node.attrib.get("exec") == "body_tracking_node"
    )
    parameters = {
        param.attrib["name"]: param.attrib.get("value")
        for param in body_tracking_node.findall("param")
    }

    expected_defaults = {
        "camera_resolution": "HD1080",
        "camera_fps": "30",
        "fusion_publish_rate_hz": "20.0",
        "fusion_diagnostics_rate_hz": "1.0",
        "body_model": "HUMAN_BODY_ACCURATE",
        "body_format": "BODY_18",
        "confidence_threshold": "40.0",
        "single_body_switch_frames": "30",
        "single_body_logical_id": "0",
        "single_body_bridge_timeout_sec": "0.5",
        "single_body_bridge_max_speed_mps": "2.0",
        "camera_body_fallback_enabled": "true",
        "camera_body_fallback_minimum_cameras": "2",
        "camera_body_fallback_consensus_distance_m": "0.5",
        "fused_body_max_jump_m": "0.5",
        "body_prediction_timeout_sec": "1.0",
        "depth_mode": "NEURAL",
        "fusion_skeleton_smoothing": "0.1",
        "fusion_minimum_allowed_cameras": "1",
        "fusion_minimum_allowed_keypoints": "7",
        "sender_tracking_enabled": "true",
        "fusion_tracking_enabled": "true",
        "body_fitting_enabled": "true",
        "apriltag_max_detection_rate_hz": "20.0",
        "apriltag_use_depth": "true",
        "apriltag_depth_inner_margin_ratio": "0.20",
        "apriltag_depth_min_valid_samples": "25",
        "apriltag_depth_min_valid_fraction": "0.25",
        "apriltag_depth_plane_inlier_threshold_m": "0.015",
        "apriltag_depth_plane_max_rmse_m": "0.010",
        "apriltag_depth_max_pnp_translation_delta_m": "0.20",
        "apriltag_depth_max_pnp_rotation_delta_deg": "20.0",
        "apriltag_depth_max_size_error_fraction": "0.25",
        "apriltag_pnp_ambiguity_reprojection_margin_px": "0.25",
        "apriltag_pnp_prior_max_age_sec": "0.25",
        "apriltag_learn_tag_transform": "true",
        "apriltag_tag_transform_bootstrap_duration_sec": "2.5",
        "apriltag_tag_transform_bootstrap_min_samples": "30",
        "apriltag_tag_transform_pair_max_age_sec": "0.10",
        "apriltag_tag_transform_bootstrap_translation_outlier_m": "0.03",
        "apriltag_tag_transform_bootstrap_rotation_outlier_deg": "8.0",
        "apriltag_tag_transform_online_alpha": "0.01",
        "apriltag_tag_transform_max_translation_step_m": "0.002",
        "apriltag_tag_transform_max_rotation_step_deg": "0.25",
        "apriltag_tag_pair_baseline_orientation_weight": "0.0",
        "apriltag_fixed_tag_frame_z_m": "1.0",
        "apriltag_kalman_position_measurement_std_m": "0.010",
        "apriltag_kalman_yaw_measurement_std_deg": "1.5",
        "apriltag_kalman_linear_acceleration_std_mps2": "1.0",
        "apriltag_kalman_yaw_acceleration_std_degps2": "90.0",
        "apriltag_kalman_initial_linear_velocity_std_mps": "1.0",
        "apriltag_kalman_initial_yaw_rate_std_degps": "90.0",
        "apriltag_kalman_reset_sec": "0.5",
    }
    expected_parameter_sources = {
        "camera_resolution": "camera_resolution",
        "camera_fps": "camera_fps",
        "body_model": "body_model",
        "body_format": "body_format",
        "confidence_threshold": "confidence_threshold",
        "single_body_switch_frames": "single_body_switch_frames",
        "single_body_logical_id": "single_body_logical_id",
        "single_body_bridge_timeout_sec":
            "single_body_bridge_timeout_sec",
        "single_body_bridge_max_speed_mps":
            "single_body_bridge_max_speed_mps",
        "camera_body_fallback_enabled":
            "camera_body_fallback_enabled",
        "camera_body_fallback_minimum_cameras":
            "camera_body_fallback_minimum_cameras",
        "camera_body_fallback_consensus_distance_m":
            "camera_body_fallback_consensus_distance_m",
        "fused_body_max_jump_m": "fused_body_max_jump_m",
        "fusion_publish_rate_hz": "fusion_publish_rate_hz",
        "fusion_diagnostics_rate_hz": "fusion_diagnostics_rate_hz",
        "body_prediction_timeout_sec": "body_prediction_timeout_sec",
        "depth_mode": "depth_mode",
        "fusion_skeleton_smoothing": "fusion_skeleton_smoothing",
        "fusion_minimum_allowed_cameras":
            "fusion_minimum_allowed_cameras",
        "fusion_minimum_allowed_keypoints":
            "fusion_minimum_allowed_keypoints",
        "sender_tracking_enabled": "sender_tracking_enabled",
        "fusion_tracking_enabled": "fusion_tracking_enabled",
        "body_fitting_enabled": "body_fitting_enabled",
        "max_detection_rate_hz": "apriltag_max_detection_rate_hz",
        "use_depth": "apriltag_use_depth",
        "depth_inner_margin_ratio": "apriltag_depth_inner_margin_ratio",
        "depth_min_valid_samples": "apriltag_depth_min_valid_samples",
        "depth_min_valid_fraction": "apriltag_depth_min_valid_fraction",
        "depth_plane_inlier_threshold_m":
            "apriltag_depth_plane_inlier_threshold_m",
        "depth_plane_max_rmse_m": "apriltag_depth_plane_max_rmse_m",
        "depth_max_pnp_translation_delta_m":
            "apriltag_depth_max_pnp_translation_delta_m",
        "depth_max_pnp_rotation_delta_deg":
            "apriltag_depth_max_pnp_rotation_delta_deg",
        "depth_max_size_error_fraction":
            "apriltag_depth_max_size_error_fraction",
        "pnp_ambiguity_reprojection_margin_px":
            "apriltag_pnp_ambiguity_reprojection_margin_px",
        "pnp_prior_max_age_sec": "apriltag_pnp_prior_max_age_sec",
        "learn_tag_transform": "apriltag_learn_tag_transform",
        "tag_transform_bootstrap_duration_sec":
            "apriltag_tag_transform_bootstrap_duration_sec",
        "tag_transform_bootstrap_min_samples":
            "apriltag_tag_transform_bootstrap_min_samples",
        "tag_transform_pair_max_age_sec":
            "apriltag_tag_transform_pair_max_age_sec",
        "tag_transform_bootstrap_translation_outlier_m":
            "apriltag_tag_transform_bootstrap_translation_outlier_m",
        "tag_transform_bootstrap_rotation_outlier_deg":
            "apriltag_tag_transform_bootstrap_rotation_outlier_deg",
        "tag_transform_online_alpha": "apriltag_tag_transform_online_alpha",
        "tag_transform_max_translation_step_m":
            "apriltag_tag_transform_max_translation_step_m",
        "tag_transform_max_rotation_step_deg":
            "apriltag_tag_transform_max_rotation_step_deg",
        "tag_pair_baseline_orientation_weight":
            "apriltag_tag_pair_baseline_orientation_weight",
        "fixed_tag_frame_z_m": "apriltag_fixed_tag_frame_z_m",
        "kalman_position_measurement_std_m":
            "apriltag_kalman_position_measurement_std_m",
        "kalman_yaw_measurement_std_deg":
            "apriltag_kalman_yaw_measurement_std_deg",
        "kalman_linear_acceleration_std_mps2":
            "apriltag_kalman_linear_acceleration_std_mps2",
        "kalman_yaw_acceleration_std_degps2":
            "apriltag_kalman_yaw_acceleration_std_degps2",
        "kalman_initial_linear_velocity_std_mps":
            "apriltag_kalman_initial_linear_velocity_std_mps",
        "kalman_initial_yaw_rate_std_degps":
            "apriltag_kalman_initial_yaw_rate_std_degps",
        "kalman_reset_sec": "apriltag_kalman_reset_sec",
    }

    assert {
        name: arguments.get(name) for name in expected_defaults
    } == expected_defaults
    for parameter_name, argument_name in expected_parameter_sources.items():
        assert parameters.get(parameter_name) == f"$(var {argument_name})"

    # These legacy controls remain available as compatibility seed/tuning.
    assert "apriltag_initial_tag_frame_offset_m" in arguments
    assert "apriltag_learn_tag_separation" in arguments
    assert "apriltag_tag_separation_ema_alpha" in arguments
    assert "apriltag_tag_separation_max_innovation_m" in arguments


def test_server_stream_defaults_match_default_profile():
    launch = ET.parse(SERVER_LAUNCH_PATH).getroot()
    arguments = {
        arg.attrib["name"]: arg.attrib.get("default")
        for arg in launch.findall("arg")
    }

    assert arguments["camera_resolution"] == "HD1080"
    assert arguments["grab_frame_rate"] == "30"
    assert arguments["stream_target_framerate"] == "30"

    camera_includes = launch.findall("include")
    assert len(camera_includes) == 3
    for include in camera_includes:
        overrides = next(
            arg.attrib["value"]
            for arg in include.findall("arg")
            if arg.attrib.get("name") == "param_overrides"
        )
        assert (
            "stream_server.target_framerate:="
            "$(var stream_target_framerate)"
        ) in overrides
        assert (
            "general.grab_resolution:=$(var camera_resolution)"
        ) in overrides
        assert "general.grab_frame_rate:=$(var grab_frame_rate)" in overrides


def test_rviz_exposes_only_three_combined_camera_overlays():
    config = RVIZ_PATH.read_text(encoding="utf-8")
    expected_topics = {
        f"/zed_{camera}/zed_node/rgb/color/rect/body_tracking_overlay"
        for camera in ("left", "center", "right")
    }
    configured_topics = set(
        re.findall(
            r"Value: (/\S+/zed_node/rgb/color/rect/body_tracking_overlay)",
            config,
        )
    )

    assert configured_topics == expected_topics
    assert config.count("Class: rviz_default_plugins/Image") == 3
    assert "/rgb/color/rect/skeleton_overlay" not in config
    assert "/rgb/color/rect/apriltag_overlay" not in config
    assert "/rgb/color/rect/image" not in config
