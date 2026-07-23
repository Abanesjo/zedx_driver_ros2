from pathlib import Path
import re
import xml.etree.ElementTree as ET


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
LAUNCH_PATH = PACKAGE_ROOT / "launch" / "body_tracking.launch.xml"
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
