import importlib.util
import json
from pathlib import Path
import sys

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
LAUNCH_PATH = (
    PACKAGE_ROOT / "launch" / "body_tracking_single_camera.launch.py"
)
RVIZ_PATH = PACKAGE_ROOT / "rviz" / "body_fusion_single_camera.rviz"


def load_launch_module():
    spec = importlib.util.spec_from_file_location(
        "body_tracking_single_camera_launch", LAUNCH_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_calibration(path, serials):
    document = {
        str(serial): {
            "FusionConfiguration": {
                "serial_number": serial,
            }
        }
        for serial in serials
    }
    path.write_text(json.dumps(document), encoding="utf-8")


@pytest.mark.parametrize(
    ("camera_id", "expected_name", "expected_port", "expected_serial"),
    [
        (0, "zed_left", 30000, 101),
        (1, "zed_center", 30004, 202),
        (2, "zed_right", 30002, 303),
    ],
)
def test_selection_uses_ordered_calibration_serial(
    tmp_path, camera_id, expected_name, expected_port, expected_serial
):
    calibration = tmp_path / "calibration.json"
    write_calibration(calibration, [101, 202, 303])
    launch_module = load_launch_module()

    selection = launch_module.load_camera_selection(calibration, camera_id)

    assert selection.name == expected_name
    assert selection.stream_port == expected_port
    assert selection.serial_number == expected_serial


@pytest.mark.parametrize("camera_id", [-1, 3, "left", "1.0", True])
def test_selection_rejects_invalid_camera_id(tmp_path, camera_id):
    calibration = tmp_path / "calibration.json"
    write_calibration(calibration, [101, 202, 303])
    launch_module = load_launch_module()

    with pytest.raises(ValueError, match="0, 1, or 2"):
        launch_module.load_camera_selection(calibration, camera_id)


def test_selection_rejects_malformed_or_mismatched_calibration(tmp_path):
    launch_module = load_launch_module()
    calibration = tmp_path / "calibration.json"
    write_calibration(calibration, [101, 202])
    with pytest.raises(RuntimeError, match="exactly three"):
        launch_module.load_camera_selection(calibration, 0)

    calibration.write_text(
        json.dumps(
            {
                "101": {"FusionConfiguration": {"serial_number": 101}},
                "202": {"FusionConfiguration": {"serial_number": 999}},
                "303": {"FusionConfiguration": {"serial_number": 303}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="does not match"):
        launch_module.load_camera_selection(calibration, 1)

    calibration.write_text(
        json.dumps(
            {
                "101": {"FusionConfiguration": {"serial_number": 101}},
                "0101": {"FusionConfiguration": {"serial_number": 101}},
                "303": {"FusionConfiguration": {"serial_number": 303}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="duplicate serial"):
        launch_module.load_camera_selection(calibration, 0)


def test_selection_validates_unselected_entries(tmp_path):
    launch_module = load_launch_module()
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "101": {"FusionConfiguration": {"serial_number": 101}},
                "202": {"FusionConfiguration": {"serial_number": 202}},
                "303": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="camera 2 is missing"):
        launch_module.load_camera_selection(calibration, 0)


def test_launch_contract_is_single_camera_fusion_with_canonical_overlay():
    source = LAUNCH_PATH.read_text(encoding="utf-8")

    assert "body_tracking.launch.xml" in source
    assert '"fusion_minimum_allowed_cameras": "1"' in source
    assert '"camera_body_fallback_minimum_cameras": "1"' in source
    assert '"publish_per_camera_skeletons": "false"' in source
    assert '"body_fitting_enabled": "true"' in source
    assert 'default_value="HUMAN_BODY_ACCURATE"' in source
    assert 'default_value="NEURAL"' in source
    assert 'default_value="40.0"' in source
    assert 'default_value="0.1"' in source
    assert "SetRemap" in source
    assert (
        "/zed_fusion/zed_node/rgb/color/rect/body_tracking_overlay"
        in source
    )

    rviz = RVIZ_PATH.read_text(encoding="utf-8")
    assert "/zed_fusion/body_trk/skeletons" in rviz
    assert (
        "/zed_fusion/zed_node/rgb/color/rect/body_tracking_overlay" in rviz
    )
    assert rviz.count("Class: rviz_default_plugins/Image") == 1
