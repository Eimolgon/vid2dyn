import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # never open windows during tests

# Make `import utils`, `import bike_model_perspective` and `import src.xxx` work.
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src", Path(__file__).parent):
    sys.path.insert(0, str(p))

IMAGE_RES = (6020, 4024)
SENSOR = (0.0235, 0.0156)
FOCAL = 0.028
WHEEL_D = 0.6604


@pytest.fixture(scope="session")
def bike_params():
    return {"lr": 1.4, "lf1": 0.5, "lf2": 0.1,
            "rf": WHEEL_D / 2, "rr": WHEEL_D / 2}


@pytest.fixture(scope="session")
def camera_params():
    return {
        "fx": FOCAL * IMAGE_RES[0] / SENSOR[0],
        "fy": FOCAL * IMAGE_RES[1] / SENSOR[1],
        "cx": IMAGE_RES[0] / 2,
        "cy": IMAGE_RES[1] / 2,
        "cam_x": 0.0, "cam_y": 0.0, "cam_z": 0.0,
        "cam_yaw": 0.0, "cam_pitch": 0.0, "cam_roll": 0.0,
    }


@pytest.fixture(scope="session")
def bounds():
    """Same as main.py, except depth (y) must be in front of the camera."""
    lo = [-np.pi / 2] * 4 + [-20.0, 0.5, -20.0]
    hi = [np.pi / 2] * 4 + [20.0, 50.0, 20.0]
    return lo, hi