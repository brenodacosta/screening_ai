from pathlib import Path

import pytest


@pytest.fixture
def jd_path() -> Path:
    return Path(__file__).resolve().parent.parent / "jobs" / "delivery_driver.md"
