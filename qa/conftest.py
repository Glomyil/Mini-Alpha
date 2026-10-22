"""qa/conftest.py —— 把 tmp_path 放到项目内。

原因：DSH 沙箱不允许在系统 TEMP 下创建子目录（pytest 的 tmp_path 会在那里建），
      会直接 PermissionError。改到 logs/.pytest_tmp/ 下即可。
在 CI / 普通终端下同样有效，不影响原有行为。
"""
import shutil
import uuid
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parents[1] / "logs" / ".pytest_tmp"


@pytest.fixture
def tmp_path():
    BASE.mkdir(parents=True, exist_ok=True)
    p = BASE / uuid.uuid4().hex
    p.mkdir()
    try:
        yield p
    finally:
        shutil.rmtree(p, ignore_errors=True)
