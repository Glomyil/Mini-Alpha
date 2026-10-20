import os
import pathlib
import re
import subprocess
import sys

import pytest
pytestmark = pytest.mark.needs_data
ROOT = pathlib.Path(__file__).resolve().parents[1]
TIMEOUT = 420  # 每个脚本最多 7 分钟（有脚本会挂死，比如演示脚本）

# 注意：不要放 test_config_loader —— 它是演示脚本（while True + sleep(3)），永远不结束
SCRIPTS = [
    "test_base", "test_base_guard", "test_data_loader",
    "test_align", "test_suspension", "test_adjust", "test_derived",
    "test_features", "test_factors", "test_assemble", "test_weights",
    "test_engine", "test_metrics", "test_report", "test_scheduler",
    "test_panel", "test_pipeline",
]
SLOW = {"test_panel", "test_pipeline"}

OK, BAD = "\u2705", "\u274c"

# 有些测试会在临时目录里建夹具；把 TEMP 指到工作区内，受限环境也写得进去
# PYTHONIOENCODING 必须显式给 utf-8：脚本会打印 ✅ / 中文，
# 默认 GBK 控制台下会 UnicodeEncodeError 直接崩（跟谁开的终端无关，CI 也一样）
TMP = ROOT / ".tmp"
TMP.mkdir(parents=True, exist_ok=True)
ENV = {**os.environ, "TEMP": str(TMP),
       "TMP": str(TMP), "PYTHONIOENCODING": "utf-8"}


@pytest.mark.parametrize("name", SCRIPTS)
def test_acceptance(name, request):
    if name in SLOW:
        request.node.add_marker(pytest.mark.slow)

    path = ROOT / "tests" / f"{name}.py"
    assert path.exists(), f"脚本不存在: {path}"

    # ① 独立进程跑（老脚本里有 sys.exit，绝不能 import ✓）
    p = subprocess.run(
        [sys.executable, "-B", str(path)],
        cwd=str(ROOT),                       # ★ 必须在项目根：脚本里都是相对路径
        env=ENV,                             # ★ 让夹具写进工作区的临时目录
        capture_output=True, text=True,
        encoding="utf-8", errors="replace",  # ★ 老脚本有中文
        timeout=TIMEOUT,                     # ★ 有脚本会挂死，必须设超时
    )
    out = (p.stdout or "") + (p.stderr or "")

    # ② 退出码必须 0
    assert p.returncode == 0, f"{name} 退出码 {p.returncode}\n{out[-3000:]}"

    # ③ 汇总行：优先用「结果: 通过 N / M」；没有就退化成数勾叉
    m = re.search(r"结果\s*[:：]\s*通过\s*(\d+)\s*/\s*(\d+)", out)
    if m:
        passed, total = int(m.group(1)), int(m.group(2))
    else:
        passed = out.count(OK)
        total = passed + out.count(BAD)

    # ④ 防 0/0 假通过（import 失败时一条断言都不会跑）
    assert total > 0, f"{name}: 一条断言都没跑（很可能 import 失败）\n{out[-3000:]}"
    assert out.count(BAD) == 0, f"{name}: 输出里有 ❌\n{out[-3000:]}"
    assert passed == total, f"{name}: 只过了 {passed}/{total}\n{out[-3000:]}"
