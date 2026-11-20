"""测试环境适配（DSH 沙箱）。

Windows 下 DSH 文件沙箱会拦截 pytest 内置 tmp_path 的创建/清理
（采用 \\\\?\\ 扩展路径，访问被拒绝）。这里用项目内临时目录覆盖内置
tmp_path fixture，使测试在无头/沙箱环境也能稳定运行。
"""
import shutil
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def tmp_path():
    base = ROOT / ".test_tmp"
    base.mkdir(exist_ok=True)
    path = base / f"case_{uuid.uuid4().hex}"
    path.mkdir()
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(autouse=True)
def _isolate_ops_paths(tmp_path, monkeypatch):
    """(功能五) 把运维日志/参数路径隔离到每个测试的临时目录，避免污染工作区。

    显式在 Settings 中传入 ops_query_log_path / ops_param_store_path 的测试不受影响
    （构造参数优先于环境变量）。
    """
    monkeypatch.setenv("OPS_QUERY_LOG_PATH", str(tmp_path / "ops" / "query_log.jsonl"))
    monkeypatch.setenv("OPS_PARAM_STORE_PATH", str(tmp_path / "ops" / "runtime_params.json"))
