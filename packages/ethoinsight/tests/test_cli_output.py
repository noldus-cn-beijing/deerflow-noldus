"""2026-06-08: save_output_json 文件权限修复 (spec §1 改动 B).

修复前 save_output_json 使用 tempfile.mkstemp() 建临时文件，默认 0o600，
os.replace 保留权限 → 最终指标 JSON 文件 0o600 → L-B catalog 校验器
(validate_catalog) 读不了 → 两层指标验证第二层静默失效。
"""

import json
import os
import stat

from ethoinsight.scripts._cli import save_output_json


def test_save_output_json_is_world_readable(tmp_path):
    out = tmp_path / "m_test.json"
    save_output_json(out, {"value": 1.23})
    mode = stat.S_IMODE(os.stat(out).st_mode)
    # red 锚点：修复前是 0o600（mkstemp 默认），validate_catalog 读不了
    assert mode == 0o644, f"expected 0o644, got {oct(mode)}"
    # 数据完整性不受影响
    assert json.loads(out.read_text())["value"] == 1.23
