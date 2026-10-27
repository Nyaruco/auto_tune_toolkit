#!/usr/bin/env python3
"""--scenes-file：换一套场景配置（场景文件加载器）。

工具里所有模块都是 `from scenes import SCENES/SCENE_BY_KEY/...` —— 名字写死在
'scenes' 上。所以换场景文件的办法是在 import 它们**之前**，把指定文件顶上
sys.modules['scenes'] 那个位置：之后不管是跑测（auto_tune.py）还是出表
（generate_report.py），拿到的都是那份场景集。

auto_tune.py / generate_report.py 两个入口都调这里，实现只有这一份。

被换上的场景文件（如 scenes_custom.py）写法照 scenes.py，一般只需定义
SCENES / SCENE_BY_KEY，其余可以 `from scenes import make_scene, CLIENT_BENCH, ...`
复用 —— 这里先 import 一遍真 scenes，所以那份文件里的 `from scenes import ...`
拿到的是原来那份（不会绕成自己）。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# 命令行写法：--scenes-file X / --scenes-file=X（--scenes 是给手打图省事的别名）
OPTS = ("--scenes-file", "--scenes")


def pick(argv) -> str:
    """从命令行参数里挑出 --scenes-file 的值；没写返回空串。"""
    path = ""
    for i, a in enumerate(argv):
        for opt in OPTS:
            if a == opt:
                path = argv[i + 1] if i + 1 < len(argv) else ""
            elif a.startswith(opt + "="):
                path = a.split("=", 1)[1]
    return path.strip()


def bootstrap(argv, base_dir) -> str:
    """按命令行里的 --scenes-file 换掉场景模块。

    返回用到的场景文件名（没指定就返回空串，此时一切照旧用同目录的 scenes.py）。
    文件不存在 / 里面写错了，当场退出并说清楚是哪个文件。
    """
    path = pick(argv)
    if not path:
        return ""

    base_dir = Path(base_dir).resolve()
    if str(base_dir) not in sys.path:          # 场景文件多半要 from scenes import ...
        sys.path.insert(0, str(base_dir))
    import scenes                             # noqa: F401  真 scenes 先进 sys.modules

    f = Path(path)
    if not f.is_absolute():
        f = base_dir / f
    if not f.is_file():
        sys.exit(f"--scenes-file 指定的场景文件不存在: {f}")

    spec = importlib.util.spec_from_file_location("scenes_alt", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)               # 场景文件自身报错就直接抛出来
    if not getattr(mod, "SCENES", None):
        sys.exit(f"--scenes-file 的 {f} 里没有 SCENES（照 scenes.py 写一个场景列表）")

    # 场景文件顶上 'scenes' 之后，别人 `from scenes import CLIENT_AIS / make_scene /
    # validate_scenes / ...` 也得能拿到 —— 那里没定义的从真 scenes 补齐，
    # 于是场景文件只要写 SCENES（+ 想覆盖的 MODEL / MODEL_NAME …）就行。
    for k, v in vars(scenes).items():
        if not k.startswith("_") and not hasattr(mod, k):
            setattr(mod, k, v)
    if not getattr(mod, "SCENE_BY_KEY", None):
        mod.SCENE_BY_KEY = {s["key"]: s for s in mod.SCENES}
    mod.validate_scenes(mod.SCENES)            # 传 SCENES —— 不传校验的是 scenes.py 那份

    sys.modules["scenes"] = mod
    print(f"[scenes] 场景集来自 {f.name}"
          f"（{len(mod.SCENES)} 个场景: {'、'.join(s['key'] for s in mod.SCENES)}）",
          file=sys.stderr)
    return f.name
