"""factors.py —— 因子注册表（只声明，不计算）

一个因子 = 一行：
    "名字": {"fn": …, "sign": ±1, "needs": [(指标名, {参数})], "requires": […], "note": "…"}

铁律：fn 只读 feats、只返回原始值；方向只写在 sign（±1）；窗口只写在 needs；缺表由 requires 声明。
加一个因子 = 加一行；加一个新指标 = 去 features.py。
跑法：python -m backtest.factors（字段自检+打印）／python tests/test_factors.py（注册表验收）
对照：教学文档/factors.py教学参考.md §7
"""
from __future__ import annotations


FACTORS = {
    "rev_5d":     {"fn": lambda f, p: f["ret_5"], "sign": -1,
                   "needs": [("ret", {"n": 5})], "note": "5 日反转"},


    "rev_20d":    {"fn": lambda f, p: f["ret_20"], "sign": -1,
                   "needs": [("ret", {"n": 20})], "note": "20日反转"},

    # 2026-11-13：sign 由 +1 改为 −1。实测 IC 全负（README:426 −0.04414；
    #   四档池 −0.040/−0.074，中性化后仍为负）⇒ 在 A 股 120 日窗口它是【反转】不是动量。
    #   改后因子值 = −ret_120，与 rev_* 同族（只是窗口更长）⇒ 名字里的 mom 是历史包袱。
    "mom_120d":   {"fn": lambda f, p: f["ret_120"], "sign": -1,
                   "needs": [("ret", {"n": 120})], "note": "120日反转（原按动量用，实测方向相反）"},

    "vol_20d":    {"fn": lambda f, p: f["vol_20"], "sign": -1,
                   "needs": [("vol", {"n": 20})], "requires": ["ret_1d_hfq"], "note": "20天价格波动率"},

    "ma_dev_20d": {"fn": lambda f, p: f["ma_dev_20"], "sign": -1,
                   "needs": [("ma_dev", {"n": 20})], "note": "20天价格偏离均值程度"},

    "amihud_20d":  {"fn": lambda f, p: f["amihud_20"], "sign": +1,
                   "needs": [("amihud", {"n": 20})], "requires": ["amount"], "note": "非流动性"},

    # 因子格式示例：（以后自己往里加）
    # "rev_5d": {"fn": lambda f, p: f["ret_5"], "sign": -1,
    #            "needs": [("ret", {"n": 5})], "note": "5 日反转"},
}
if __name__ == "__main__":
    # 冒烟：字段自检 + 打印（真算因子要等 assemble.py）
    from backtest.features import FEATURES
    bad = []
    for name, spec in FACTORS.items():
        for key in ("fn", "sign", "needs"):
            if key not in spec:
                bad.append(name + " 缺 " + key)
        if spec.get("sign") not in (1, -1):
            bad.append(name + " 的 sign = " +
                       repr(spec.get("sign")) + "（只能 +1 / -1）")
        for fname, kw in spec["needs"]:
            if fname not in FEATURES:
                bad.append(name + " 要的指标 " + fname + " 不在 FEATURES 里")
            if not isinstance(kw, dict):
                bad.append(name + " 的 " + fname + " 参数不是 dict")
    for name, spec in FACTORS.items():
        print(f"  {name:<10} sign={spec['sign']:+d}  needs={spec['needs']}"
              f"  requires={spec.get('requires', [])}  {spec.get('note', '')}")
    print(f"共 {len(FACTORS)} 个因子 ｜ 字段自检 " + ("❌ " + str(bad) if bad else "✅"))
    print("下一步：assemble.py（按 needs 并集算 + 会话缓存 + requires 跳过 + 相关性小表）")
