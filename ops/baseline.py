import argparse
import statistics
import subprocess
from datetime import datetime
from pathlib import Path

REPORTS = Path("reports")
BASELINE = Path("BASELINE.md")
SECTION = "## 变更记录"


def read_report(path):
    """返回 {'ann': {层:年化%}, 'vol': {'0':波动%}, 'to': {层:累计换手}}。"""
    cols, ann, vol, to = None, {}, {}, {}
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s.startswith("|"):
            continue
        c = [x.strip() for x in s.strip("|").split("|")]
        if cols is None:                                     # 表头：记列名位置
            if "年化" in c and "累计换手" in c:
                cols = {n: i for i, n in enumerate(c)}
            continue
        if set("".join(c).replace(" ", "")) <= set("-:"):     # 对齐行
            continue
        if not c[0].isdigit():                               # 非层号
            continue
        ann[c[0]] = float(c[cols["年化"]].rstrip("%"))
        if c[0] == "0":
            vol["0"] = float(c[cols["波动"]].rstrip("%"))
        if c[0] in ("0", "4"):
            to[c[0]] = float(c[cols["累计换手"]])
    return {"ann": ann, "vol": vol, "to": to}


def mono(ann):
    """分层单调性：层序 0..4 与年化排名的相关系数；层不齐返回 None。"""
    if len(ann) != 5:
        return None
    return statistics.correlation([ann[str(i)] for i in range(5)], list(range(5)))


def num(a, b, unit="", fmt="{:.2f}"):
    """一个数：本次值 + (前值, 差)。"""
    out = fmt.format(a) + unit
    if b is None:
        return out + "(前-)"
    return out + "(前" + fmt.format(b) + unit + ", " + "{:+.2f}".format(a - b) + ")"


def pct(a, b):
    """一个数：本次值 + (前值, 相对变化%)。"""
    out = "{:.2f}%".format(a)
    if b is None or b == 0:
        return out + "(前-)"
    return out + "(前{:.2f}%, {:+.2f}%)".format(b, (a / b - 1) * 100)


def diff_text(cur, prev):
    """把一次对比渲染成一段话，放进「数字变化」列。"""
    p = prev or {}
    parts = [
        "年化 层0 " + num(cur["ann"]["0"], p.get("ann", {}).get("0"), "%"),
        "层4 " + num(cur["ann"]["4"], p.get("ann", {}).get("4"), "%"),
        "波动 层0 " + pct(cur["vol"]["0"], (p.get("vol") or {}).get("0")),
        "换手 层0 " + num(cur["to"]["0"], (p.get("to")
                       or {}).get("0"), "", "{:.4f}"),
        "层4 " + num(cur["to"]["4"], (p.get("to")
                    or {}).get("4"), "", "{:.4f}"),
    ]
    m, pm = mono(cur["ann"]), (mono(p["ann"]) if p else None)
    parts.append("单调性 " + (num(m, pm, "", "{:.2f}") if m is not None else "-"))
    return "；".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticket", required=True)
    ap.add_argument("--change", required=True)
    a = ap.parse_args()

    # ① 本次与上次的报告（本次 = 不含 _prev；上次 = 对应的 _prev）
    cur_files = [f for f in sorted(REPORTS.glob("report_*_close.md"))
                 if not f.stem.endswith("_prev")]
    if not cur_files:
        raise SystemExit(f"{REPORTS} 里没有 report_*_close.md")

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    rows = []
    for f in cur_files:
        factor = f.stem[len("report_"):-len("_close")]
        prev_path = f.with_name(f.stem + "_prev.md")
        cur = read_report(f)
        prev = read_report(prev_path) if prev_path.exists() else None
        rows.append(f"| {stamp} | {a.ticket} | {factor} {a.change} | "
                    f"{diff_text(cur, prev)} | 待人工确认 |")

    # ② 定位「## 变更记录」表尾，插入
    lines = BASELINE.read_text(encoding="utf-8").split("\n")
    start = next((i for i, l in enumerate(lines)
                 if l.startswith(SECTION)), None)
    if start is None:
        raise SystemExit(f"BASELINE.md 里找不到「{SECTION}」")
    hdr = align = last = None
    for i in range(start, len(lines)):
        if lines[i].startswith("|"):
            c = [x.strip() for x in lines[i].strip().strip("|").split("|")]
            if hdr is None:
                hdr = i
            elif set("".join(c).replace(" ", "")) <= set("-:"):
                align = align if align is not None else i
            else:
                last = i
        elif hdr is not None:
            break
    anchor = last if last is not None else (
        align if align is not None else hdr)
    lines[anchor + 1:anchor + 1] = rows
    BASELINE.write_text("\n".join(lines), encoding="utf-8")
    print(f"已追加 {len(rows)} 行（{stamp}）")
    for r in rows:
        print("  " + r[:110])


if __name__ == "__main__":
    main()
