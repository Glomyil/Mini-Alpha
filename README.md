# Mini-Alpha

A 股（含退市）中低频量化研究装置：原始 parquet 到回测报告，一条命令跑完。
判据是**口径可信 + 研究可复现**，不是「找一个赚钱策略」。

---

## 0. 这是什么，不是什么

**是什么**：把「下载 - 清洗 - 宽表 - 指标 - 因子 - 分层 - 撮合 - 报告」串起来的研究装置。
每一步都能单独跑、单独验收；口径写在代码里，且只允许有一个定义处。

**不是什么**：能直接上实盘的策略库。多空是**纯度诊断**（A 股个股做空基本不可行），
可执行的只有「末层相对基准的超额」。

---

## 1. 安装

需要 **Python ≥ 3.10**

```powershell
pip install -r requirements.txt
```

| 类别 | 包 | 用途 |
| :-- | :-- | :-- |
| 数据 | `baostock` `pandas` `numpy` `pyarrow` | 下载 / 计算 / 读写 parquet |
| 校验 | `duckdb` | 全库横截面体检（`ops/health_check.py`）|
| 配置与日志 | `pyyaml` `loguru` | `config/application.yml` / `logs/run.log` |
| 出图与网络 | `matplotlib` `requests` | 报告净值图 / HTTP |

装完自检：

```powershell
python -c "import baostock,pandas,numpy,pyarrow,duckdb,yaml,loguru,matplotlib,requests; print('deps ok')"
```

---

## 2. 快速开始

### 2.1 拿数据

数据不入库（见 `.gitignore` 的 `data/storage/`），首次使用要自己下。

```powershell
# 全量下载到今日：数小时量级，断了重跑同一条即可续
python -m data.fetcher.baostock_stock_fetcher --start 2015-01-01

# 之后每日增量
python -m data.fetcher.baostock_stock_fetcher --update
```

`--start` 按你的起点改，`--end` 默认给到今天。baostock 是**单会话**账号：
下载在跑时不要同时运行任何会 `bs.login()` 的脚本，否则会把它踢掉（报 `10001001`）。

### 2.2 清洗 → 面板 → 回测

一条命令串完：

```powershell
python -m data.cleaner.pipeline all
```

也可以分段跑：

```powershell
python -m ops.clean_all          # 只看清洗过程，不落盘（约 4 分钟）
python -m data.cleaner.clean_to_disk   # 清洗并落盘到 data/storage/cleaner_daily/
python -m backtest.panel         # 净数据 → 9 张宽表（首次 30–90 s，之后命中缓存约 1 s）
python main.py                   # 逐因子回测 + 出报告
```

### 2.3 产出在哪

| 文件 | 内容 |
| :-- | :-- |
| `reports/report_<因子>_close.md` | 每个因子一份报告（分层表 / IC / 多空 / 口径声明）|
| `reports/nav_<因子>_close.png` | 分层净值曲线 |
| `logs/run.log` | 每次运行的关键数字 |
| `data/storage/cleaner_meta.json` | 清洗总账（只数 / 行数 / 失败清单 / 源指纹）|

### 2.4 样本外复现性检验

```powershell
python -m ops.oos_compare              # 切点读 yml 的 research.cut
python -m ops.oos_compare --cut 2020-01-02 --factors amihud_20d
```

判据见 §7。结果只打印到屏幕，记账人工写进 `docs/dev/oos/ledger.md`。

---

## 3. 目录与数据流

```
data/storage/stock_daily/      原始 parquet（5,477 只，含退市）
      |  data/cleaner/*        6 个环节：align -> suspension -> price
      |                                    -> limits -> adjust -> derived
      v
data/storage/cleaner_daily/    净数据（约 958 MB）——「原料」
      |  backtest/panel.py     把列搬成宽表（FIELDS）
      v
9 张宽表（交易日 × 5,477 只），缓存 292 MB
      hfq_close / hfq_open / ret_1d_hfq / amount      价格与成交
      can_trade / reason                              能不能买卖 / 为什么不能
      float_mv / industry / is_st                     中性化与股票池口径
      |  backtest/features.py  指标（ret / vol / ma_dev / amihud）
      |  backtest/factors.py   因子注册表（6 个）
      |  backtest/weights.py   横截面名次 -> n 层 -> 层内等权
      |  backtest/engine.py    T+1 开盘撮合，8 本账
      |  backtest/metrics.py   年化 / 波动 / 回撤 / IC / 多空 / t
      v  backtest/report.py    md + 净值图 + 口径声明
reports/
```

清洗与回测之间**没有隐藏依赖**：清洗层只产出净数据，面板层只读净数据，
回测层只读宽表。每一层的输入输出都是落盘的 parquet，可以单独检查。

---

## 4. 怎么扩展

先问一句：我要的指标在 `panels` 的 key 里能找到吗？

| 情况 | 改哪里 |
| :-- | :-- |
| 指标已有，只是加个因子 | `backtest/factors.py` 加一行 |
| 要用已有指标做组合 | 同上，`fn` 里做运算 |
| 净数据里有、宽表里没有 | `backtest/panel.py` 两处 |
| 宽表里也没有 | `backtest/features.py` + `backtest/factors.py` |

### 4.1 加一个因子（最常见，只改 1 处），以十日反转因子为例：

```python
"rev_10d": {
    "fn": lambda f, p: f["ret_10"],       # 取哪个指标列
    "needs": [("ret", {"n": 10})],        # 要算哪些指标
    "requires": ["ret_1d_hfq"],           # 要哪些面板表
    "sign": -1,                           # 方向只乘这一次
    "note": "10 日反转",
},
```

### 4.2 加一张面板宽表

`backtest/panel.py` 三处：① `FIELDS` 里加名字 ② `build_panels` 里加一个收集列表 + `append`
③ `return` 的字典里加一项。**要 `fillna` 的列必须是 float** —— 整数 dtype 在 `concat`
时用 0 填缺失而不是 NaN，会把「未上市」变成「正常值」。

### 4.3 加一个新指标

`backtest/features.py` 写一个函数（进 `FEATURES` 注册表），再在 `factors.py` 的 `needs` 里引用。

约束：输出的 index/columns 必须与 panels 一致（千万千万不要 `dropna` 或是 `reindex`），
预热段必须显式写成 NaN，并 `astype("float32")`。

---

## 5. 口径（改之前先读，这些是硬约定）

| 项 | 约定 |
| :-- | :-- |
| 年化常数 | 日频指标 **244** 交易日/年；多空/层差按**调仓频率**（月 12 / 季 4 / 周 52）。两套不可混用 |
| 成交与估值 | 调仓日 T 收盘算权重，**T+1 开盘成交**（缺价即不成交，绝不 ffill）；估值用收盘 |
| 复权 | 后复权，`hfq = close × back_adj_factor` |
| `can_trade` | **市场事实**（停牌 / 涨跌停 / 无价），研究口径永远不改它 |
| 剔 ST | 把因子值置 NaN，**不是**改 `can_trade` |
| 分层 | 横截面名次 → n 层 → 层内等权 `1/只数`；因子值的绝对大小不进入权重 |
| 换手 | 单边 = `0.5 × Σ|Δw|`，按调仓期累计 |
| 多空 | `∏(1 + r_末 − r_首)`，**不是两个净值之比**（净值比会让弱势层越难看结果越好看）|
| 风险口径 | `reason`：0 正常 / 1 停牌 / 2 涨停买不到 / 3 跌停卖不掉 / 4 不在市 / 5 有行无价 |

名词的完整定义在 `GLOSSARY.md`，字段契约在 `docs/dev/_contracts/`。

---

## 6. 命令大全（终端直接运行即可）

| 命令 | 用途 |
| :-- | :-- |
| `python main.py` | 逐因子跑全链 + 出报告 |
| `python -m ops.oos_compare` | 样本外复现性检验（§7）|
| `python -m ops.health_check` | 全库数据体检（约 33 s）|
| `python -m ops.panel_probe` | 独立探针，重算一遍面板统计做对账 |
| `python -m pytest --ignore=qa/heavy -q` | 回归闸门（不需要数据）|
| `python -m pytest qa/heavy -q` | 需要真面板的重测试 |

`qa/` 存有pytest测试文件，`tests/` 存有早期独立验收脚本。

---

## 7. 样本外判据

用于评估一个因子在过去呈现的规律，在未来是否会重新复现

| # | 判据 | 门槛 |
| :-: | :-- | :-- |
| 1 | 方向 | 验证段 IC 均值 > 0（因子值已含 sign）|
| 2 | IC-IR | 验证 >= 0.5 × 训练 |
| 3 | 多空 t | 验证段 `t_nw >= 1.5`（用调仓期收益差，不是日频）|
| 4 | p 双条件 | `p(多空t) < 0.05` 且 `p(形状重合 rho) < 0.05`（对期 bootstrap）|
| 5 | IC 同号 | 两段 IC 均值同号 |

- 切点变量cut默认 `2020-01-02`，两段**各自独立起跑**（不是跑全期再截取）
- OOS 固定 **10 层**（5 层会把最差 10% 稀释一半）
- 判决只记「通过 / 不通过」到 `docs/dev/oos/ledger.md`，不记收益数字
- 个人认为OOS的样本外校验实际上只负责判断重合度，收益本身并不在我们的考察范围内
---

## 8. 配置中心

参数在 `config/application.yml`，打开对应文件根据内设规则自行修改参数即可，代码里用 `config.get('research.xxx')` 读。
改参数不用改代码；`research` 节写了未登记的键会直接报错（拼错的键不会静默回落默认值）。

影响结果的参数会进 `RunSpec` 指纹（`cfg_hash`），每次运行打印并快照到
`logs/config_snapshot/<时间>_<hash>/`。所以任何一个数字都能回答「哪份代码 + 哪份数据 + 哪套参数」。

---

## 9. 局限与路线图

| 阶段 | 内容 |
| :-: | :-- |
| 完成 | 单因子研究闭环（清洗 → 宽表 → 指标 → 因子 → 分层 → 报告）+ 基准线 + 成本敏感性 |
| 完成 | 基本面数据下载器 + 面板中性化（行业 / ln 流通市值）|
| 进行中 | 基本面清洗链：五表 → 事件宽表 → 日频 as-of → 挂进面板 |
| 之后 | 多因子合成与相关性去重 → 组合层 → 执行层（容量 / 融券）|

已知但不会影响单因子回测结果的交易假设（现金透支、无订单批、无容量上限、无融券费…）逐条记在
`TECH_DEBT.md`

---

## 10. 数据说明与免责

- 行情来自 **baostock**，受其使用条款约束（禁止并发连接、有每日请求上限等因素）；数据文件**不入库**。
- 本仓库是研究装置，不构成任何投资建议。回测结果受成交假设、成本模型等因素影响，只用于判断因子的可行性和质量
- 历史回测好看不代表未来有效；`docs/dev/oos/ledger.md` 记录的就是哪些结论在样本外没站住。

---

## 文档索引

| 文件 | 内容 |
| :-- | :-- |
| `PITFALLS.md` | 踩坑实录：症状 / 根因 / 实验证据 / 教训 |
| `TECH_DEBT.md` | 已知但故意没修的假设，逐条标了触发条件 |
| `GLOSSARY.md` | 术语与口径的唯一权威（含命名纪律）|
| `docs/dev/_methodology.md` | 开发方法论：判据怎么定、什么算做完 |
| `docs/dev/_contracts/` | 数据契约：哪个文件负责哪一列 |
| `docs/dev/oos/ledger.md` | 样本外判决台账 |
