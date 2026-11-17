"""
系统常量定义
AI 锁定，真人不动 —— 路径、超时、重试、随机种子等系统级参数。
"""

from pathlib import Path

# ==============================
# 路径常量
# ==============================

# 项目根目录
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# 数据存储根目录
DATA_STORAGE: Path = PROJECT_ROOT / "data" / "storage"

# ETF 日线数据目录
ETF_DAILY_DIR: Path = DATA_STORAGE / "etf_daily"

# ETF 60分钟线数据目录
ETF_60MIN_DIR: Path = DATA_STORAGE / "etf_60min"

# ETF 代码清单文件
ETF_LIST_FILE: Path = DATA_STORAGE / "etf_list.parquet"

# ETF 更新元信息文件
ETF_META_FILE: Path = DATA_STORAGE / "etf_meta.json"

# ==============================
# 股票数据目录（A股，含退市，raw + 官方因子）
# 2026-09-07 新增：品种正式切换为股票后使用
# ==============================

# 股票日线目录（每只一文件：sh.600000.parquet）
STOCK_DAILY_DIR: Path = DATA_STORAGE / "stock_daily"

# 股票下载元信息文件
STOCK_META_FILE: Path = DATA_STORAGE / "stock_meta.json"

# ==============================
# 清洗/派生数据目录（2026-09-10 新增）
# ==============================

# 清洗结果目录（净数据写回目标；执行器 pipeline.py 用它）
CLEANER_DAILY_DIR: Path = DATA_STORAGE / "cleaner_daily"

# 因子面板目录（预留：因子层落地时启用，先留注释不动）
# FACTOR_DIR: Path = DATA_STORAGE / "factors"

# ==============================
# 财务（基本面）数据目录（2026-11-13 新增，T-6.2）
# 每只股票一文件：data/storage/fundamental/{table}/{code}.parquet
#   table ∈ profit / growth / balance / operation / cashflow
# ==============================

FUNDAMENTAL_DIR: Path = DATA_STORAGE / "fundamental"

# 下载进度账本（断点续传 + 每日预算记账）
FUNDAMENTAL_STATE_FILE: Path = DATA_STORAGE / "fundamental_state.json"

# 每日请求预算（baostock 官方上限 5 万，留 2 千余量）
FUNDAMENTAL_DAILY_BUDGET: int = 48000

# 逐季请求间隔（秒，含抖动；比 K 线的 0.6 稍快，因为财报单次负载极小）
FUNDAMENTAL_SLEEP: float = 0.25

# ==============================
# 日志目录（2026-09-10 新增：原先散落在各脚本里硬编码）
# ==============================

LOG_DIR: Path = PROJECT_ROOT / "data" / "logs"

# ==============================
# 网络与超时常量
# ==============================

# Baostock 请求超时（秒）
REQUEST_TIMEOUT: float = 30.0

# 重试次数
MAX_RETRIES: int = 3

# 指数退避基数（秒）
RETRY_BACKOFF_BASE: float = 1.0

# ==============================
# Baostock 相关常量
# ==============================

# 批量下载批次大小（只，防止封 IP）
BAO_BATCH_SIZE: int = 50

# 批次间休眠间隔（秒）
BAO_BATCH_SLEEP: float = 1.5

# ==============================
# 随机种子
# ==============================

RANDOM_SEED: int = 42

# ==============================
# 数据文件格式
# ==============================

# Parquet 压缩算法
PARQUET_COMPRESSION: str = "snappy"
