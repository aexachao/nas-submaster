"""pytest 配置：将项目根目录加入 sys.path，使得 `from services.xxx import ...` 可以工作。"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
