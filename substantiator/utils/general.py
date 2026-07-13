from datetime import datetime
from pathlib import Path

PROJ_ROOT = Path(__file__).parent.parent.parent

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")