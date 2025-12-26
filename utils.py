import os
import uuid
from datetime import datetime

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def unique_name(prefix: str, ext: str):
    return f"{prefix}_{timestamp()}_{uuid.uuid4().hex}.{ext}"
