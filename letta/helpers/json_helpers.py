import json
from datetime import datetime
from typing import Any


def json_loads(data: Any):
    return json.loads(data, strict=False)


def json_dumps(data: Any, indent: int = 2):
    def safe_serializer(obj: Any):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, bytes):
            return obj.decode("utf-8", errors="replace")
        raise TypeError(f"Type {type(obj)} not serializable")

    return json.dumps(data, indent=indent, default=safe_serializer, ensure_ascii=False)
