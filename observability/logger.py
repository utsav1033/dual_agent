import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Optional


class ObservabilityLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / f"interactions_{datetime.now().strftime('%Y%m%d')}.jsonl"

    def log(
        self,
        model: str,
        user_message: str,
        assistant_response: str,
        latency: float,
        blocked: bool = False,
        tool_used: Optional[str] = None,
        metadata: Optional[dict] = None,
    ):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "user_message": user_message,
            "assistant_response": assistant_response,
            "latency_seconds": round(latency, 3),
            "blocked": blocked,
            "tool_used": tool_used,
            "metadata": metadata or {},
        }
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def get_stats(self) -> dict:
        if not self.log_file.exists():
            return {}

        entries = []
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))

        if not entries:
            return {}

        stats = defaultdict(lambda: {
            "count": 0, "total_latency": 0.0, "blocked": 0, "tool_calls": 0
        })
        for e in entries:
            m = e["model"]
            stats[m]["count"] += 1
            stats[m]["total_latency"] += e["latency_seconds"]
            if e.get("blocked"):
                stats[m]["blocked"] += 1
            if e.get("tool_used"):
                stats[m]["tool_calls"] += 1

        result = {}
        for model, s in stats.items():
            n = s["count"]
            result[model] = {
                "total_interactions": n,
                "avg_latency_s": round(s["total_latency"] / n, 3) if n else 0,
                "blocked_count": s["blocked"],
                "tool_calls": s["tool_calls"],
            }
        return result

    def get_all_entries(self) -> list:
        if not self.log_file.exists():
            return []
        entries = []
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries
