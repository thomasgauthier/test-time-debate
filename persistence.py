import asyncio
import functools
import hashlib
import json
import os
import sqlite3
from typing import Any, Callable

import dspy


class DurableCache:
    def __init__(self, db_path: str = "swarm_state.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS call_cache (
                    key_hash TEXT PRIMARY KEY,
                    component_name TEXT,
                    intent_json TEXT,
                    result_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def _generate_key(self, component: Any, kwargs: dict) -> str:
        # Get current LM settings from DSPy
        lm = dspy.settings.lm

        # Capture the model and its specific parameters (temperature, max_tokens, etc.)
        model_info = {
            "model": getattr(lm, "model", "unknown"),
            "lm_kwargs": getattr(lm, "kwargs", {}),
        }

        # Create a stable representation of the intent including the model state
        intent = {
            "model_info": model_info,
            "component": component.__class__.__name__,
            "kwargs": kwargs,
        }

        # Sort keys to ensure stable hashing
        intent_str = json.dumps(intent, sort_keys=True, default=str)
        return hashlib.sha256(intent_str.encode()).hexdigest(), intent_str

    def get(self, component: Any, kwargs: dict) -> dspy.Prediction:
        key_hash, _ = self._generate_key(component, kwargs)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT result_json FROM call_cache WHERE key_hash = ?", (key_hash,))
            row = cursor.fetchone()
            if row and row[0]:
                data = json.loads(row[0])
                return dspy.Prediction(**data)
        return None

    def set(self, component: Any, kwargs: dict, result: dspy.Prediction):
        key_hash, intent_str = self._generate_key(component, kwargs)

        # Serialize the dspy.Prediction
        if hasattr(result, "toDict"):
            result_data = result.toDict()
        else:
            result_data = dict(result)

        result_json = json.dumps(result_data, default=str)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO call_cache (key_hash, component_name, intent_json, result_json) VALUES (?, ?, ?, ?)",
                (key_hash, component.__class__.__name__, intent_str, result_json),
            )


cache = DurableCache()


def durable_memo(func):
    @functools.wraps(func)
    async def wrapper(component, *args, **kwargs):
        from config import VERBOSE, console

        # Check cache first
        cached_result = cache.get(component, kwargs)
        if cached_result is not None:
            if VERBOSE:
                component_name = component.__class__.__name__
                if hasattr(component, "signature"):
                    component_name = f"{component_name}({component.signature.__name__})"
                console.print(f"[bold cyan]⚡ Cache Hit:[/] [cyan]{component_name}[/]")
            return cached_result
        else:
            key_hash, _ = cache._generate_key(component, kwargs)

            # Save the miss to disk for debugging
            # os.makedirs("cache_misses_kwargs", exist_ok=True)
            # with open(os.path.join("cache_misses_kwargs", f"{key_hash}.json"), "w") as f:
            #     json.dump(kwargs, f, indent=2, default=str)

            if VERBOSE:
                component_name = component.__class__.__name__
                if hasattr(component, "signature"):
                    component_name = f"{component_name}({component.signature.__name__})"
                console.print(f"[bold red]❌ Cache Miss:[/] [cyan]{component_name}[/] [dim]({key_hash})[/]")

        # Execute original function
        result = await func(component, *args, **kwargs)

        # Persist result
        cache.set(component, kwargs, result)
        return result

    return wrapper
