"""Redis cache and Neon persistence for Free Teacher conversation memory."""

import asyncio
import json
import logging
import os
import threading
from typing import Any, Optional
from urllib.parse import quote

import httpx
import psycopg
from psycopg_pool import ConnectionPool

try:
    from redis import asyncio as redis_async
except ImportError:
    redis_async = None


logger = logging.getLogger(__name__)

NEON_DATABASE_URL = os.getenv("NEON_DATABASE_URL", "").strip()
REDIS_REST_URL = (
    os.getenv("UPSTASH_REDIS_REST_URL") or os.getenv("REDIS_REST_URL") or ""
).strip().rstrip("/")
REDIS_REST_TOKEN = (
    os.getenv("UPSTASH_REDIS_REST_TOKEN") or os.getenv("REDIS_REST_TOKEN") or ""
).strip()
REDIS_URL = (os.getenv("UPSTASH_REDIS_URL") or os.getenv("REDIS_URL") or "").strip()
MEMORY_TTL_SECONDS = max(60, int(os.getenv("FREE_TEACHER_MEMORY_TTL_SECONDS", "604800")))
MEMORY_MAX_TURNS = max(1, int(os.getenv("FREE_TEACHER_MEMORY_MAX_TURNS", "30")))
MEMORY_DB_POOL_MAX = max(2, int(os.getenv("NEON_POOL_MAX", "10")))

_redis_client = None
_memory_pool: Optional[ConnectionPool] = None
_memory_pool_opened = False
_memory_pool_open_lock = threading.Lock()
_memory_init_lock = asyncio.Lock()
_memory_persist_lock = asyncio.Lock()


def _clean_text(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def _test_neon_connectivity_sync() -> bool:
    if not NEON_DATABASE_URL:
        logger.error("[DB CHECK] NEON_DATABASE_URL is missing; memory is disabled.")
        return False
    try:
        with psycopg.connect(NEON_DATABASE_URL, connect_timeout=8) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1, current_database(), current_user, "
                    "to_regclass('public.free_teacher_memory')::text"
                )
                row = cur.fetchone()
        logger.info("[DB CHECK] NEON CONNECTED | database=%s | user=%s", row[1], row[2])
        if not row[3]:
            logger.error(
                "[DB CHECK] Table public.free_teacher_memory is missing; "
                "run the supplied SQL schema."
            )
            return False
        logger.info("[DB CHECK] MEMORY TABLE EXISTS | public.free_teacher_memory")
        return True
    except Exception:
        logger.exception("[DB CHECK] Neon connectivity check failed.")
        return False


async def startup_check() -> None:
    logger.info("[DB CHECK] Checking direct Neon memory storage...")
    if not NEON_DATABASE_URL:
        logger.error("[DB CHECK] NEON_DATABASE_URL is missing.")
        return
    await asyncio.to_thread(_test_neon_connectivity_sync)


def _get_open_memory_pool_sync() -> Optional[ConnectionPool]:
    global _memory_pool_opened
    if not NEON_DATABASE_URL:
        return None
    if _memory_pool is None:
        logger.error("Neon memory pool has not been initialized.")
        return None
    if not _memory_pool_opened:
        with _memory_pool_open_lock:
            if not _memory_pool_opened:
                try:
                    _memory_pool.open(wait=True, timeout=8)
                    _memory_pool_opened = True
                except Exception:
                    logger.exception("Could not open the Free Teacher Neon connection pool.")
                    return None
    return _memory_pool


async def _ensure_memory_clients() -> None:
    global _memory_pool
    async with _memory_init_lock:
        if NEON_DATABASE_URL and _memory_pool is None:
            _memory_pool = ConnectionPool(
                conninfo=NEON_DATABASE_URL,
                min_size=1,
                max_size=MEMORY_DB_POOL_MAX,
                open=False,
                kwargs={"autocommit": False, "prepare_threshold": None},
                timeout=8,
                name="free-teacher-memory",
            )


def _hydrate_turns_from_neon_sync(
    user_id: str, source: str, target: str, topic: str
) -> list[dict[str, str]]:
    if not NEON_DATABASE_URL or not user_id:
        return []
    pool = _get_open_memory_pool_sync()
    if pool is None:
        return []
    try:
        with pool.connection(timeout=8) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT turns FROM public.free_teacher_memory
                       WHERE user_id = %s AND source_language = %s
                         AND target_language = %s AND topic = %s
                       LIMIT 1""",
                    (user_id, source, target, topic or "general"),
                )
                row = cur.fetchone()
        if not row or row[0] is None:
            return []
        raw = json.loads(row[0]) if isinstance(row[0], str) else row[0]
        if not isinstance(raw, list):
            return []
        turns = [
            {
                "role": str(item.get("role", "assistant")),
                "text": str(item.get("text", ""))[:700],
                **({"at": str(item["at"])} if item.get("at") else {}),
            }
            for item in raw
            if isinstance(item, dict) and item.get("text")
        ]
        return turns[-MEMORY_MAX_TURNS:]
    except Exception:
        logger.exception("Free Teacher memory load from Neon failed.")
        return []


async def _hydrate_turns_from_neon(
    user_id: str, source: str, target: str, topic: str
) -> list[dict[str, str]]:
    try:
        await _ensure_memory_clients()
        return await asyncio.to_thread(
            _hydrate_turns_from_neon_sync, user_id, source, target, topic
        )
    except Exception:
        logger.exception("Free Teacher memory load task failed.")
        return []


async def _redis_memory_command(*args: str) -> Any:
    global _redis_client
    if REDIS_REST_URL and REDIS_REST_TOKEN:
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(4.0, connect=2.0)
            ) as client:
                response = await client.post(
                    REDIS_REST_URL,
                    headers={
                        "Authorization": f"Bearer {REDIS_REST_TOKEN}",
                        "Content-Type": "application/json",
                    },
                    json=list(args),
                )
                response.raise_for_status()
                body = response.json()
                if body.get("error"):
                    raise RuntimeError(str(body["error"]))
                return body.get("result")
        except Exception:
            logger.exception(
                "[REDIS] Upstash REST command failed. Check URL/token and write permissions."
            )
            return None

    if REDIS_URL:
        if redis_async is None:
            logger.error(
                "[REDIS] REDIS_URL is set, but the 'redis' package is missing. Run: uv add redis"
            )
            return None
        try:
            if _redis_client is None:
                _redis_client = redis_async.from_url(
                    REDIS_URL,
                    decode_responses=True,
                    socket_connect_timeout=3,
                    socket_timeout=4,
                    health_check_interval=30,
                )
            return await _redis_client.execute_command(*args)
        except Exception:
            logger.exception(
                "[REDIS] Redis protocol command failed. "
                "Check URL and ensure credentials allow GET/SET/EX."
            )
            return None

    logger.warning(
        "[REDIS] Not configured; set UPSTASH_REDIS_REST_URL + "
        "UPSTASH_REDIS_REST_TOKEN or REDIS_URL."
    )
    return None


def _redis_memory_key(user_id: str, source: str, target: str, topic: str) -> str:
    return "free_teacher_memory:" + ":".join(
        quote(str(part), safe="")
        for part in (user_id, source, target, topic or "general")
    )


async def _load_turns_from_redis(
    user_id: str, source: str, target: str, topic: str
) -> list[dict[str, str]]:
    result = await _redis_memory_command(
        "GET", _redis_memory_key(user_id, source, target, topic)
    )
    if not result:
        return []
    try:
        raw = json.loads(result) if isinstance(result, str) else result
        if not isinstance(raw, list):
            return []
        return [
            {
                "role": str(item.get("role", "assistant")),
                "text": str(item.get("text", ""))[:700],
                **({"at": str(item["at"])} if item.get("at") else {}),
            }
            for item in raw[-MEMORY_MAX_TURNS:]
            if isinstance(item, dict) and item.get("text")
        ]
    except Exception:
        logger.exception("[REDIS] Could not decode cached Free Teacher memory.")
        return []


async def _save_turns_to_redis(snapshot: dict[str, Any]) -> bool:
    user_id = _clean_text(snapshot.get("user_id"))
    source = _clean_text(snapshot.get("source_language"))
    target = _clean_text(snapshot.get("target_language"))
    topic = _clean_text(snapshot.get("topic"), "general")
    if not user_id or not source or not target:
        logger.error("[REDIS] Cannot save memory: user/language identifiers are missing.")
        return False
    turns = snapshot.get("turns", [])
    if not isinstance(turns, list):
        logger.error("[REDIS] Cannot save memory: turns must be a list.")
        return False
    value = json.dumps(turns[-MEMORY_MAX_TURNS:], ensure_ascii=False)
    result = await _redis_memory_command(
        "SET",
        _redis_memory_key(user_id, source, target, topic),
        value,
        "EX",
        str(MEMORY_TTL_SECONDS),
    )
    if result == "OK":
        logger.info("[REDIS] Free Teacher memory cache saved; turns=%s", len(turns))
        return True
    logger.error("[REDIS] Free Teacher memory cache was not saved.")
    return False


def _persist_snapshot_to_neon_sync(snapshot: dict[str, Any]) -> bool:
    if not NEON_DATABASE_URL:
        logger.error("Cannot save Free Teacher memory: NEON_DATABASE_URL is missing.")
        return False
    user_id = _clean_text(snapshot.get("user_id"))
    source = _clean_text(snapshot.get("source_language"))
    target = _clean_text(snapshot.get("target_language"))
    topic = _clean_text(snapshot.get("topic"), "general")
    if not user_id or not source or not target:
        logger.error(
            "Cannot save Free Teacher memory: "
            "user_id/source_language/target_language is missing."
        )
        return False
    turns = snapshot.get("turns", [])
    if not isinstance(turns, list):
        logger.error("Cannot save Free Teacher memory: turns must be a list.")
        return False
    pool = _get_open_memory_pool_sync()
    if pool is None:
        return False
    try:
        with pool.connection(timeout=8) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO public.free_teacher_memory
                       (user_id, source_language, target_language, topic, turns, updated_at)
                       VALUES (%s, %s, %s, %s, %s::jsonb, NOW())
                       ON CONFLICT (user_id, source_language, target_language, topic)
                       DO UPDATE SET turns = EXCLUDED.turns, updated_at = NOW()""",
                    (
                        user_id,
                        source,
                        target,
                        topic,
                        json.dumps(turns[-MEMORY_MAX_TURNS:], ensure_ascii=False),
                    ),
                )
            conn.commit()
        return True
    except Exception:
        logger.exception("Free Teacher memory persistence to Neon failed.")
        return False


async def _persist_snapshot_to_neon(snapshot: dict[str, Any]) -> bool:
    try:
        await _ensure_memory_clients()
        return await asyncio.to_thread(_persist_snapshot_to_neon_sync, snapshot)
    except Exception:
        logger.exception("Free Teacher memory persistence task failed.")
        return False


async def persist_snapshot(snapshot: dict[str, Any]) -> None:
    """Write a transcript snapshot to Redis first and Neon second."""
    async with _memory_persist_lock:
        redis_saved = await _save_turns_to_redis(snapshot)
        neon_saved = await _persist_snapshot_to_neon(snapshot)
    logger.info(
        "[MEMORY SAVE] user=%s turns=%s redis=%s neon=%s",
        _clean_text(snapshot.get("user_id"), "-"),
        len(snapshot.get("turns", [])) if isinstance(snapshot.get("turns"), list) else 0,
        redis_saved,
        neon_saved,
    )
    if not neon_saved:
        logger.error("[MEMORY SAVE] Neon write failed; inspect the Neon error above.")


async def load_memory(
    user_id: str, source: str, target: str, topic: str
) -> list[dict[str, str]]:
    cached = await _load_turns_from_redis(user_id, source, target, topic)
    if cached:
        logger.info("[MEMORY] Loaded %s turns from Redis cache.", len(cached))
        return cached
    turns = await _hydrate_turns_from_neon(user_id, source, target, topic)
    logger.info("[MEMORY] Loaded %s turns from Neon.", len(turns))
    if turns:
        await _save_turns_to_redis(
            {
                "user_id": user_id,
                "source_language": source,
                "target_language": target,
                "topic": topic,
                "turns": turns,
            }
        )
    return turns
