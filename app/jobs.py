"""Estado de los jobs y cola de trabajo, ambos en Redis.

Claves:
  agentops:job:<id>          HASH  estado del job (status, query, result, error, timings, ...)
  agentops:job:<id>:events   LIST  historial de transiciones (auditoría)
  agentops:jobs              ZSET  índice de jobs por fecha de creación
  agentops:queue             LIST  cola de mensajes para los workers ({"job_id", "action": start|resume})

Ciclo de vida:
  PENDING -> RUNNING -> DONE
                     -> FAILED              (excepción o timeout: el cliente nunca queda en polling infinito)
                     -> AWAITING_APPROVAL -> PENDING (POST /approve) -> RUNNING -> ...

Las transiciones son atómicas (script Lua "compare-and-set"): dos aprobaciones simultáneas o dos workers
tomando el mismo job no pueden pisarse.
"""

from __future__ import annotations

import json
import time
import uuid
from enum import StrEnum
from typing import Any

from app.config import settings
from app.redis_client import get_redis


class JobStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    DONE = "DONE"
    FAILED = "FAILED"


TERMINAL = {JobStatus.DONE, JobStatus.FAILED}
JSON_FIELDS = {"approval_request", "approval", "decisions", "contributions", "trace", "error_detail"}

# KEYS[1]=hash del job · ARGV[1]=estados esperados separados por coma ("*" = cualquiera)
# ARGV[2]=nuevo estado · ARGV[3..]=pares campo/valor adicionales
_TRANSITION_LUA = """
local cur = redis.call('HGET', KEYS[1], 'status')
if not cur then return -1 end
if ARGV[1] ~= '*' then
  local ok = false
  for s in string.gmatch(ARGV[1], '([^,]+)') do
    if s == cur then ok = true end
  end
  if not ok then return 0 end
end
redis.call('HSET', KEYS[1], 'status', ARGV[2])
for i = 3, #ARGV, 2 do
  redis.call('HSET', KEYS[1], ARGV[i], ARGV[i + 1])
end
return 1
"""


def now() -> float:
    return round(time.time(), 3)


def _job_key(job_id: str) -> str:
    return settings.key("job", job_id)


def _encode(fields: dict[str, Any]) -> dict[str, str]:
    out = {}
    for k, v in fields.items():
        if v is None:
            continue
        out[k] = json.dumps(v, ensure_ascii=False, default=str) if k in JSON_FIELDS else str(v)
    return out


def _decode(raw: dict[str, str]) -> dict[str, Any]:
    job: dict[str, Any] = {}
    for k, v in raw.items():
        if k in JSON_FIELDS:
            job[k] = json.loads(v)
        elif k in {"created_at", "updated_at", "started_at", "run_started_at", "finished_at", "duration_seconds",
                   "processing_seconds"}:
            job[k] = float(v)
        elif k in {"max_steps", "runs"}:
            job[k] = int(v)
        else:
            job[k] = v
    return job


async def _log_event(job_id: str, status: str, detail: str = "") -> None:
    redis = get_redis()
    key = f"{_job_key(job_id)}:events"
    await redis.rpush(key, json.dumps({"ts": now(), "status": status, "detail": detail}, ensure_ascii=False))
    await redis.expire(key, settings.job_ttl_seconds)


async def create_job(query: str, max_steps: int) -> dict[str, Any]:
    redis = get_redis()
    job_id = uuid.uuid4().hex
    ts = now()
    fields = {
        "id": job_id,
        "status": JobStatus.PENDING,
        "query": query,
        "max_steps": max_steps,
        "created_at": ts,
        "updated_at": ts,
        "runs": 0,
    }
    async with redis.pipeline(transaction=True) as pipe:
        pipe.hset(_job_key(job_id), mapping=_encode(fields))
        pipe.expire(_job_key(job_id), settings.job_ttl_seconds)
        pipe.zadd(settings.key("jobs"), {job_id: ts})
        await pipe.execute()
    await _log_event(job_id, JobStatus.PENDING, "creado")
    await enqueue(job_id, "start")
    return fields


async def get_job(job_id: str) -> dict[str, Any] | None:
    raw = await get_redis().hgetall(_job_key(job_id))
    return _decode(raw) if raw else None


async def get_events(job_id: str) -> list[dict[str, Any]]:
    return [json.loads(e) for e in await get_redis().lrange(f"{_job_key(job_id)}:events", 0, -1)]


async def list_jobs(limit: int = 20) -> list[dict[str, Any]]:
    redis = get_redis()
    ids = await redis.zrevrange(settings.key("jobs"), 0, limit - 1)
    jobs = []
    for job_id in ids:
        if job := await get_job(job_id):
            jobs.append(job)
        else:  # expiró el hash: limpiar el índice
            await redis.zrem(settings.key("jobs"), job_id)
    return jobs


async def transition(
    job_id: str, new_status: JobStatus, expected: set[JobStatus] | None = None, detail: str = "", **fields: Any
) -> bool:
    """Cambia el estado solo si el actual está en `expected`. Devuelve True si se aplicó."""
    extra = _encode({**fields, "updated_at": now()})
    args = [",".join(expected) if expected else "*", str(new_status)]
    for k, v in extra.items():
        args += [k, v]
    result = await get_redis().eval(_TRANSITION_LUA, 1, _job_key(job_id), *args)
    if result == 1:
        await _log_event(job_id, new_status, detail)
        return True
    return False


async def update(job_id: str, **fields: Any) -> None:
    await get_redis().hset(_job_key(job_id), mapping=_encode({**fields, "updated_at": now()}))


async def incr_runs(job_id: str) -> int:
    return int(await get_redis().hincrby(_job_key(job_id), "runs", 1))


# ------------------------------------------------------------------------------------------- cola
async def enqueue(job_id: str, action: str, payload: dict[str, Any] | None = None) -> None:
    msg = {"job_id": job_id, "action": action, "payload": payload or {}}
    await get_redis().lpush(settings.key("queue"), json.dumps(msg, ensure_ascii=False))


async def dequeue(timeout: int = 1) -> dict[str, Any] | None:
    item = await get_redis().brpop([settings.key("queue")], timeout=timeout)
    return json.loads(item[1]) if item else None


async def queue_length() -> int:
    return int(await get_redis().llen(settings.key("queue")))
