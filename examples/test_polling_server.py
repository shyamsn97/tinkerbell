import asyncio
import json
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Optional
from redis.asyncio import Redis

REDIS_URL = "redis://localhost:6379"
TICK_INTERVAL = 10.0  # seconds per cycle
JOB_QUEUE_KEY = "job_queue"
INPUT_KEY_PREFIX = "job_input:"
RESULT_KEY_PREFIX = "job_result:"

# (You can adjust these)
REQUEST_TIMEOUT = 12.0  # how long we’ll wait in the request handler

app = FastAPI()
redis: Redis

async def expensive_job(inputs: dict) -> dict:
    # Simulate something expensive
    await asyncio.sleep(1)
    # Just echo back
    return {"received": inputs, "processed_at": asyncio.get_event_loop().time()}

async def tick_loop():
    """Runs every TICK_INTERVAL, empties the queue, processes jobs."""
    while True:
        await asyncio.sleep(TICK_INTERVAL)
        # Pop all jobs in the queue
        job_ids = []
        while True:
            raw = await redis.lpop(JOB_QUEUE_KEY)
            if raw is None:
                break
            job_ids.append(raw.decode() if isinstance(raw, bytes) else raw)
        if not job_ids:
            continue

        # Process each job
        for job_id in job_ids:
            input_key = INPUT_KEY_PREFIX + job_id
            raw = await redis.get(input_key)
            if raw is None:
                # missing input, skip
                continue
            try:
                data = json.loads(raw)
            except Exception as e:
                # JSON parse error
                res = {"error": f"invalid input json: {e}"}
                await redis.set(RESULT_KEY_PREFIX + job_id, json.dumps(res))
                continue

            try:
                result = await expensive_job(data)
                await redis.set(RESULT_KEY_PREFIX + job_id, json.dumps({"result": result}))
            except Exception as e:
                await redis.set(RESULT_KEY_PREFIX + job_id, json.dumps({"error": str(e)}))

            # Clean up input, optionally expire result later
            await redis.delete(input_key)
            # set an expiry on the result so memory is cleaned
            await redis.expire(RESULT_KEY_PREFIX + job_id, ex=60 * 5)

@app.on_event("startup")
async def startup():
    global redis
    redis = Redis.from_url(REDIS_URL, decode_responses=False)
    # Kick off ticker loop
    asyncio.create_task(tick_loop())

@app.on_event("shutdown")
async def shutdown():
    await redis.close()

class JobRequest(BaseModel):
    payload: dict

class JobResponse(BaseModel):
    result: Any

@app.post("/submit_and_wait", response_model=JobResponse)
async def submit_and_wait(req: JobRequest):
    # Create job id, store input, enqueue
    job_id = str(uuid.uuid4())
    input_key = INPUT_KEY_PREFIX + job_id
    # store input JSON
    await redis.set(input_key, json.dumps(req.payload))
    await redis.rpush(JOB_QUEUE_KEY, job_id)

    # Now wait for the result up to timeout
    result_key = RESULT_KEY_PREFIX + job_id
    try:
        # Polling loop: you can also use Redis PUB/SUB or blocking waits.
        waited = 0.0
        interval = 0.1  # poll every 100ms
        while waited < REQUEST_TIMEOUT:
            raw = await redis.get(result_key)
            if raw is not None:
                # parse and return
                obj = json.loads(raw)
                if "error" in obj:
                    # you might raise HTTPException or return error structure
                    raise HTTPException(status_code=500, detail=f"Job error: {obj['error']}")
                return JobResponse(result=obj["result"])
            await asyncio.sleep(interval)
            waited += interval
    except asyncio.CancelledError:
        # Request was cancelled (client disconnected)
        raise
    # Timeout case
    raise HTTPException(status_code=202, detail="Job not finished within timeout")

