import logging
import time
from typing import Any, Dict

import ray
from pydantic import BaseModel

from tinkerbell.types.requests import ActorRequest

logger = logging.getLogger(__name__)


class GlobalStoreRequest(BaseModel):
    start_time: float | None = None
    request: ActorRequest | None = None

    def __post_init__(self):
        if self.start_time is None:
            self.start_time = time.time()


@ray.remote
class GlobalStore:
    def __init__(self):
        self.request_queue: Dict[str, list[ActorRequest]] = {}
        self.results: Dict[str, Any] = {}

    async def get_keys(self) -> list[str]:
        return list(self.results.keys())

    async def get_request_queue(self) -> Dict[str, list[ActorRequest]]:
        return self.request_queue

    async def get_requests(self, model_id: str) -> list[ActorRequest]:
        if model_id not in self.request_queue:
            return []
        return self.request_queue[model_id]

    async def clear_request_queue(self, model_id: str):
        if model_id not in self.request_queue:
            return
        self.request_queue[model_id] = []

    async def add_request_to_queue(self, request: ActorRequest):
        if request.model_id not in self.request_queue:
            self.request_queue[request.model_id] = []
        self.request_queue[request.model_id].append(request)

    async def get_results(self) -> Dict[str, Any]:
        return self.results

    async def get_result(self, request_id: str) -> Any:
        has_result = request_id in self.results
        logger.debug(
            f"[GlobalStore.get_result] request_id: {request_id}, has_result: {has_result}"
        )
        if not has_result:
            return None
        return self.results.get(request_id)

    async def delete_result(self, request_id: str) -> bool:
        """Delete a result from the store. Returns True if deleted, False if not found."""
        if request_id in self.results:
            del self.results[request_id]
            logger.debug(
                f"[GlobalStore.delete_result] Deleted result for request_id: {request_id}"
            )
            return True
        return False

    async def set_result(self, request_id: str, result: Any):
        logger.info(
            f"[GlobalStore.set_result] Storing result for request_id: {request_id}"
        )
        logger.info(
            f"[GlobalStore.set_result] Result keys: {result.keys() if isinstance(result, dict) else 'not a dict'}"
        )
        self.results[request_id] = result
        logger.info(
            f"[GlobalStore.set_result] Successfully stored. Total results: {len(self.results)}"
        )
