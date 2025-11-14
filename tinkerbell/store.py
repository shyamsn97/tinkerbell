import time
from typing import Any, Dict

import ray
from pydantic import BaseModel

from tinkerbell.types.requests import ActorRequest


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

    def get_request_queue(self) -> Dict[str, list[ActorRequest]]:
        return self.request_queue

    def get_requests(self, model_id: str) -> list[ActorRequest]:
        if model_id not in self.request_queue:
            return []
        return self.request_queue[model_id]

    def clear_request_queue(self, model_id: str):
        if model_id not in self.request_queue:
            return
        self.request_queue[model_id] = []

    def add_request_to_queue(self, request: ActorRequest):
        if request.model_id not in self.request_queue:
            self.request_queue[request.model_id] = []
        self.request_queue[request.model_id].append(request)

    def get_results(self) -> Dict[str, Any]:
        return self.results

    def get_result(self, request_id: str) -> Any:
        if request_id not in self.results:
            return None
        return self.results[request_id]

    def set_result(self, request_id: str, result: Any):
        self.results[request_id] = result
