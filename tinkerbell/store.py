from typing import Any, Dict

import ray

from tinkerbell.types.requests import ActorRequest


@ray.remote
class GlobalStore:
    def __init__(self):
        self.request_queue: Dict[str, list[ActorRequest]] = {}
        self.results: Dict[str, Any] = {}

    async def get_keys(self) -> list[str]:
        return list(self.results.keys())

    async def get_request_queue(self) -> Dict[str, list[ActorRequest]]:
        return self.request_queue

    async def get_requests(self, queue_key: str) -> list[ActorRequest]:
        return self.request_queue.get(queue_key, [])

    async def clear_request_queue(self, queue_key: str):
        if queue_key in self.request_queue:
            self.request_queue[queue_key] = []

    async def add_request_to_queue(self, request: ActorRequest):
        adapter_name = getattr(request, "adapter_name", None) or ""
        queue_key = f"{request.model_id}:{adapter_name}"
        if queue_key not in self.request_queue:
            self.request_queue[queue_key] = []
        self.request_queue[queue_key].append(request)

    async def get_results(self) -> Dict[str, Any]:
        return self.results

    async def get_result(self, request_id: str) -> Any:
        return self.results.get(request_id)

    async def delete_result(self, request_id: str) -> bool:
        if request_id in self.results:
            del self.results[request_id]
            return True
        return False

    async def set_result(self, request_id: str, result: Any):
        self.results[request_id] = result
