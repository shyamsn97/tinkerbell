from typing import Any, Dict

import ray


@ray.remote
class GlobalStore:
    def __init__(self):
        self.request_queue: Dict[str, list[Any]] = {}
        self.results: Dict[str, Any] = {}

    async def get_keys(self) -> list[str]:
        return list(self.results.keys())

    async def get_request_queue(self) -> Dict[str, list[Any]]:
        return self.request_queue

    async def get_requests(self, queue_key: str) -> list[Any]:
        return self.request_queue.get(queue_key, [])

    async def clear_request_queue(self, queue_key: str):
        if queue_key in self.request_queue:
            self.request_queue[queue_key] = []

    async def add_request_to_queue(self, request: Any):
        adapter_name = getattr(request, "adapter_name", None) or ""
        queue_key = f"{request.model_name}:{adapter_name}"
        if queue_key not in self.request_queue:
            self.request_queue[queue_key] = []
        self.request_queue[queue_key].append(request)

    async def get_results(self) -> Dict[str, Any]:
        return self.results

    async def get_result(self, request_id: str) -> Any:
        return self.results.get(request_id)

    async def pop_result(self, request_id: str) -> Any:
        """Get and delete result in one call - reduces Ray overhead."""
        return self.results.pop(request_id, None)

    async def delete_result(self, request_id: str) -> bool:
        if request_id in self.results:
            del self.results[request_id]
            return True
        return False

    async def set_result(self, request_id: str, result: Any):
        self.results[request_id] = result
