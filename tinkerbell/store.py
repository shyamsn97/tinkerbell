from typing import Any, Dict, List

import ray


@ray.remote
class GlobalStore:
    def __init__(self):
        self.request_queue: Dict[str, list[Any]] = {}
        self.results: Dict[str, Any] = {}
        # Actor registries for cross-replica access
        self.sampling_actors: Dict[str, Any] = {}  # model_name -> actor handle
        self.training_actors: Dict[str, Dict[str, Any]] = (
            {}
        )  # model_name -> {workers, base_model, adapters}

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

    # Sampling actor registry methods
    async def set_sampling_actor(self, model_name: str, actor: Any):
        """Store a sampling actor handle for cross-replica access."""
        self.sampling_actors[model_name] = actor

    async def get_sampling_actor(self, model_name: str) -> Any:
        """Get a sampling actor handle by model name."""
        return self.sampling_actors.get(model_name)

    async def delete_sampling_actor(self, model_name: str) -> bool:
        """Remove a sampling actor from the registry."""
        if model_name in self.sampling_actors:
            del self.sampling_actors[model_name]
            return True
        return False

    async def get_sampling_actor_names(self) -> list[str]:
        """Get all registered sampling actor model names."""
        return list(self.sampling_actors.keys())

    # Training actor registry methods
    async def set_training_actors(
        self,
        model_name: str,
        workers: List[Any],
        base_model: str,
        adapters: Dict[str, str] = None,
    ):
        """Store training actor workers for cross-replica access.

        Args:
            model_name: Model identifier
            workers: List of Ray actor handles (TrainingActor workers)
            base_model: Base model name/path
            adapters: Dict of adapter_name -> checkpoint_path
        """
        self.training_actors[model_name] = {
            "workers": workers,
            "base_model": base_model,
            "adapters": adapters or {},
        }

    async def get_training_actors(self, model_name: str) -> Dict[str, Any]:
        """Get training actor info by model name.

        Returns dict with 'workers', 'base_model', 'adapters' or None if not found.
        """
        return self.training_actors.get(model_name)

    async def update_training_actor_adapters(
        self, model_name: str, adapter_name: str, checkpoint_path: str
    ):
        """Update the adapters dict for a training actor group."""
        if model_name in self.training_actors:
            self.training_actors[model_name]["adapters"][adapter_name] = checkpoint_path

    async def delete_training_actors(self, model_name: str) -> bool:
        """Remove training actors from the registry."""
        if model_name in self.training_actors:
            del self.training_actors[model_name]
            return True
        return False

    async def get_training_actor_names(self) -> list[str]:
        """Get all registered training actor model names."""
        return list(self.training_actors.keys())

    # Cleanup methods
    async def clear_all_actors(self):
        """Clear all actor registries (for cleanup on shutdown)."""
        self.sampling_actors.clear()
        self.training_actors.clear()

    async def get_all_registered_actors(self) -> Dict[str, Any]:
        """Get summary of all registered actors."""
        return {
            "sampling_actors": list(self.sampling_actors.keys()),
            "training_actors": list(self.training_actors.keys()),
        }
