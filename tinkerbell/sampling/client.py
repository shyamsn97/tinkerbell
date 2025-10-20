import httpx


class SamplingClient:
    def __init__(
        self,
        server_url: str,
        timeout: int = 600,
    ):
        self.timeout = timeout
        self.server_url = server_url

    def health(self) -> dict:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(f"{self.server_url}/health")
            return response.json()

    async def health_async(self) -> dict:
        with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.server_url}/health")
            return response.json()

    def generate(self, prompts: list[str], sampling_params: dict, generate_kwargs: dict) -> dict:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.server_url}/batch_generate",
                json={
                    "prompts": prompts,
                    "sampling_params": sampling_params,
                    "generate_kwargs": generate_kwargs,
                },
            )
            return response.json()

    async def generate_async(self, prompts: list[str], sampling_params: dict, generate_kwargs: dict) -> dict:
        with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.server_url}/batch_generate",
                json={
                    "prompts": prompts,
                    "sampling_params": sampling_params,
                    "generate_kwargs": generate_kwargs,
                },
            )
            return response.json()