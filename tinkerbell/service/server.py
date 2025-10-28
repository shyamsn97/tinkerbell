from fastapi import FastAPI
from tinkerbell.service.models import HealthResponse, CreateTrainingServerRequest, CreateTrainingServerResponse
from tinkerbell.server.training.server import TrainingServer

class Service:
    def __init__(self, server_url: str, ):
        self.server_url = server_url
        self.app = FastAPI(title=f"Tinkerbell Service")
        self.setup_routes()
        self.training_servers = {}

    def setup_routes(self):

        @self.app.get("/health")
        async def health() -> HealthResponse:
            return HealthResponse(
                status="healthy",
                server_url=self.server_url,
            )

        @self.app.post("/create_training_server")
        async def create_training_server(
            request: CreateTrainingServerRequest
        ) -> CreateTrainingServerResponse:
            return await self.create_training_server(request)

    async def create_training_server(self, request: CreateTrainingServerRequest) -> CreateTrainingServerResponse:
        """Create a training server for the given model name."""
        name = request.model_name
        training_server = TrainingServer(name=name)
        training_server_url = training_server.run()
        self.training_servers[name] = training_server_url
        return CreateTrainingServerResponse(
            success=True,
            message=f"Training server started for {name}",
            training_server_url=training_server_url,
        )
