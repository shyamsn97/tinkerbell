import abc
from fastapi import FastAPI
from tinkerbell.server.training.models import SetupTrainRequest, SetupTrainResponse, ForwardRequest, ForwardResponse, ForwardBackwardRequest, ForwardBackwardResponse, HealthResponse
import uvicorn

class TrainingServer(metaclass=abc.ABCMeta):

    def __init__(
        self,
        name: str,
        host: str = "0.0.0.0",
        port: int = 8000,
    ):
        self.name = name
        self.host = host
        self.port = port
        self.app = FastAPI()
        self.setup_routes()
        self.health_status = "not_initialized"

    def setup_routes(self):

        @self.app.get("/health")
        async def health() -> HealthResponse:
            return HealthResponse(
                status=self.health_status,
                name=self.name,
            )

        @self.app.post("/setup")
        async def setup(request: SetupTrainRequest) -> SetupTrainResponse:
            response = await self.setup(request)
            self.health_status = "initialized"
            return response

        @self.app.post("/forward")
        async def forward(request: ForwardRequest):
            response = await self.forward(request)
            return response

        @self.app.post("/forward_backward")
        async def forward_backward(request: ForwardBackwardRequest):
            response = await self.forward_backward(request)
            return response

    @abc.abstractmethod
    async def setup(self, request: SetupTrainRequest) -> SetupTrainResponse:
        """Setup the training environment.

        Args:
            request (SetupTrainRequest): A request containing the training environment configuration.

        Returns:
            SetupTrainResponse: A response containing the training environment configuration.
        """

    @abc.abstractmethod
    async def forward(self, request: ForwardRequest) -> ForwardResponse:
        """Forward pass through the model and compute the loss.

        Args:
            request (ForwardRequest): A request containing the data and loss function.

        Returns:
            ForwardResponse: A response containing the output and loss.
        """

    @abc.abstractmethod
    async def forward_backward(self, request: ForwardBackwardRequest) -> ForwardBackwardResponse:
        """Forward and backward pass through the model and compute the loss, as well as do a backward pass.

        Args:
            request (ForwardBackwardRequest): A request containing the data and loss function.

        Returns:
            ForwardBackwardResponse: A response containing the output and loss.
        """

    def run(self) -> str:
        """Run the FastAPI server"""
        uvicorn.run(self.app, host=self.host, port=self.port)
        self.server_url = f"https://{self.host}:{self.port}"
        return self.server_url
