from __future__ import annotations

import abc

from fastapi import FastAPI

from tinkerbell.service.models import (
    ActorStatusRequest,
    ActorStatusResponse,
    CreateTrainingActorsRequest,
    CreateTrainingActorsResponse,
    ForwardBackwardRequest,
    ForwardBackwardResponse,
    ForwardRequest,
    ForwardResponse,
)

APP = FastAPI()


class ServiceBackend(metaclass=abc.ABCMeta):
    def __init__(self, server_url: str | None = None):
        self.server_url = server_url
        self.health_status = "healthy"
        self.name = "TinkerbellService"

    @abc.abstractmethod
    async def create_training_actors(
        self, request: CreateTrainingActorsRequest
    ) -> CreateTrainingActorsResponse:
        """Create training actors for the given model name.

        Args:
            request (CreateTrainingActorsRequest): A request containing the training environment configuration.

        Returns:
            CreateTrainingActorsResponse: A response containing the training environment configuration.
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
    async def forward_backward(
        self, request: ForwardBackwardRequest
    ) -> ForwardBackwardResponse:
        """Forward and backward pass through the model and compute the loss, as well as do a backward pass.

        Args:
            request (ForwardBackwardRequest): A request containing the data and loss function.

        Returns:
            ForwardBackwardResponse: A response containing the output and loss.
        """

    @abc.abstractmethod
    async def get_actor_status(
        self, request: ActorStatusRequest
    ) -> ActorStatusResponse:
        """Get the status of the training actors.

        Args:
            request (ActorStatusRequest): A request containing the model name.

        Returns:
            ActorStatusResponse: A response containing the status of the training actors.
        """
