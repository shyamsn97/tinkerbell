import abc
from typing import Any

import torch
import torch.nn as nn

from tinkerbell.types.data import TensorData
from tinkerbell.types.datum import Datum
from tinkerbell.types.lora_config import LoraConfig


class TrainingModel(metaclass=abc.ABCMeta):

    def __init__(
        self,
        rank: int,
        world_size: int,
        model_id: str,
        model_kwargs: dict[str, Any],
        parallelize_plan: dict[str, str],
        lora_config: LoraConfig | None = None,
        initialize_random_weights: bool = False,
    ):
        self.rank = rank
        self.world_size = world_size
        self.model_id = model_id
        self.model_kwargs = model_kwargs
        self.parallelize_plan = parallelize_plan
        self.lora_config = lora_config
        self.initialize_random_weights = initialize_random_weights
        self.should_merge_lora = False
        self.setup()

    def setup(self) -> None:
        """
        Setup the model.
        """
        self.model = self.create_model(
            model_id=self.model_id,
            model_kwargs=self.model_kwargs,
        )
        self.model = self.setup_lora(
            model=self.model,
            lora_config=self.lora_config,
        )
        self.model = self.parallelize(
            model=self.model,
            parallelize_plan=self.parallelize_plan,
        )

    @abc.abstractmethod
    def create_model(self, model_id: str, model_kwargs: dict[str, Any]) -> nn.Module:
        """
        Create a model from a model ID and model kwargs.

        Args:
            model_id: ID of the model to create
            model_kwargs: Kwargs to pass to the model constructor
        """

    @abc.abstractmethod
    def setup_lora(self, lora_config: LoraConfig) -> nn.Module:
        pass

    @abc.abstractmethod
    def parallelize(
        self,
        model: nn.Module,
        parallelize_plan: dict[str, str],
    ) -> nn.Module:
        pass

    @abc.abstractmethod
    def stack_inputs(
        self,
        data: list[Datum],
    ) -> tuple[dict[str, TensorData], dict[str, TensorData]]:
        """
        Stack a list of Datum objects into batched tensors.

        Args:
            data: List of Datum objects to stack

        Returns:
            Tuple of (model_inputs, loss_fn_inputs) where each is a dict of stacked tensors
        """

    @abc.abstractmethod
    def forward(
        self,
        model_inputs: dict[str, TensorData],
        loss_fn_inputs: dict[str, TensorData],
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            model_inputs: Dictionary of model inputs
            loss_fn_inputs: Dictionary of loss function inputs

        Returns:
            Output tensor
        """

    @abc.abstractmethod
    def save_model(self, path: str) -> None:
        """
        Save the model to a path.

        Args:
            path: Path to save the model
        """

    def train(self) -> None:
        """
        Train the model.
        """
        self.model.train()

    def eval(self) -> None:
        """
        Evaluate the model.
        """
        self.model.eval()

    def __call__(
        self,
        model_inputs: dict[str, TensorData],
        loss_fn_inputs: dict[str, TensorData],
    ) -> Any:
        return self.forward(model_inputs, loss_fn_inputs)
