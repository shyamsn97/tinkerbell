import abc
from typing import Any


class TrainingServer(abc.ABC):

    @abc.abstractmethod
    def forward(self, data: list[Any], loss_fn) -> Any:
        """Forward pass through the model and compute the loss.

        Args:
            data (list[Any]): A list of data points to be processed by the model.
            loss_fn (_type_): A loss function to be used to compute the loss.

        Returns:
            Any: A list of loss values.
        """

    @abc.abstractmethod
    def forward_backward(self, data: list[Any], loss_fn) -> Any:
        """Forward and backward pass through the model and compute the loss, as well as do a backward pass.

        Args:
            data (list[Any]): A list of data points to be processed by the model.
            loss_fn (_type_): A loss function to be used to compute the loss.

        Returns:
            Any: A list of loss values.
        """

    @abc.abstractmethod
    def optim_step(self, optimizer_params) -> Any:
        """Perform an optimization step.

        Args:
            optimizer_params (_type_): A dictionary of optimizer parameters.

        Returns:
            Any: A dictionary of optimizer parameters.
        """

    @abc.abstractmethod
    def save_state(self) -> Any:
        """Save the state of the model.

        Returns:
            Any: A dictionary of the model state.
        """

    @abc.abstractmethod
    def load_state(self, state: Any) -> Any:
        """Load the state of the model.

        Args:
            state (_type_): A dictionary of the model state.

        Returns:
            Any: A dictionary of the model state.
        """

    @abc.abstractmethod
    def deploy(self, config) -> Any:
        """Deploy the model.

        Args:
            config (_type_): A dictionary of the model / server configuration.

        Returns:
            Any: A dictionary of the model / server configuration.
        """
