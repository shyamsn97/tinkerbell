from typing import Any

from tinkerbell.server.training.server import TrainingServer


class LocalTrainingServer(TrainingServer):

    def __init__(self, model_config):
        self.model_config = model_config

    def deploy(self, config):
        pass

    def forward(self, data: list[Any], loss_fn) -> Any:
        pass

    def forward_backward(self, data: list[Any], loss_fn) -> Any:
        pass

    def optim_step(self, optimizer_params) -> Any:
        pass
