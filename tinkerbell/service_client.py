# from typing import Any

# from pydantic import BaseModel

# # from tinkerbell.client import SamplingClient, TrainingClient


# class CreateTrainingClientConfig(BaseModel):
#     model_name: str | None = None
#     model_config: dict | None = None
#     lora_config: dict | None = None
#     model_checkpoint_path: str | None = None
#     deploy_config: dict | None = None


# class CreateSamplingClientConfig(BaseModel):
#     model_name: str | None = None
#     model_config: dict | None = None
#     lora_config: dict | None = None
#     model_checkpoint_path: str | None = None
#     deploy_config: dict | None = None


# def create_training_client(
#     config: CreateTrainingClientConfig | None = None, redeploy: bool = False, **kwargs
# ) -> Any: ...


# def create_sampling_client(
#     config: CreateSamplingClientConfig | None = None, redeploy: bool = False, **kwargs
# ) -> Any: ...


# class ServiceClient:
#     def __init__(self):
#         self.clients = {}

#     def create_training_client(
#         self,
#         config: CreateTrainingClientConfig | None = None,
#         redeploy: bool = False,
#         **kwargs,
#     ) -> Any:
#         """Create a client for the service.

#         Args:
#             **kwargs: Keyword arguments to create the client.

#         Returns:
#             Any: A client for the service.
#         """
#         if config is None:
#             config = CreateTrainingClientConfig()
#         config_dict = config.model_dump()
#         merged_config_dict = {**config_dict, **kwargs}
#         config = CreateTrainingClientConfig(**merged_config_dict)

#         self.clients[config.model_name] = create_training_client(
#             config, redeploy=redeploy
#         )

#     def create_sampling_client(
#         self,
#         config: CreateSamplingClientConfig | None = None,
#         redeploy: bool = False,
#         **kwargs,
#     ) -> Any:
#         """Create a client for the service.

#         Args:
#             **kwargs: Keyword arguments to create the client.

#         Returns:
#             Any: A client for the service.
#         """
#         if config is None:
#             config = CreateSamplingClientConfig()
#         config_dict = config.model_dump()
#         merged_config_dict = {**config_dict, **kwargs}
#         config = CreateSamplingClientConfig(**merged_config_dict)

#         self.clients[config.model_name] = create_sampling_client(
#             config, redeploy=redeploy
#         )
