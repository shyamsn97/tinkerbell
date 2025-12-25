from typing import Any, Dict, List, Optional, Union

from pydantic import Field

from ._models import BaseModel, LossFnType, StrictBase
from .data import MultimodalDataInputFormat, TensorData
from .datum import Datum
from .lora_config import LoraConfig
from .optimizer import DEFAULT_SCHEDULER_PARAMS


class CreateTrainingActorsRequest(StrictBase):
    world_size: int
    base_model: str  # HuggingFace model path (e.g., "Qwen/Qwen3-0.6B")
    model_name: Optional[str] = None  # Actor group name; defaults to cleaned base_model
    adapter_name: Optional[str] = None  # For multi-LoRA: names the adapter
    model_kwargs: dict[str, Any] = Field(default_factory=lambda: {})
    parallelize_plan: dict[str, str] = Field(default_factory=lambda: {})
    scheduler_params: dict[str, Any] = Field(
        default_factory=lambda: DEFAULT_SCHEDULER_PARAMS
    )
    lora_config: Optional[LoraConfig | dict[str, Any]] = Field(default=None)
    ray_worker_options: dict[str, Any] = Field(default_factory=lambda: {})
    wait_until_ready: bool = False
    initialize_random_weights: bool = False


class SaveCheckpointRequest(StrictBase):
    model_name: str  # Actor group name for routing
    checkpoint_path: str
    adapter_name: Optional[str] = None  # Which LoRA adapter to save


class PushToHubRequest(StrictBase):
    model_name: str  # Actor group name for routing
    repo_id: str  # Hugging Face Hub repository ID (e.g., "username/model-name")
    adapter_name: Optional[str] = None  # Which LoRA adapter to push
    token: Optional[str] = None  # Hugging Face token for authentication
    private: bool = False  # Whether to create a private repository
    commit_message: Optional[str] = None  # Commit message
    push_kwargs: dict[str, Any] = Field(
        default_factory=lambda: {}
    )  # Additional push_to_hub kwargs


class ForwardRequest(StrictBase):
    model_name: str  # Actor group name for routing
    request_id: Optional[str] = None
    data: list[Datum] = Field(default_factory=lambda: [])
    forward_kwargs: dict[str, Any] = Field(default_factory=lambda: {})


class ForwardBackwardRequest(StrictBase):
    model_name: str  # Actor group name for routing
    request_id: Optional[str] = None
    adapter_name: Optional[str] = None  # Which LoRA adapter to use
    data: list[Datum] = Field(default_factory=lambda: [])
    forward_kwargs: dict[str, Any] = Field(default_factory=lambda: {})
    loss_fn: LossFnType = "cross_entropy"
    return_logprobs: bool = False
    zero_grad: bool = (
        True  # Zero gradients before forward/backward (set False for gradient accumulation)
    )
    optimizer_params: Optional[dict[str, Any]] = (
        None  # If provided, run optim_step after backward (combines into single round trip)
    )
    immediate: bool = (
        False  # If True, process the batch queue immediately instead of waiting for clock cycle
    )


class ActorStatusRequest(StrictBase):
    model_name: str  # Actor group name for routing


class CreateSamplingActorRequest(StrictBase):
    base_model: str  # HuggingFace model path (e.g., "Qwen/Qwen3-0.6B")
    model_name: Optional[str] = None  # Actor group name; defaults to cleaned base_model
    tp_size: int
    engine_kwargs: dict[str, Any] = Field(default_factory=lambda: {})


class SampleRequest(StrictBase):
    model_name: Optional[str] = None  # Actor group name for routing
    # The input prompt. It can be a single prompt or a batch of prompts.
    text: Optional[Union[List[str], str]] = None
    # The token ids for text; one can specify either text or input_ids
    input_ids: Optional[Union[List[List[int]], List[int], TensorData]] = None
    # The embeddings for input_ids; one can specify either text or input_ids or input_embeds.
    input_embeds: Optional[
        Union[List[List[List[float]]], List[List[float]], TensorData]
    ] = None
    # The image input. It can be an image instance, file name, URL, or base64 encoded string.
    # Can be formatted as:
    # - Single image for a single request
    # - List of images (one per request in a batch)
    # - List of lists of images (multiple images per request)
    # See also python/sglang/srt/utils.py:load_image for more details.
    image_data: Optional[MultimodalDataInputFormat] = None
    # The video input. Like image data, it can be a file name, a url, or base64 encoded string.
    video_data: Optional[MultimodalDataInputFormat] = None
    # The audio input. Like image data, it can be a file name, a url, or base64 encoded string.
    audio_data: Optional[MultimodalDataInputFormat] = None
    # The sampling_params. See descriptions below.
    sampling_params: Optional[Union[List[Dict], Dict]] = None
    # Whether to return logprobs.
    return_logprob: Optional[Union[List[bool], bool]] = True
    # If return logprobs, the start location in the prompt for returning logprobs.
    # By default, this value is "-1", which means it will only return logprobs for output tokens.
    logprob_start_len: Optional[Union[List[int], int]] = None
    # If return logprobs, the number of top logprobs to return at each position.
    # Default to 1 to ensure output_top_logprobs is returned (needed for make_logprobs_tensor)
    top_logprobs_num: Optional[Union[List[int], int]] = 100
    # If return logprobs, the token ids to return logprob for.
    token_ids_logprob: Optional[Union[List[List[int]], List[int]]] = None
    # Whether to detokenize tokens in text in the returned logprobs.
    return_text_in_logprobs: bool = False
    # Whether to stream output.
    stream: bool = False
    # Whether to log metrics for this request (e.g. health_sample calls do not log metrics)
    log_metrics: bool = True
    # Whether to return hidden states
    return_hidden_states: Union[List[bool], bool] = False

    # The modalities of the image data [image, multi-images, video]
    modalities: Optional[List[str]] = None
    # Session info for continual prompting
    session_params: Optional[Union[List[Dict], Dict]] = None

    # The path to the LoRA adaptors
    lora_path: Optional[Union[List[Optional[str]], Optional[str]]] = None
    # The uid of LoRA adaptors, should be initialized by tokenizer manager
    lora_id: Optional[Union[List[Optional[str]], Optional[str]]] = None

    # Custom logit processor for advanced sampling control. Must be a serialized instance
    # of `CustomLogitProcessor` in python/sglang/srt/sampling/custom_logit_processor.py
    # Use the processor's `to_str()` method to generate the serialized string.
    custom_logit_processor: Optional[Union[List[Optional[str]], str]] = None

    # For disaggregated sampling
    bootstrap_host: Optional[Union[List[str], str]] = None
    bootstrap_port: Optional[Union[List[Optional[int]], int]] = None
    bootstrap_room: Optional[Union[List[int], int]] = None
    bootstrap_pair_key: Optional[Union[List[str], str]] = None

    # Validation step duration
    validation_time: Optional[float] = None

    # For data parallel rank routing
    data_parallel_rank: Optional[int] = None

    # For background responses (OpenAI responses API)
    background: bool = False

    # Conversation id used for tracking requests
    conversation_id: Optional[str] = None

    # Priority for the request
    priority: Optional[int] = None

    # Extra key for classifying the request (e.g. cache_salt)
    extra_key: Optional[Union[List[str], str]] = None

    # Whether to disallow logging for this request (e.g. due to ZDR)
    no_logs: bool = False

    # For custom metric labels
    custom_labels: Optional[Dict[str, str]] = None

    # (Internal) Whether to return bytes for image generation
    return_bytes: bool = False

    # Whether to return entropy
    return_entropy: bool = False


class LoadCheckpointRequest(StrictBase):
    model_name: str  # Actor group name for routing
    checkpoint_path: str
    pin_lora: bool = False


class ShutdownSamplingActorRequest(StrictBase):
    model_name: str  # Actor group name for routing


class PollResultRequest(BaseModel):
    request_id: str
    model_name: Optional[str] = None  # Actor group name (optional, for context)
