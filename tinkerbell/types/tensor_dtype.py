from typing_extensions import Literal, TypeAlias

__all__ = ["TensorDtype", "_key_to_type"]

TensorDtype: TypeAlias = Literal["int64", "float32"]

_key_to_type = {
    "target_tokens": "int64",
    "weights": "float32",
    "advantages": "float32",
    "logprobs": "float32",
    "clip_low_threshold": "float32",
    "clip_high_threshold": "float32",
}
