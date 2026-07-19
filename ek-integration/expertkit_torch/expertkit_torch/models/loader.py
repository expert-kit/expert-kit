"""Load supported Hugging Face models with routed Expert Kit MoE blocks."""

from __future__ import annotations

import threading
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from expertkit_torch.client import RoutedMoEClient
from expertkit_torch.models._common import RoutedLayerIds

ModelMode = Literal["expertkit", "local"]
ModelDType = Literal["auto", "float16", "bfloat16", "float32"]
ModelTransport = Literal["grpc", "shm"]

_MODEL_LOAD_LOCK = threading.Lock()
_MISSING = object()
_ROUTED_EXPERT_WEIGHT_PATTERN = r"layers\.\d+\.(?:mlp|block_sparse_moe)\.experts\.\d+\."
_DTYPES: dict[ModelDType, str | torch.dtype] = {
    "auto": "auto",
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


@dataclass(frozen=True)
class _AdapterSpec:
    module: ModuleType
    class_name: str
    layer_ids: Callable[[Any], tuple[int, ...]]
    experts_per_layer: Callable[[Any], int]
    create_class: Callable[[RoutedMoEClient, RoutedLayerIds], type[torch.nn.Module]]


@dataclass
class LoadedModel:
    """Own one loaded model, tokenizer, and optional Expert Kit client."""

    model: Any
    tokenizer: Any
    model_type: str
    _client: RoutedMoEClient | None = None
    _closed: bool = False

    def close(self) -> None:
        """Close the model's Transport client; repeated calls are safe."""

        if self._closed:
            return
        self._closed = True
        if self._client is not None:
            self._client.close()

    def __enter__(self) -> LoadedModel:
        """Return this loaded model for use in a synchronous context."""

        if self._closed:
            raise RuntimeError("loaded model is already closed")
        return self

    def __exit__(self, *_: object) -> None:
        """Close the owned Expert Kit client."""

        self.close()


def _qwen_layer_ids(config: Any) -> tuple[int, ...]:
    mlp_only_layers = set(config.mlp_only_layers or ())
    return tuple(
        layer_id
        for layer_id in range(config.num_hidden_layers)
        if layer_id not in mlp_only_layers
        and config.num_experts > 0
        and (layer_id + 1) % config.decoder_sparse_step == 0
    )


def _deepseek_layer_ids(config: Any) -> tuple[int, ...]:
    return tuple(range(config.first_k_dense_replace, config.num_hidden_layers))


def _all_layer_ids(config: Any) -> tuple[int, ...]:
    return tuple(range(config.num_hidden_layers))


def _adapter_spec(model_type: str) -> _AdapterSpec:
    if model_type == "qwen3_moe":
        from transformers.models.qwen3_moe import modeling_qwen3_moe

        from expertkit_torch.models.qwen3_moe import create_routed_moe_class

        return _AdapterSpec(
            modeling_qwen3_moe,
            "Qwen3MoeSparseMoeBlock",
            _qwen_layer_ids,
            lambda config: config.num_experts,
            create_routed_moe_class,
        )
    if model_type == "deepseek_v2":
        from transformers.models.deepseek_v2 import modeling_deepseek_v2

        from expertkit_torch.models.deepseek_v2 import create_routed_moe_class

        return _AdapterSpec(
            modeling_deepseek_v2,
            "DeepseekV2MoE",
            _deepseek_layer_ids,
            lambda config: config.n_routed_experts,
            create_routed_moe_class,
        )
    if model_type == "deepseek_v3":
        from transformers.models.deepseek_v3 import modeling_deepseek_v3

        from expertkit_torch.models.deepseek_v3 import create_routed_moe_class

        return _AdapterSpec(
            modeling_deepseek_v3,
            "DeepseekV3MoE",
            _deepseek_layer_ids,
            lambda config: config.n_routed_experts,
            create_routed_moe_class,
        )
    if model_type == "mixtral":
        from transformers.models.mixtral import modeling_mixtral

        from expertkit_torch.models.mixtral import create_routed_moe_class

        return _AdapterSpec(
            modeling_mixtral,
            "MixtralSparseMoeBlock",
            _all_layer_ids,
            lambda config: config.num_local_experts,
            create_routed_moe_class,
        )
    raise ValueError(
        f"unsupported model_type {model_type!r}; expected qwen3_moe, "
        "deepseek_v2, deepseek_v3, or mixtral"
    )


@contextmanager
def _temporary_model_class(
    spec: _AdapterSpec,
    replacement: type[torch.nn.Module],
):
    original = getattr(spec.module, spec.class_name)
    setattr(spec.module, spec.class_name, replacement)
    try:
        yield
    finally:
        setattr(spec.module, spec.class_name, original)


@contextmanager
def _ignore_expected_routed_expert_weights(config: Any):
    """Hide only checkpoint keys intentionally omitted from the Frontend model."""

    model_class = AutoModelForCausalLM._model_mapping[type(config)]
    attribute = "_keys_to_ignore_on_load_unexpected"
    original = model_class.__dict__.get(attribute, _MISSING)
    current = list(getattr(model_class, attribute, None) or ())
    if _ROUTED_EXPERT_WEIGHT_PATTERN not in current:
        current.append(_ROUTED_EXPERT_WEIGHT_PATTERN)
    setattr(model_class, attribute, current)
    try:
        yield
    finally:
        if original is _MISSING:
            delattr(model_class, attribute)
        else:
            setattr(model_class, attribute, original)


def _validate_loading_arguments(
    *,
    model_path: str | Path,
    mode: ModelMode,
    controller_endpoint: str,
    instance_id: int,
    dtype: ModelDType,
    transport: ModelTransport,
) -> str:
    resolved_path = str(model_path)
    if not resolved_path.strip():
        raise ValueError("model_path must not be empty")
    if mode not in ("expertkit", "local"):
        raise ValueError("mode must be 'expertkit' or 'local'")
    if dtype not in _DTYPES:
        raise ValueError("dtype must be auto, float16, bfloat16, or float32")
    if transport not in ("grpc", "shm"):
        raise ValueError("transport must be 'grpc' or 'shm'")
    if mode == "expertkit":
        if not controller_endpoint.strip():
            raise ValueError("controller_endpoint must not be empty")
        if isinstance(instance_id, bool) or not isinstance(instance_id, int) or instance_id <= 0:
            raise ValueError("instance_id must be a positive integer")
    return resolved_path


def load_model(
    model_path: str | Path,
    *,
    mode: ModelMode = "expertkit",
    controller_endpoint: str = "127.0.0.1:5002",
    instance_id: int = 1,
    device: str | torch.device = "cuda:0",
    dtype: ModelDType = "auto",
    transport: ModelTransport = "grpc",
) -> LoadedModel:
    """Load one supported causal language model.

    Args:
        model_path: Local Hugging Face checkpoint directory.
        mode: Whether routed experts use Expert Kit or remain local.
        controller_endpoint: Controller topology endpoint used in Expert Kit mode.
        instance_id: Model instance registered with the Controller.
        device: Single Frontend device that owns attention and routing.
        dtype: Checkpoint loading dtype or ``auto`` to use checkpoint metadata.
        transport: Worker data path. Shared memory requires the same Host.

    Returns:
        A context-manageable object owning the model, tokenizer, and Transport
        client. The loaded model is in evaluation mode on ``device``.

    Raises:
        ValueError: Arguments or the checkpoint model type are unsupported.
        RuntimeError: The constructed model does not match its configured routed
            layer count.
    """

    resolved_path = _validate_loading_arguments(
        model_path=model_path,
        mode=mode,
        controller_endpoint=controller_endpoint,
        instance_id=instance_id,
        dtype=dtype,
        transport=transport,
    )
    config = AutoConfig.from_pretrained(resolved_path)
    model_type = str(config.model_type)
    spec = _adapter_spec(model_type)
    tokenizer = AutoTokenizer.from_pretrained(resolved_path)
    resolved_device = torch.device(device)
    client: RoutedMoEClient | None = None

    try:
        with _MODEL_LOAD_LOCK:
            if mode == "expertkit":
                layer_ids = RoutedLayerIds(spec.layer_ids(config))
                if not layer_ids.values:
                    raise ValueError(f"{model_type} configuration has no routed MoE layers")
                client = RoutedMoEClient(
                    controller_endpoint,
                    instance_id=instance_id,
                    num_layers=config.num_hidden_layers,
                    experts_per_layer=spec.experts_per_layer(config),
                    hidden_dim=config.hidden_size,
                    top_k=config.num_experts_per_tok,
                    transport=transport,
                )
                replacement = spec.create_class(client, layer_ids)
                with (
                    _temporary_model_class(spec, replacement),
                    _ignore_expected_routed_expert_weights(config),
                ):
                    model = AutoModelForCausalLM.from_pretrained(
                        resolved_path,
                        dtype=_DTYPES[dtype],
                    )
                layer_ids.require_complete()
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    resolved_path,
                    dtype=_DTYPES[dtype],
                )
        model = model.to(resolved_device)
        model.eval()
    except BaseException:
        if client is not None:
            client.close()
        raise

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        model_type=model_type,
        _client=client,
    )
