import os
from dataclasses import dataclass
from transformers import PretrainedConfig
from typing import Optional

@dataclass
class EkClientConfig:
    ek_mode: str = "expert_mode"  # or "moe_mode"
    ek_addr: str = "localhost:5002"
    ek_debug_mode: bool = False
    ek_client_timeout: int = 2  # seconds
    ek_model_name: str = ""

def collect_ek_client_cfg(cfg: Optional[PretrainedConfig]=None) -> EkClientConfig:
    # init cfg from file and env variables
    ek_cfg = EkClientConfig()

    # try get config from model config
    if cfg is not None:
        ek_cfg.ek_mode = getattr(cfg, "ek_mode", ek_cfg.ek_mode)
        ek_cfg.ek_addr = getattr(cfg, "ek_backend_addr", ek_cfg.ek_addr)
        ek_cfg.ek_debug_mode = getattr(cfg, "ek_debug_mode", ek_cfg.ek_debug_mode)
        ek_cfg.ek_client_timeout = getattr(cfg, "ek_client_timeout", ek_cfg.ek_client_timeout)
        ek_cfg.ek_model_name = getattr(cfg, "ek_model_name", cfg.model_name_or_path)

    # then from environment variables
    ek_cfg.ek_mode = os.getenv("EK_MODE", ek_cfg.ek_mode)
    ek_cfg.ek_addr = os.getenv("EK_ADDR", ek_cfg.ek_addr)
    ek_cfg.ek_debug_mode = os.getenv("EK_DEBUG_MODE", str(ek_cfg.ek_debug_mode)) == "1"
    ek_cfg.ek_client_timeout = os.getenv("EK_CLIENT_TIMEOUT", int(ek_cfg.ek_client_timeout))
    ek_cfg.ek_model_name = os.getenv("EK_MODEL_NAME", ek_cfg.ek_model_name)

    if not ek_cfg.ek_model_name:
        raise ValueError("EK_MODEL_NAME must be set in config or environment variables")

    # type essure
    ek_cfg.ek_client_timeout = int(ek_cfg.ek_client_timeout)
    ek_cfg.ek_debug_mode = bool(ek_cfg.ek_debug_mode)
    ek_cfg.ek_mode = str(ek_cfg.ek_mode)
    ek_cfg.ek_addr = str(ek_cfg.ek_addr)

    return ek_cfg