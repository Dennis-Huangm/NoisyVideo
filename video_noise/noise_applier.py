# Denis
# -*- coding: utf-8 -*-
from typing import Dict, Callable
import torch


class NoiseRegistry:
    _instance = None
    _registry: Dict[str, Callable[[torch.Tensor, float], torch.Tensor]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register(cls, name: str) -> Callable:
        def decorator(func: Callable) -> Callable:
            cls._registry[name] = func
            return func
        return decorator

    @classmethod
    def get_noise(cls, name: str) -> Callable:
        if name not in cls._registry:
            raise ValueError(f"Noise type '{name}' not registered")
        return cls._registry[name]

    @classmethod
    def list_noises(cls) -> list:
        return list(cls._registry.keys())
