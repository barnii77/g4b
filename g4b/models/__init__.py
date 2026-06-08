from .model import Model
from .gemma4e import Gemma4E

models: dict[str, type[Model]] = {"gemma4": Gemma4E}

__all__ = ["Model", "models"]
