from abc import ABC, abstractmethod
from typing import Any, Dict, Type

# Registry for perturbation types
PERTURBATION_REGISTRY: Dict[str, Type['Perturbation']] = {}

def register_perturbation(name: str):
    """Decorator to register a perturbation class."""
    def decorator(cls: Type['Perturbation']):
        PERTURBATION_REGISTRY[name] = cls
        return cls
    return decorator

class Perturbation(ABC):
    """Base class for all perturbations."""

    @abstractmethod
    def apply(self, data: Any, **kwargs) -> Any:
        """Apply the perturbation to the data.

        Args:
            data: The input data (text or image).
            **kwargs: Additional parameters for the perturbation.

        Returns:
            The perturbed data.
        """
        pass

def apply_perturbation(data: Any, perturbation_type: str, **kwargs) -> Any:
    """Apply a registered perturbation to the data.

    Args:
        data: The input data.
        perturbation_type: The name of the perturbation to apply.
        **kwargs: Parameters for the perturbation.

    Returns:
        The perturbed data.

    Raises:
        ValueError: If the perturbation type is not registered.
    """
    if perturbation_type not in PERTURBATION_REGISTRY:
        raise ValueError(f"Perturbation type '{perturbation_type}' not registered.")
    perturbation_class = PERTURBATION_REGISTRY[perturbation_type]
    perturbation = perturbation_class()
    return perturbation.apply(data, **kwargs)