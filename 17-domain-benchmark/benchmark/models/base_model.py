from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def predict(self, prompt: str, context: str = "") -> str:
        raise NotImplementedError
