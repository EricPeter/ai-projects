import anthropic
from .base_model import BaseModel


class AnthropicModel(BaseModel):
    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model
        self._client = anthropic.Anthropic()

    def predict(self, prompt: str, context: str = "") -> str:
        content = f"Context:\n{context}\n\n{prompt}" if context else prompt
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=512,
            system="You are a legal AI assistant. Answer precisely and concisely.",
            messages=[{"role": "user", "content": content}],
        )
        return msg.content[0].text
