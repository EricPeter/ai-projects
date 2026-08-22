from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_a_url: str = "http://model-a:8000"
    model_b_url: str = "http://model-b:8000"
    traffic_split: float = 0.10  # fraction routed to model B
    min_samples: int = 100        # per variant before significance test runs
    alpha: float = 0.05           # significance level
    router_port: int = 8022

    model_config = {"env_prefix": ""}


settings = Settings()
