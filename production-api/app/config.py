from functools import lru_cache
from pydantic_settings import BaseSettings

class Settings(BaseSettings):

    openai_api_key: str

    primary_model: str = "gpt-4o-mini"
    fallback_model: str = "gpt-5-mini"

    app_env: str = "development"

    @property
    def is_production(self) -> bool:
        return self.app_env == 'production'

@lru_cache
def get_settings() -> Settings:
    return Settings()
