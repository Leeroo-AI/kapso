"""
Configuration settings for Leeroopedia Content Service.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Wiki data path (mounted volume)
    wiki_data_path: str = "/wikis"

    # MediaWiki database settings for authentication
    db_host: str = "localhost"
    db_name: str = "wiki"
    db_user: str = "mediawiki"
    db_password: str = ""

    # Signup URL for 401 responses
    signup_url: str = "https://leeroopedia.com/"

    # Rate limiting
    export_rate_limit: int = 10  # requests per hour
    export_rate_window: int = 3600  # seconds (1 hour)

    # Export output directory (for saving exports to disk)
    export_output_dir: str = "/exports"

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
