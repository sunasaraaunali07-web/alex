from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    GEMINI_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-3.8-flash"
    GROQ_LLM_MODEL: str = "qwen/qwen3.8-27b"
    FALLBACK_TTS_VOICE: str = "en-US-GuyNeural"
    GROQ_STT_MODEL: str = "whisper-large-v3-turbo"
    DEFAULT_TTS_VOICE: str = "en-US-JennyNeural"
    
    class Config:
        env_file = ".env"
        extra = "ignore"

def get_settings() -> Settings:
    """Always re-reads .env so keys take effect immediately without restarting."""
    return Settings(_env_file=".env")
