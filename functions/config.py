# app/config.py

import os
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class Environment(Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class LogLevel(Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class OpenAIConfig:
    """OpenAI API configuration"""
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o"))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("OPENAI_MAX_TOKENS", "300")))
    temperature: float = field(default_factory=lambda: float(os.getenv("OPENAI_TEMPERATURE", "0.7")))
    top_p: float = field(default_factory=lambda: float(os.getenv("OPENAI_TOP_P", "0.9")))
    timeout: int = field(default_factory=lambda: int(os.getenv("OPENAI_TIMEOUT", "30")))
    max_retries: int = field(default_factory=lambda: int(os.getenv("OPENAI_MAX_RETRIES", "3")))
    retry_delay: float = field(default_factory=lambda: float(os.getenv("OPENAI_RETRY_DELAY", "1.0")))
    rate_limit_calls: int = field(default_factory=lambda: int(os.getenv("OPENAI_RATE_LIMIT_CALLS", "10")))
    rate_limit_window: float = field(default_factory=lambda: float(os.getenv("OPENAI_RATE_LIMIT_WINDOW", "60.0")))

@dataclass
class PlannerConfig:
    """Planner configuration"""
    default_language: str = field(default_factory=lambda: os.getenv("PLANNER_DEFAULT_LANGUAGE", "thai"))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("PLANNER_MAX_TOKENS", "200")))
    temperature: float = field(default_factory=lambda: float(os.getenv("PLANNER_TEMPERATURE", "0.7")))
    top_p: float = field(default_factory=lambda: float(os.getenv("PLANNER_TOP_P", "0.9")))
    enable_emojis: bool = field(default_factory=lambda: os.getenv("PLANNER_ENABLE_EMOJIS", "true").lower() == "true")
    enable_motivation: bool = field(default_factory=lambda: os.getenv("PLANNER_ENABLE_MOTIVATION", "true").lower() == "true")

@dataclass
class FirebaseConfig:
    """Firebase configuration"""
    project_id: str = field(default_factory=lambda: os.getenv("FIREBASE_PROJECT_ID", ""))
    region: str = field(default_factory=lambda: os.getenv("FIREBASE_REGION", "us-central1"))
    max_instances: int = field(default_factory=lambda: int(os.getenv("FIREBASE_MAX_INSTANCES", "5")))

@dataclass
class SecurityConfig:
    """Security configuration"""
    enable_input_validation: bool = field(default_factory=lambda: os.getenv("SECURITY_ENABLE_INPUT_VALIDATION", "true").lower() == "true")
    enable_rate_limiting: bool = field(default_factory=lambda: os.getenv("SECURITY_ENABLE_RATE_LIMITING", "true").lower() == "true")
    max_input_length: int = field(default_factory=lambda: int(os.getenv("SECURITY_MAX_INPUT_LENGTH", "10000")))
    allowed_languages: list = field(default_factory=lambda: os.getenv("SECURITY_ALLOWED_LANGUAGES", "en,th,zh,ja,ko").split(","))

@dataclass
class MonitoringConfig:
    """Monitoring and logging configuration"""
    log_level: LogLevel = field(default_factory=lambda: LogLevel(os.getenv("LOG_LEVEL", "INFO").upper()))
    enable_metrics: bool = field(default_factory=lambda: os.getenv("MONITORING_ENABLE_METRICS", "true").lower() == "true")
    enable_performance_logging: bool = field(default_factory=lambda: os.getenv("MONITORING_ENABLE_PERFORMANCE", "true").lower() == "true")
    log_format: str = field(default_factory=lambda: os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

@dataclass
class AppConfig:
    """Main application configuration"""
    environment: Environment = field(default_factory=lambda: Environment(os.getenv("ENVIRONMENT", "development")))
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    firebase: FirebaseConfig = field(default_factory=FirebaseConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self._validate_config()
        self._setup_logging()
    
    def _validate_config(self):
        """Validate configuration values"""
        if not self.openai.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        if not self.firebase.project_id:
            logger.warning("FIREBASE_PROJECT_ID not set, using default")
        
        if self.openai.max_tokens <= 0:
            raise ValueError("OPENAI_MAX_TOKENS must be positive")
        
        if not (0 <= self.openai.temperature <= 2):
            raise ValueError("OPENAI_TEMPERATURE must be between 0 and 2")
        
        if not (0 <= self.openai.top_p <= 1):
            raise ValueError("OPENAI_TOP_P must be between 0 and 1")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=self.monitoring.log_level.value,
            format=self.monitoring.log_format,
            force=True
        )
        
        # Set specific logger levels
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (for logging/debugging)"""
        return {
            "environment": self.environment.value,
            "debug": self.debug,
            "openai": {
                "model": self.openai.model,
                "max_tokens": self.openai.max_tokens,
                "temperature": self.openai.temperature,
                "top_p": self.openai.top_p,
                "timeout": self.openai.timeout,
                "max_retries": self.openai.max_retries,
                "rate_limit_calls": self.openai.rate_limit_calls,
                "rate_limit_window": self.openai.rate_limit_window
            },
            "planner": {
                "default_language": self.planner.default_language,
                "max_tokens": self.planner.max_tokens,
                "temperature": self.planner.temperature,
                "enable_emojis": self.planner.enable_emojis,
                "enable_motivation": self.planner.enable_motivation
            },
            "firebase": {
                "project_id": self.firebase.project_id,
                "region": self.firebase.region,
                "max_instances": self.firebase.max_instances
            },
            "security": {
                "enable_input_validation": self.security.enable_input_validation,
                "enable_rate_limiting": self.security.enable_rate_limiting,
                "max_input_length": self.security.max_input_length,
                "allowed_languages": self.security.allowed_languages
            },
            "monitoring": {
                "log_level": self.monitoring.log_level.value,
                "enable_metrics": self.monitoring.enable_metrics,
                "enable_performance_logging": self.monitoring.enable_performance_logging
            }
        }
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment == Environment.PRODUCTION
    
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment == Environment.DEVELOPMENT

# Global configuration instance
_config: Optional[AppConfig] = None

def get_config() -> AppConfig:
    """Get the global configuration instance"""
    global _config
    if _config is None:
        _config = AppConfig()
        logger.info(f"Configuration loaded for environment: {_config.environment.value}")
        if _config.debug:
            logger.debug(f"Configuration: {_config.to_dict()}")
    return _config

def reload_config() -> AppConfig:
    """Reload configuration from environment variables"""
    global _config
    _config = None
    return get_config()

# Environment-specific configurations
def get_development_config() -> AppConfig:
    """Get development-specific configuration"""
    os.environ["ENVIRONMENT"] = "development"
    os.environ["DEBUG"] = "true"
    os.environ["LOG_LEVEL"] = "DEBUG"
    return reload_config()

def get_production_config() -> AppConfig:
    """Get production-specific configuration"""
    os.environ["ENVIRONMENT"] = "production"
    os.environ["DEBUG"] = "false"
    os.environ["LOG_LEVEL"] = "INFO"
    return reload_config() 