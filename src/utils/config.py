"""
Configuration management for the Manifold Swarms Trading Bot
"""
import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass
from pydantic import BaseModel, Field


@dataclass
class BotConfig:
    """Main bot configuration"""
    name: str
    version: str
    description: str


@dataclass
class TradingConfig:
    """Trading parameters configuration"""
    target_user: str
    check_interval: int
    max_position_size: float
    max_active_positions: int
    min_consensus_confidence: float


@dataclass
class RiskConfig:
    """Risk management configuration"""
    max_total_exposure: float
    min_balance_reserve: float
    stop_loss_threshold: float
    take_profit_threshold: float
    max_single_market_exposure: float


@dataclass
class LLMConfig:
    """LLM provider configuration"""
    provider: str
    model: str
    temperature: float
    max_tokens: int
    fallback_providers: list


@dataclass
class AgentConfig:
    """Agent configuration"""
    timeout: int
    max_retries: int
    parallel_analysis: bool
    consensus_threshold: float


@dataclass
class WorkflowConfig:
    """Workflow configuration"""
    main_workflow: str
    analysis_pattern: str
    consensus_pattern: str
    max_loops: int


@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    log_level: str
    enable_dashboard: bool
    dashboard_port: int
    prometheus_port: int
    save_agent_states: bool


class ConfigManager:
    """Centralized configuration management"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self._config_cache: Dict[str, Any] = {}
        self._load_all_configs()
    
    def _load_all_configs(self):
        """Load all configuration files"""
        # Load main config
        self._config_cache['main'] = self._load_yaml('config.yaml')
        
        # Load agent configs
        self._config_cache['agents'] = self._load_yaml('agents.yaml')
        
        # Load workflow configs
        self._config_cache['workflows'] = self._load_yaml('workflows.yaml')
        
        # Load risk limits
        self._config_cache['risk'] = self._load_yaml('risk_limits.yaml')
        
        # Load environment variables
        self._load_env_vars()
    
    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load YAML configuration file"""
        filepath = self.config_dir / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def _load_env_vars(self):
        """Load environment variables"""
        env_config = {}
        
        # API Keys
        env_config['manifold_api_key'] = os.getenv('MANIFOLD_API_KEY')
        env_config['openai_api_key'] = os.getenv('OPENAI_API_KEY')
        env_config['anthropic_api_key'] = os.getenv('ANTHROPIC_API_KEY')
        env_config['cohere_api_key'] = os.getenv('COHERE_API_KEY')
        
        # Database
        env_config['database_url'] = os.getenv('DATABASE_URL')
        env_config['sqlite_url'] = os.getenv('SQLITE_URL')
        
        # Redis
        env_config['redis_host'] = os.getenv('REDIS_HOST', 'localhost')
        env_config['redis_port'] = int(os.getenv('REDIS_PORT', 6379))
        env_config['redis_db'] = int(os.getenv('REDIS_DB', 0))
        
        self._config_cache['env'] = env_config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self._config_cache
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_bot_config(self) -> BotConfig:
        """Get bot configuration"""
        main_config = self.get('bot', {})
        return BotConfig(
            name=main_config.get('name', 'Manifold Swarms Bot'),
            version=main_config.get('version', '1.0.0'),
            description=main_config.get('description', 'Multi-agent trading system')
        )
    
    def get_trading_config(self) -> TradingConfig:
        """Get trading configuration"""
        trading_config = self.get('trading', {})
        return TradingConfig(
            target_user=trading_config.get('target_user', 'MikhailTal'),
            check_interval=trading_config.get('check_interval', 300),
            max_position_size=trading_config.get('max_position_size', 0.10),
            max_active_positions=trading_config.get('max_active_positions', 5),
            min_consensus_confidence=trading_config.get('min_consensus_confidence', 0.65)
        )
    
    def get_risk_config(self) -> RiskConfig:
        """Get risk configuration"""
        risk_config = self.get('risk', {})
        return RiskConfig(
            max_total_exposure=risk_config.get('max_total_exposure', 0.80),
            min_balance_reserve=risk_config.get('min_balance_reserve', 0.10),
            stop_loss_threshold=risk_config.get('stop_loss_threshold', -0.15),
            take_profit_threshold=risk_config.get('take_profit_threshold', 0.25),
            max_single_market_exposure=risk_config.get('max_single_market_exposure', 0.10)
        )
    
    def get_llm_config(self) -> LLMConfig:
        """Get LLM configuration"""
        llm_config = self.get('llm', {})
        return LLMConfig(
            provider=llm_config.get('provider', 'openai'),
            model=llm_config.get('model', 'gpt-4'),
            temperature=llm_config.get('temperature', 0.1),
            max_tokens=llm_config.get('max_tokens', 2000),
            fallback_providers=llm_config.get('fallback_providers', ['anthropic', 'cohere'])
        )
    
    def get_agent_config(self) -> AgentConfig:
        """Get agent configuration"""
        agent_config = self.get('agents', {})
        return AgentConfig(
            timeout=agent_config.get('timeout', 300),
            max_retries=agent_config.get('max_retries', 3),
            parallel_analysis=agent_config.get('parallel_analysis', True),
            consensus_threshold=agent_config.get('consensus_threshold', 0.65)
        )
    
    def get_workflow_config(self) -> WorkflowConfig:
        """Get workflow configuration"""
        workflow_config = self.get('workflows', {})
        return WorkflowConfig(
            main_workflow=workflow_config.get('main_workflow', 'sequential'),
            analysis_pattern=workflow_config.get('analysis_pattern', 'agent_rearrange'),
            consensus_pattern=workflow_config.get('consensus_pattern', 'mixture_of_agents'),
            max_loops=workflow_config.get('max_loops', 1)
        )
    
    def get_monitoring_config(self) -> MonitoringConfig:
        """Get monitoring configuration"""
        monitoring_config = self.get('monitoring', {})
        return MonitoringConfig(
            log_level=monitoring_config.get('log_level', 'INFO'),
            enable_dashboard=monitoring_config.get('enable_dashboard', True),
            dashboard_port=monitoring_config.get('dashboard_port', 8501),
            prometheus_port=monitoring_config.get('prometheus_port', 8000),
            save_agent_states=monitoring_config.get('save_agent_states', True)
        )
    
    def get_agent_prompt(self, agent_name: str) -> Optional[str]:
        """Get system prompt for specific agent"""
        return self.get(f'agents.{agent_name}.system_prompt')
    
    def get_workflow_config_by_name(self, workflow_name: str) -> Optional[Dict[str, Any]]:
        """Get specific workflow configuration"""
        workflows = self.get('workflows', {})
        return workflows.get(workflow_name)
    
    def get_risk_limits(self) -> Dict[str, Any]:
        """Get risk limits configuration"""
        return self.get('risk_limits', {})
    
    def is_valid(self) -> bool:
        """Validate configuration"""
        required_keys = [
            'env.manifold_api_key',
            'env.openai_api_key'
        ]
        
        for key in required_keys:
            if not self.get(key):
                print(f"Missing required configuration: {key}")
                return False
        
        return True


# Global configuration instance
config = ConfigManager()