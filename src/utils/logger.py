"""
Logging utilities for the Manifold Swarms Trading Bot
"""
import sys
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime
from loguru import logger
from src.utils.config import config


class LoggerSetup:
    """Setup and configure logging for the application"""
    
    def __init__(self):
        self.log_level = config.get('monitoring.log_level', 'INFO')
        self.log_dir = Path('logs')
        self.log_dir.mkdir(exist_ok=True)
        
    def setup_logger(self):
        """Setup loguru logger with custom configuration"""
        # Remove default logger
        logger.remove()
        
        # Console logger
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=self.log_level,
            colorize=True
        )
        
        # File logger - all logs
        logger.add(
            self.log_dir / "manifold_swarms.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG",
            rotation="10 MB",
            retention="30 days",
            compression="zip"
        )
        
        # File logger - errors only
        logger.add(
            self.log_dir / "errors.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="ERROR",
            rotation="5 MB",
            retention="90 days",
            compression="zip"
        )
        
        # File logger - trading decisions
        logger.add(
            self.log_dir / "trading.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {extra[agent]} | {extra[market]} | {extra[action]} | {message}",
            level="INFO",
            filter=lambda record: "agent" in record["extra"],
            rotation="5 MB",
            retention="90 days"
        )
        
        return logger


# Setup logger instance
logger_setup = LoggerSetup()
log = logger_setup.setup_logger()


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name"""
    return logger.bind(name=name)


def log_trading_decision(agent: str, market: str, action: str, message: str):
    """Log trading decisions with structured format"""
    log.bind(agent=agent, market=market, action=action).info(message)


def log_agent_performance(agent: str, metric: str, value: float):
    """Log agent performance metrics"""
    log.bind(agent=agent, metric=metric).info(f"Performance: {metric} = {value}")


def log_workflow_start(workflow_name: str, agents: list):
    """Log workflow initiation"""
    log.info(f"Starting workflow: {workflow_name} with agents: {', '.join(agents)}")


def log_workflow_complete(workflow_name: str, duration: float, result: str):
    """Log workflow completion"""
    log.info(f"Completed workflow: {workflow_name} in {duration:.2f}s - Result: {result}")


def log_error(error: Exception, context: str = ""):
    """Log errors with context"""
    if context:
        log.error(f"Error in {context}: {str(error)}")
    else:
        log.error(f"Error: {str(error)}")


def log_api_call(endpoint: str, method: str, status_code: int, response_time: float):
    """Log API calls for monitoring"""
    log.debug(f"API Call: {method} {endpoint} - {status_code} in {response_time:.2f}s")


class AgentLogger:
    """Specialized logger for agent activities"""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.logger = logger.bind(agent=agent_name)
    
    def signal_generated(self, market_id: str, action: str, confidence: float, reasoning: str):
        """Log agent signal generation"""
        self.logger.info(f"Signal: {action} on {market_id} (confidence: {confidence:.2f}) - {reasoning[:100]}...")
    
    def analysis_complete(self, market_id: str, score: float, key_findings: list):
        """Log analysis completion"""
        findings_str = ", ".join(key_findings[:3])
        self.logger.info(f"Analysis: {market_id} (score: {score:.2f}) - {findings_str}")
    
    def error(self, error: Exception, context: str = ""):
        """Log agent error"""
        self.logger.error(f"Agent error: {context} - {str(error)}")
    
    def communication(self, target_agent: str, message_type: str, content: str):
        """Log agent communication"""
        self.logger.debug(f"Comm: {message_type} to {target_agent} - {content[:50]}...")


class WorkflowLogger:
    """Specialized logger for workflow activities"""
    
    def __init__(self, workflow_name: str):
        self.workflow_name = workflow_name
        self.logger = logger.bind(workflow=workflow_name)
    
    def stage_start(self, stage_name: str, agent: str):
        """Log workflow stage start"""
        self.logger.info(f"Stage start: {stage_name} by {agent}")
    
    def stage_complete(self, stage_name: str, agent: str, duration: float):
        """Log workflow stage completion"""
        self.logger.info(f"Stage complete: {stage_name} by {agent} in {duration:.2f}s")
    
    def workflow_complete(self, total_duration: float, result_summary: str):
        """Log workflow completion"""
        self.logger.info(f"Workflow complete in {total_duration:.2f}s - {result_summary}")
    
    def workflow_failed(self, error: Exception, failed_stage: str):
        """Log workflow failure"""
        self.logger.error(f"Workflow failed at {failed_stage}: {str(error)}")


class TradingLogger:
    """Specialized logger for trading activities"""
    
    def __init__(self):
        self.logger = logger.bind(category="trading")
    
    def trade_executed(self, market_id: str, action: str, amount: float, price: float):
        """Log trade execution"""
        self.logger.info(f"Trade: {action} {amount:.2f}M$ on {market_id} @ {price:.2f}")
    
    def position_opened(self, market_id: str, stake: float, outcome: str):
        """Log position opening"""
        self.logger.info(f"Position opened: {market_id} - {stake:.2f}M$ on {outcome}")
    
    def position_closed(self, market_id: str, pnl: float, pnl_percent: float):
        """Log position closing"""
        profit_loss = "profit" if pnl > 0 else "loss"
        self.logger.info(f"Position closed: {market_id} - {profit_loss} {abs(pnl):.2f}M$ ({pnl_percent:+.1%})")
    
    def risk_breach(self, breach_type: str, current_value: float, limit: float):
        """Log risk limit breaches"""
        self.logger.warning(f"Risk breach: {breach_type} - {current_value:.2f} > {limit:.2f}")
    
    def portfolio_summary(self, balance: float, exposure: float, positions: int, daily_pnl: float):
        """Log portfolio summary"""
        self.logger.info(f"Portfolio: {balance:.2f}M$ balance, {exposure:.1%} exposure, {positions} positions, P&L: {daily_pnl:+.2f}M$")