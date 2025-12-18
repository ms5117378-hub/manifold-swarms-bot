"""
Setup script for initializing the Swarms trading system
"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.utils.config import config
from src.utils.logger import get_logger

log = get_logger(__name__)


def setup_swarms():
    """Initialize the Swarms trading system"""
    try:
        log.info("Setting up Swarms trading system...")
        
        # Create necessary directories
        directories = [
            "logs",
            "agent_states", 
            "data",
            "db"
        ]
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            log.info(f"Created directory: {directory}")
        
        # Validate configuration
        if config.is_valid():
            log.info("✅ Configuration is valid")
        else:
            log.warning("⚠️ Configuration validation failed. Please check your .env file")
        
        # Test Swarms import
        try:
            from swarms import Agent
            from swarm_models import OpenAIChat
            log.info("✅ Swarms packages imported successfully")
        except ImportError as e:
            log.error(f"❌ Failed to import Swarms packages: {str(e)}")
            return False
        
        # Test manifoldbot import
        try:
            import manifoldbot
            log.info("✅ Manifoldbot package imported successfully")
        except ImportError as e:
            log.warning(f"⚠️ Manifoldbot package not found: {str(e)}")
        
        log.info("🚀 Swarms trading system setup complete!")
        return True
        
    except Exception as e:
        log.error(f"❌ Setup failed: {str(e)}")
        return False


if __name__ == "__main__":
    success = setup_swarms()
    sys.exit(0 if success else 1)