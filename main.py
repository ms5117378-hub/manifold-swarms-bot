"""
Main entry point for the Manifold Swarms Trading Bot
"""
import asyncio
import sys
import os
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.swarms_core.communication import AgentCoordinator
from src.swarms_core.workflows import TradingWorkflowManager
from src.manifold.market_fetcher import MarketDiscovery, MarketMonitor
from src.agents import (
    FundamentalAnalyst, TechnicalAnalyst, SentimentAnalyst,
    ValueInvestor, MomentumTrader, MeanReversionTrader, ArbitrageFinder,
    TradingOrchestrator, RiskManager, TradeExecutor, PortfolioManager
)
from src.utils.config import config
from src.utils.logger import get_logger, log

# Initialize logger
logger = get_logger(__name__)


class ManifoldSwarmsBot:
    """Main bot class that orchestrates the entire trading system"""
    
    def __init__(self):
        self.coordinator = AgentCoordinator()
        self.workflow_manager = TradingWorkflowManager(self.coordinator)
        self.market_discovery = MarketDiscovery()
        self.market_monitor = MarketMonitor(self.market_discovery)
        
        self.is_running = False
        self.startup_time = datetime.now()
        
    async def initialize(self):
        """Initialize the trading bot"""
        try:
            logger.info("Initializing Manifold Swarms Trading Bot...")
            
            # Validate configuration
            if not config.is_valid():
                logger.error("Invalid configuration. Please check your settings.")
                return False
            
            # Initialize workflows (this will create and register all agents)
            logger.info("Initializing workflows and agents...")
            # Workflows are automatically initialized in TradingWorkflowManager
            
            # Start market monitoring
            logger.info("Starting market monitoring...")
            # Market monitoring will be started in run() method
            
            logger.info("Bot initialization completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize bot: {str(e)}")
            return False
    
    async def run(self):
        """Main bot execution loop"""
        if not await self.initialize():
            return
        
        self.is_running = True
        logger.info("Starting Manifold Swarms Trading Bot...")
        
        try:
            # Start market monitoring in background
            monitor_task = asyncio.create_task(self.market_monitor.start_monitoring())
            
            # Main trading loop
            while self.is_running:
                try:
                    # Discover new markets
                    markets = await self.market_discovery.discover_markets()
                    logger.info(f"Discovered {len(markets)} markets")
                    
                    # Analyze each market
                    for market in markets[:5]:  # Limit to 5 markets per cycle
                        try:
                            # Execute workflow for market
                            result = await self.workflow_manager.execute_market_analysis(market)
                            
                            if result:
                                logger.info(f"Analysis completed for {market.id}: {result.get('status', 'unknown')}")
                            else:
                                logger.warning(f"Analysis failed for {market.id}")
                            
                            # Small delay between markets
                            await asyncio.sleep(1)
                            
                        except Exception as e:
                            logger.error(f"Error analyzing market {market.id}: {str(e)}")
                    
                    # Wait for next cycle
                    check_interval = config.get('trading.check_interval', 300)  # 5 minutes
                    logger.info(f"Waiting {check_interval} seconds for next cycle...")
                    await asyncio.sleep(check_interval)
                    
                except Exception as e:
                    logger.error(f"Error in main trading loop: {str(e)}")
                    await asyncio.sleep(60)  # Wait 1 minute before retrying
            
        except KeyboardInterrupt:
            logger.info("Received shutdown signal...")
        except Exception as e:
            logger.error(f"Fatal error in main loop: {str(e)}")
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Gracefully shutdown the bot"""
        logger.info("Shutting down Manifold Swarms Trading Bot...")
        
        self.is_running = False
        
        # Stop market monitoring
        self.market_monitor.stop_monitoring()
        
        # Save agent states
        for agent_name, agent in self.coordinator.active_agents.items():
            try:
                state_file = f"agent_states/{agent_name.lower().replace(' ', '_')}_state.json"
                agent.save_state(state_file)
            except Exception as e:
                logger.error(f"Error saving state for {agent_name}: {str(e)}")
        
        # Log final statistics
        uptime = datetime.now() - self.startup_time
        logger.info(f"Bot shutdown complete. Uptime: {uptime}")
        
        # Get workflow statistics
        stats = self.workflow_manager.get_workflow_statistics()
        logger.info(f"Workflow Statistics: {stats}")
    
    def get_status(self):
        """Get current bot status"""
        return {
            'is_running': self.is_running,
            'uptime': str(datetime.now() - self.startup_time) if self.startup_time else "Not started",
            'active_agents': len(self.coordinator.active_agents),
            'workflow_stats': self.workflow_manager.get_workflow_statistics(),
            'market_stats': self.market_discovery.get_market_statistics()
        }


async def main():
    """Main entry point"""
    bot = ManifoldSwarmsBot()
    
    try:
        await bot.run()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("agent_states", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Run the bot
    asyncio.run(main())