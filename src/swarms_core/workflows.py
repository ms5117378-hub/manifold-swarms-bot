"""
Swarms workflow implementations for the Manifold Trading Bot
"""
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

from swarms import SequentialWorkflow, AgentRearrange, MixtureOfAgents
from src.models import Market, AgentSignal, ConsensusDecision
from src.agents import (
    FundamentalAnalyst, TechnicalAnalyst, SentimentAnalyst,
    ValueInvestor, MomentumTrader, MeanReversionTrader, ArbitrageFinder,
    TradingOrchestrator, RiskManager, TradeExecutor, PortfolioManager
)
from src.swarms_core.communication import AgentCoordinator
from src.utils.config import config
from src.utils.logger import get_logger, WorkflowLogger

log = get_logger(__name__)


class ManifoldTradingWorkflows:
    """Container for all Swarms-based trading workflows"""
    
    def __init__(self, coordinator: AgentCoordinator):
        self.coordinator = coordinator
        self.workflow_logger = WorkflowLogger("ManifoldTradingWorkflows")
        
        # Initialize agents
        self._initialize_agents()
        
        # Initialize workflows
        self.sequential_workflow = None
        self.parallel_workflow = None
        self.consensus_workflow = None
        self.hierarchical_workflow = None
        
        # Setup workflows
        self._setup_workflows()
    
    def _initialize_agents(self):
        """Initialize all trading agents"""
        # Analyst agents
        self.fundamental_analyst = FundamentalAnalyst()
        self.technical_analyst = TechnicalAnalyst()
        self.sentiment_analyst = SentimentAnalyst()
        
        # Strategy agents
        self.value_investor = ValueInvestor()
        self.momentum_trader = MomentumTrader()
        self.mean_reversion_trader = MeanReversionTrader()
        self.arbitrage_finder = ArbitrageFinder()
        
        # Management agents
        self.orchestrator = TradingOrchestrator(self.coordinator)
        self.risk_manager = RiskManager()
        self.trade_executor = TradeExecutor()
        self.portfolio_manager = PortfolioManager()
        
        # Register all agents with coordinator
        agents = [
            self.fundamental_analyst, self.technical_analyst, self.sentiment_analyst,
            self.value_investor, self.momentum_trader, self.mean_reversion_trader, self.arbitrage_finder,
            self.orchestrator, self.risk_manager, self.trade_executor, self.portfolio_manager
        ]
        
        for agent in agents:
            self.coordinator.register_agent(agent)
        
        log.info("All agents initialized and registered")
    
    def _setup_workflows(self):
        """Setup all workflow patterns"""
        try:
            # Sequential workflow
            self._setup_sequential_workflow()
            
            # Parallel analysis workflow
            self._setup_parallel_workflow()
            
            # Consensus workflow
            self._setup_consensus_workflow()
            
            # Hierarchical workflow
            self._setup_hierarchical_workflow()
            
            log.info("All workflows setup completed")
            
        except Exception as e:
            log.error(f"Error setting up workflows: {str(e)}")
    
    def _setup_sequential_workflow(self):
        """Setup sequential trading workflow"""
        try:
            self.sequential_workflow = SequentialWorkflow(
                name="Sequential Trading Pipeline",
                description="Sequential execution through all trading stages",
                agents=[
                    self.fundamental_analyst.swarms_agent,
                    self.technical_analyst.swarms_agent,
                    self.sentiment_analyst.swarms_agent,
                    self.value_investor.swarms_agent,
                    self.momentum_trader.swarms_agent,
                    self.mean_reversion_trader.swarms_agent,
                    self.arbitrage_finder.swarms_agent,
                    self.risk_manager.swarms_agent,
                    self.trade_executor.swarms_agent
                ],
                max_loops=1,
                verbose=True
            )
            
            log.info("Sequential workflow setup completed")
            
        except Exception as e:
            log.error(f"Error setting up sequential workflow: {str(e)}")
    
    def _setup_parallel_workflow(self):
        """Setup parallel analysis workflow using AgentRearrange"""
        try:
            # Define the flow pattern for parallel analysis
            flow_pattern = f"""
            {self.fundamental_analyst.agent_name} -> {self.value_investor.agent_name}
            {self.technical_analyst.agent_name} -> {self.momentum_trader.agent_name}, {self.mean_reversion_trader.agent_name}
            {self.sentiment_analyst.agent_name} -> {self.arbitrage_finder.agent_name}
            {self.value_investor.agent_name}, {self.momentum_trader.agent_name}, {self.mean_reversion_trader.agent_name}, {self.arbitrage_finder.agent_name} -> {self.risk_manager.agent_name}
            {self.risk_manager.agent_name} -> {self.trade_executor.agent_name}
            """
            
            self.parallel_workflow = AgentRearrange(
                name="Parallel Analysis Workflow",
                description="Parallel market analysis followed by strategy evaluation",
                agents=[
                    self.fundamental_analyst.swarms_agent,
                    self.technical_analyst.swarms_agent,
                    self.sentiment_analyst.swarms_agent,
                    self.value_investor.swarms_agent,
                    self.momentum_trader.swarms_agent,
                    self.mean_reversion_trader.swarms_agent,
                    self.arbitrage_finder.swarms_agent,
                    self.risk_manager.swarms_agent,
                    self.trade_executor.swarms_agent
                ],
                flow=flow_pattern,
                max_loops=1,
                verbose=True
            )
            
            log.info("Parallel workflow setup completed")
            
        except Exception as e:
            log.error(f"Error setting up parallel workflow: {str(e)}")
    
    def _setup_consensus_workflow(self):
        """Setup consensus building workflow using MixtureOfAgents"""
        try:
            self.consensus_workflow = MixtureOfAgents(
                name="Strategy Consensus Builder",
                agents=[
                    self.value_investor.swarms_agent,
                    self.momentum_trader.swarms_agent,
                    self.mean_reversion_trader.swarms_agent,
                    self.arbitrage_finder.swarms_agent
                ],
                aggregator_agent=self.orchestrator.swarms_agent,
                aggregator_system_prompt="""
                Review all strategy recommendations and synthesize into final decision:
                
                1. Weight by agent historical performance
                2. Require minimum consensus confidence of 0.65
                3. Consider conflicting signals
                4. Apply risk management constraints
                
                Provide:
                - Final action: BUY, SELL, HOLD
                - Consensus confidence: 0.0 to 1.0
                - Position size recommendation
                - Detailed reasoning
                - Key contributing factors
                """,
                layers=2,  # Two rounds of refinement
                verbose=True
            )
            
            log.info("Consensus workflow setup completed")
            
        except Exception as e:
            log.error(f"Error setting up consensus workflow: {str(e)}")
    
    def _setup_hierarchical_workflow(self):
        """Setup hierarchical workflow with department structure"""
        # This is a custom implementation since Swarms doesn't have built-in hierarchical workflow
        self.hierarchical_workflow = {
            'name': 'Hierarchical Trading System',
            'structure': {
                'orchestrator': self.orchestrator,
                'analysis_department': {
                    'fundamental_analyst': self.fundamental_analyst,
                    'technical_analyst': self.technical_analyst,
                    'sentiment_analyst': self.sentiment_analyst
                },
                'strategy_department': {
                    'value_investor': self.value_investor,
                    'momentum_trader': self.momentum_trader,
                    'mean_reversion_trader': self.mean_reversion_trader,
                    'arbitrage_finder': self.arbitrage_finder
                },
                'operations_department': {
                    'risk_manager': self.risk_manager,
                    'trade_executor': self.trade_executor,
                    'portfolio_manager': self.portfolio_manager
                }
            },
            'decision_flow': 'bottom_up_analysis_top_down_decisions'
        }
        
        log.info("Hierarchical workflow setup completed")
    
    async def run_sequential_workflow(self, market: Market) -> Dict[str, Any]:
        """Run sequential trading workflow"""
        workflow_id = f"sequential_{market.id}_{datetime.now().strftime('%H%M%S')}"
        
        try:
            self.workflow_logger.workflow_start("Sequential Trading Pipeline", [
                "Fundamental Analysis", "Technical Analysis", "Sentiment Analysis",
                "Strategy Evaluation", "Risk Management", "Trade Execution"
            ])
            
            start_time = datetime.now()
            
            # Run sequential workflow
            result = await self.sequential_workflow.run(
                f"Analyze and trade on market: {market.question}"
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            self.workflow_logger.workflow_complete(
                "Sequential Trading Pipeline", 
                duration, 
                f"Result: {str(result)[:100]}..."
            )
            
            return {
                'workflow_id': workflow_id,
                'workflow_type': 'sequential',
                'market_id': market.id,
                'result': result,
                'duration': duration,
                'status': 'completed',
                'started_at': start_time.isoformat(),
                'completed_at': end_time.isoformat()
            }
            
        except Exception as e:
            log.error(f"Error in sequential workflow: {str(e)}")
            return {
                'workflow_id': workflow_id,
                'workflow_type': 'sequential',
                'market_id': market.id,
                'result': None,
                'error': str(e),
                'status': 'failed',
                'started_at': datetime.now().isoformat()
            }
    
    async def run_parallel_workflow(self, market: Market) -> Dict[str, Any]:
        """Run parallel analysis workflow"""
        workflow_id = f"parallel_{market.id}_{datetime.now().strftime('%H%M%S')}"
        
        try:
            self.workflow_logger.workflow_start("Parallel Analysis Workflow", [
                "Fundamental Analysis", "Technical Analysis", "Sentiment Analysis",
                "Strategy Evaluation", "Risk Management", "Trade Execution"
            ])
            
            start_time = datetime.now()
            
            # Run parallel workflow
            result = await self.parallel_workflow.run(
                f"Analyze and trade on market: {market.question}"
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            self.workflow_logger.workflow_complete(
                "Parallel Analysis Workflow",
                duration,
                f"Result: {str(result)[:100]}..."
            )
            
            return {
                'workflow_id': workflow_id,
                'workflow_type': 'parallel',
                'market_id': market.id,
                'result': result,
                'duration': duration,
                'status': 'completed',
                'started_at': start_time.isoformat(),
                'completed_at': end_time.isoformat()
            }
            
        except Exception as e:
            log.error(f"Error in parallel workflow: {str(e)}")
            return {
                'workflow_id': workflow_id,
                'workflow_type': 'parallel',
                'market_id': market.id,
                'result': None,
                'error': str(e),
                'status': 'failed',
                'started_at': datetime.now().isoformat()
            }
    
    async def run_consensus_workflow(self, market: Market, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run consensus building workflow"""
        workflow_id = f"consensus_{market.id}_{datetime.now().strftime('%H%M%S')}"
        
        try:
            self.workflow_logger.workflow_start("Consensus Building Workflow", [
                "Value Investor", "Momentum Trader", "Mean Reversion Trader", "Arbitrage Finder"
            ])
            
            start_time = datetime.now()
            
            # Prepare input for consensus workflow
            market_context = f"""
            Market: {market.question}
            Current Probability: {market.probability}
            Volume: {market.volume}
            Analysis Data: {analysis_data}
            """
            
            # Run consensus workflow
            result = await self.consensus_workflow.run(market_context)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            self.workflow_logger.workflow_complete(
                "Consensus Building Workflow",
                duration,
                f"Consensus: {str(result)[:100]}..."
            )
            
            return {
                'workflow_id': workflow_id,
                'workflow_type': 'consensus',
                'market_id': market.id,
                'result': result,
                'duration': duration,
                'status': 'completed',
                'started_at': start_time.isoformat(),
                'completed_at': end_time.isoformat()
            }
            
        except Exception as e:
            log.error(f"Error in consensus workflow: {str(e)}")
            return {
                'workflow_id': workflow_id,
                'workflow_type': 'consensus',
                'market_id': market.id,
                'result': None,
                'error': str(e),
                'status': 'failed',
                'started_at': datetime.now().isoformat()
            }
    
    async def run_hierarchical_workflow(self, market: Market) -> Dict[str, Any]:
        """Run hierarchical workflow"""
        workflow_id = f"hierarchical_{market.id}_{datetime.now().strftime('%H%M%S')}"
        
        try:
            self.workflow_logger.workflow_start("Hierarchical Trading Workflow", [
                "Analysis Department", "Strategy Department", "Operations Department"
            ])
            
            start_time = datetime.now()
            
            # Stage 1: Analysis Department (parallel)
            self.workflow_logger.stage_start("Analysis Department", "Multiple Agents")
            
            analysis_tasks = []
            analysis_agents = [
                self.fundamental_analyst,
                self.technical_analyst,
                self.sentiment_analyst
            ]
            
            for agent in analysis_agents:
                task = asyncio.create_task(agent.analyze_market(market))
                analysis_tasks.append((agent.agent_name, task))
            
            analysis_results = {}
            for agent_name, task in analysis_tasks:
                try:
                    result = await task
                    analysis_results[agent_name] = result
                    self.workflow_logger.stage_complete("Analysis Department", agent_name, 0)
                except Exception as e:
                    log.error(f"Analysis failed for {agent_name}: {str(e)}")
                    analysis_results[agent_name] = None
            
            # Stage 2: Strategy Department (parallel)
            self.workflow_logger.stage_start("Strategy Department", "Multiple Agents")
            
            strategy_tasks = []
            strategy_agents = [
                self.value_investor,
                self.momentum_trader,
                self.mean_reversion_trader,
                self.arbitrage_finder
            ]
            
            # Prepare analysis data
            analysis_data = {
                'fundamental_score': market.fundamental_score or 0.5,
                'technical_score': market.technical_score or 0.5,
                'sentiment_score': market.sentiment_score or 0.5
            }
            
            for agent in strategy_agents:
                task = asyncio.create_task(agent.analyze_market(market))
                strategy_tasks.append((agent.agent_name, task))
            
            strategy_results = {}
            for agent_name, task in strategy_tasks:
                try:
                    result = await task
                    strategy_results[agent_name] = result
                    self.workflow_logger.stage_complete("Strategy Department", agent_name, 0)
                except Exception as e:
                    log.error(f"Strategy evaluation failed for {agent_name}: {str(e)}")
                    strategy_results[agent_name] = None
            
            # Stage 3: Orchestrator decision
            self.workflow_logger.stage_start("Decision Making", "Orchestrator")
            
            # Build consensus from strategy results
            strategy_signals = [s for s in strategy_results.values() if s is not None]
            if strategy_signals:
                consensus = await self.coordinator.build_consensus(market.id, strategy_signals)
            else:
                consensus = ConsensusDecision(
                    market_id=market.id,
                    final_action="HOLD",
                    consensus_confidence=0.0,
                    participating_agents=[],
                    agent_signals=[],
                    orchestrator_reasoning="No strategy signals available",
                    risk_approved=False,
                    position_size=0.0
                )
            
            self.workflow_logger.stage_complete("Decision Making", "Orchestrator", 0)
            
            # Stage 4: Operations Department
            self.workflow_logger.stage_start("Operations", "Risk Manager")
            
            # Risk validation
            risk_decision = {
                'market_id': market.id,
                'action': consensus.final_action,
                'position_size': consensus.position_size,
                'confidence': consensus.consensus_confidence,
                'agent_signals': [
                    {
                        'agent': signal.agent_name,
                        'action': signal.action.value,
                        'confidence': signal.confidence
                    }
                    for signal in consensus.agent_signals
                ]
            }
            
            risk_validation = await self.risk_manager.validate_trading_decision(risk_decision)
            consensus.risk_approved = risk_validation.get('approved', False)
            
            self.workflow_logger.stage_complete("Operations", "Risk Manager", 0)
            
            # Stage 5: Execution (if approved)
            if consensus.risk_approved and consensus.final_action != "HOLD":
                self.workflow_logger.stage_start("Execution", "Trade Executor")
                
                execution_result = await self.trade_executor.execute_trade_decision({
                    'market_id': market.id,
                    'action': consensus.final_action,
                    'outcome': "YES" if consensus.final_action == "BUY" else "NO",
                    'position_size': risk_validation.get('adjusted_position_size', consensus.position_size),
                    'confidence': consensus.consensus_confidence
                })
                
                self.workflow_logger.stage_complete("Operations", "Trade Executor", 0)
            else:
                execution_result = {'status': 'SKIPPED', 'reason': 'Not approved or HOLD action'}
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            self.workflow_logger.workflow_complete(
                "Hierarchical Trading Workflow",
                duration,
                f"Final action: {consensus.final_action}, Risk approved: {consensus.risk_approved}"
            )
            
            return {
                'workflow_id': workflow_id,
                'workflow_type': 'hierarchical',
                'market_id': market.id,
                'analysis_results': analysis_results,
                'strategy_results': strategy_results,
                'consensus': consensus,
                'risk_validation': risk_validation,
                'execution_result': execution_result,
                'duration': duration,
                'status': 'completed',
                'started_at': start_time.isoformat(),
                'completed_at': end_time.isoformat()
            }
            
        except Exception as e:
            log.error(f"Error in hierarchical workflow: {str(e)}")
            return {
                'workflow_id': workflow_id,
                'workflow_type': 'hierarchical',
                'market_id': market.id,
                'result': None,
                'error': str(e),
                'status': 'failed',
                'started_at': datetime.now().isoformat()
            }
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get status of all workflows"""
        return {
            'sequential_workflow': {
                'available': self.sequential_workflow is not None,
                'agents': len(self.sequential_workflow.agents) if self.sequential_workflow else 0
            },
            'parallel_workflow': {
                'available': self.parallel_workflow is not None,
                'agents': len(self.parallel_workflow.agents) if self.parallel_workflow else 0
            },
            'consensus_workflow': {
                'available': self.consensus_workflow is not None,
                'agents': len(self.consensus_workflow.agents) if self.consensus_workflow else 0,
                'layers': self.consensus_workflow.layers if self.consensus_workflow else 0
            },
            'hierarchical_workflow': {
                'available': self.hierarchical_workflow is not None,
                'departments': len(self.hierarchical_workflow['structure']) if self.hierarchical_workflow else 0
            }
        }
    
    async def run_optimal_workflow(self, market: Market, workflow_type: str = "auto") -> Dict[str, Any]:
        """Run the optimal workflow based on market conditions and configuration"""
        
        if workflow_type == "auto":
            # Choose workflow based on configuration
            configured_workflow = config.get('workflows.main_workflow', 'hierarchical')
            workflow_type = configured_workflow
        
        log.info(f"Running {workflow_type} workflow for market {market.id}")
        
        if workflow_type == "sequential":
            return await self.run_sequential_workflow(market)
        elif workflow_type == "parallel":
            return await self.run_parallel_workflow(market)
        elif workflow_type == "consensus":
            # Need analysis data first
            analysis_data = {
                'fundamental_score': 0.5,
                'technical_score': 0.5,
                'sentiment_score': 0.5
            }
            return await self.run_consensus_workflow(market, analysis_data)
        elif workflow_type == "hierarchical":
            return await self.run_hierarchical_workflow(market)
        else:
            raise ValueError(f"Unknown workflow type: {workflow_type}")


class WorkflowManager:
    """High-level workflow management"""
    
    def __init__(self, coordinator: AgentCoordinator):
        self.coordinator = coordinator
        self.workflows = ManifoldTradingWorkflows(coordinator)
        self.active_workflow_runs: Dict[str, Dict[str, Any]] = {}
        
    async def execute_market_analysis(self, market: Market, workflow_type: str = "auto") -> Dict[str, Any]:
        """Execute market analysis using optimal workflow"""
        run_id = f"run_{market.id}_{datetime.now().strftime('%H%M%S')}"
        
        try:
            # Start workflow run
            self.active_workflow_runs[run_id] = {
                'run_id': run_id,
                'market_id': market.id,
                'workflow_type': workflow_type,
                'started_at': datetime.now(),
                'status': 'running'
            }
            
            # Execute workflow
            result = await self.workflows.run_optimal_workflow(market, workflow_type)
            
            # Update run status
            self.active_workflow_runs[run_id].update({
                'status': 'completed',
                'result': result,
                'completed_at': datetime.now()
            })
            
            return result
            
        except Exception as e:
            log.error(f"Error executing workflow for {market.id}: {str(e)}")
            
            # Update run status
            if run_id in self.active_workflow_runs:
                self.active_workflow_runs[run_id].update({
                    'status': 'failed',
                    'error': str(e),
                    'completed_at': datetime.now()
                })
            
            return None
    
    def get_active_runs(self) -> Dict[str, Dict[str, Any]]:
        """Get all active workflow runs"""
        return self.active_workflow_runs.copy()
    
    def get_workflow_statistics(self) -> Dict[str, Any]:
        """Get workflow execution statistics"""
        completed_runs = [run for run in self.active_workflow_runs.values() if run['status'] == 'completed']
        failed_runs = [run for run in self.active_workflow_runs.values() if run['status'] == 'failed']
        
        workflow_types = {}
        for run in completed_runs:
            workflow_type = run.get('workflow_type', 'unknown')
            if workflow_type not in workflow_types:
                workflow_types[workflow_type] = {'completed': 0, 'failed': 0}
            workflow_types[workflow_type]['completed'] += 1
        
        for run in failed_runs:
            workflow_type = run.get('workflow_type', 'unknown')
            if workflow_type not in workflow_types:
                workflow_types[workflow_type] = {'completed': 0, 'failed': 0}
            workflow_types[workflow_type]['failed'] += 1
        
        return {
            'total_runs': len(self.active_workflow_runs),
            'completed_runs': len(completed_runs),
            'failed_runs': len(failed_runs),
            'success_rate': len(completed_runs) / len(self.active_workflow_runs) if self.active_workflow_runs else 0,
            'workflow_types': workflow_types,
            'workflow_status': self.workflows.get_workflow_status()
        }