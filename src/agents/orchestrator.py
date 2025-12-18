"""
Trading Orchestrator Agent
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio

from src.agents.base_agent import BaseManifoldAgent
from src.models import (
    AgentSignal, ConsensusDecision, Market, TradingAction, 
    AgentRole, AgentMessage
)
from src.swarms_core.communication import AgentCoordinator
from src.utils.config import config
from src.utils.logger import get_logger

log = get_logger(__name__)


class TradingOrchestrator(BaseManifoldAgent):
    """Orchestrates the entire trading system, coordinates agents, makes final decisions"""
    
    def __init__(self, coordinator: AgentCoordinator):
        self.coordinator = coordinator
        super().__init__("Trading Orchestrator", AgentRole.ORCHESTRATOR)
        
        # Orchestrator-specific state
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        self.decision_history: List[ConsensusDecision] = []
        self.agent_performance_weights: Dict[str, float] = {}
        
        # Configuration
        self.min_consensus_confidence = config.get('trading.min_consensus_confidence', 0.65)
        self.max_concurrent_workflows = config.get('agents.max_concurrent_workflows', 5)
        
    def get_default_prompt(self) -> str:
        return config.get_agent_prompt('orchestrator') or """
        You are the orchestrator of a sophisticated multi-agent trading system for prediction markets.
        
        Your responsibilities:
        1. Coordinate market analysts, strategists, and execution agents
        2. Synthesize recommendations from all agents
        3. Make final trading decisions based on consensus
        4. Ensure agent communication and collaboration
        5. Monitor overall system performance
        
        Decision criteria:
        - Require minimum consensus confidence of 0.65
        - Weight agent inputs by historical performance
        - Prioritize risk management and capital preservation
        - Consider market conditions and liquidity
        
        Always provide clear reasoning for your decisions and maintain trading discipline.
        """
    
    async def analyze_market(self, market: Market) -> AgentSignal:
        """Orchestrator doesn't generate direct signals, but coordinates the analysis process"""
        # This method is called for compatibility, but orchestrator works differently
        return AgentSignal(
            agent_name=self.agent_name,
            agent_role=self.agent_role,
            market_id=market.id,
            action=TradingAction.HOLD,
            outcome="",
            confidence=0.0,
            reasoning="Orchestrator coordinates analysis rather than generating signals"
        )
    
    async def orchestrate_market_analysis(self, market: Market) -> Optional[ConsensusDecision]:
        """Orchestrate complete market analysis workflow"""
        workflow_id = f"analysis_{market.id}_{datetime.now().strftime('%H%M%S')}"
        
        try:
            # Check workflow limits
            if len(self.active_workflows) >= self.max_concurrent_workflows:
                log.warning(f"Maximum concurrent workflows ({self.max_concurrent_workflows}) reached")
                return None
            
            # Initialize workflow
            workflow = {
                'id': workflow_id,
                'market_id': market.id,
                'stage': 'initialization',
                'started_at': datetime.now(),
                'status': 'running',
                'market': market,
                'results': {}
            }
            
            self.active_workflows[workflow_id] = workflow
            
            log.info(f"Starting orchestrated analysis for market {market.id}")
            
            # Stage 1: Market Analysis
            workflow['stage'] = 'market_analysis'
            analysis_results = await self._coordinate_market_analysis(market)
            workflow['results']['analysis'] = analysis_results
            
            # Stage 2: Strategy Evaluation
            workflow['stage'] = 'strategy_evaluation'
            strategy_results = await self._coordinate_strategy_evaluation(market, analysis_results)
            workflow['results']['strategy'] = strategy_results
            
            # Stage 3: Consensus Building
            workflow['stage'] = 'consensus_building'
            consensus_decision = await self._build_consensus(market, analysis_results, strategy_results)
            workflow['results']['consensus'] = consensus_decision
            
            # Stage 4: Risk Validation
            workflow['stage'] = 'risk_validation'
            risk_approval = await self._validate_with_risk_manager(consensus_decision)
            workflow['results']['risk'] = risk_approval
            
            # Finalize decision
            if risk_approval.get('approved', False):
                consensus_decision.risk_approved = True
                consensus_decision.position_size = risk_approval.get('adjusted_position_size', 
                                                                    consensus_decision.position_size)
                workflow['status'] = 'completed'
            else:
                consensus_decision.risk_approved = False
                workflow['status'] = 'rejected_by_risk'
                log.info(f"Decision rejected by risk manager: {risk_approval.get('reason', 'Unknown')}")
            
            # Record decision
            self.decision_history.append(consensus_decision)
            
            # Clean up workflow
            del self.active_workflows[workflow_id]
            
            log.info(f"Completed orchestrated analysis for {market.id}: {consensus_decision.final_action}")
            
            return consensus_decision
            
        except Exception as e:
            log.error(f"Error in orchestrated analysis for {market.id}: {str(e)}")
            if workflow_id in self.active_workflows:
                self.active_workflows[workflow_id]['status'] = 'error'
                self.active_workflows[workflow_id]['error'] = str(e)
            return None
    
    async def _coordinate_market_analysis(self, market: Market) -> Dict[str, AgentSignal]:
        """Coordinate parallel market analysis by analyst agents"""
        analyst_agents = ['Fundamental Analyst', 'Technical Analyst', 'Sentiment Analyst']
        
        # Send analysis requests to all analyst agents
        tasks = []
        for agent_name in analyst_agents:
            if agent_name in self.coordinator.active_agents:
                agent = self.coordinator.active_agents[agent_name]
                task = asyncio.create_task(agent.analyze_market(market))
                tasks.append((agent_name, task))
        
        # Wait for all analyses to complete
        analysis_results = {}
        for agent_name, task in tasks:
            try:
                signal = await task
                analysis_results[agent_name] = signal
                log.info(f"Analysis complete: {agent_name} for {market.id}")
            except Exception as e:
                log.error(f"Analysis failed for {agent_name}: {str(e)}")
                # Create error signal
                analysis_results[agent_name] = AgentSignal(
                    agent_name=agent_name,
                    agent_role=AgentRole.ANALYST,
                    market_id=market.id,
                    action=TradingAction.HOLD,
                    outcome="",
                    confidence=0.0,
                    reasoning=f"Analysis failed: {str(e)}"
                )
        
        # Update market with analysis scores
        for agent_name, signal in analysis_results.items():
            if 'Fundamental' in agent_name:
                market.fundamental_score = signal.confidence
            elif 'Technical' in agent_name:
                market.technical_score = signal.confidence
            elif 'Sentiment' in agent_name:
                market.sentiment_score = signal.confidence
            
            market.agent_analyses[agent_name] = {
                'score': signal.confidence,
                'reasoning': signal.reasoning,
                'timestamp': signal.timestamp
            }
        
        return analysis_results
    
    async def _coordinate_strategy_evaluation(self, market: Market, 
                                           analysis_results: Dict[str, AgentSignal]) -> Dict[str, AgentSignal]:
        """Coordinate strategy evaluation by strategy agents"""
        strategy_agents = ['Value Investor', 'Momentum Trader', 'Mean Reversion Trader', 'Arbitrage Finder']
        
        # Prepare analysis data for strategy agents
        analysis_data = {
            'fundamental_score': market.fundamental_score or 0.5,
            'technical_score': market.technical_score or 0.5,
            'sentiment_score': market.sentiment_score or 0.5
        }
        
        # Send evaluation requests to all strategy agents
        tasks = []
        for agent_name in strategy_agents:
            if agent_name in self.coordinator.active_agents:
                agent = self.coordinator.active_agents[agent_name]
                task = asyncio.create_task(agent.analyze_market(market))
                tasks.append((agent_name, task))
        
        # Wait for all strategy evaluations to complete
        strategy_results = {}
        for agent_name, task in tasks:
            try:
                signal = await task
                strategy_results[agent_name] = signal
                log.info(f"Strategy evaluation complete: {agent_name} for {market.id}")
            except Exception as e:
                log.error(f"Strategy evaluation failed for {agent_name}: {str(e)}")
                # Create error signal
                strategy_results[agent_name] = AgentSignal(
                    agent_name=agent_name,
                    agent_role=AgentRole.STRATEGIST,
                    market_id=market.id,
                    action=TradingAction.HOLD,
                    outcome="",
                    confidence=0.0,
                    reasoning=f"Strategy evaluation failed: {str(e)}"
                )
        
        return strategy_results
    
    async def _build_consensus(self, market: Market, 
                             analysis_results: Dict[str, AgentSignal],
                             strategy_results: Dict[str, AgentSignal]) -> ConsensusDecision:
        """Build consensus from all agent signals"""
        # Collect all trading signals (exclude HOLD signals from final consensus)
        all_signals = list(strategy_results.values())
        trading_signals = [s for s in all_signals if s.action != TradingAction.HOLD]
        
        if not trading_signals:
            # All agents recommend HOLD
            return ConsensusDecision(
                market_id=market.id,
                final_action="HOLD",
                consensus_confidence=1.0,
                participating_agents=list(all_signals),
                agent_signals=all_signals,
                orchestrator_reasoning="All strategy agents recommend HOLD",
                risk_approved=False,
                position_size=0.0
            )
        
        # Weight signals by agent performance
        weighted_signals = self._weight_signals_by_performance(trading_signals)
        
        # Calculate consensus
        consensus = await self.coordinator.consensus_builder.build_consensus(market.id, weighted_signals)
        
        # Add orchestrator reasoning
        consensus.orchestrator_reasoning = self._generate_consensus_reasoning(
            market, analysis_results, strategy_results, consensus
        )
        
        return consensus
    
    def _weight_signals_by_performance(self, signals: List[AgentSignal]) -> List[AgentSignal]:
        """Weight signals by agent historical performance"""
        weighted_signals = []
        
        for signal in signals:
            # Get agent performance weight
            base_weight = self.agent_performance_weights.get(signal.agent_name, 1.0)
            
            # Adjust by signal confidence
            adjusted_confidence = signal.confidence * base_weight
            
            # Create weighted signal
            weighted_signal = AgentSignal(
                agent_name=signal.agent_name,
                agent_role=signal.agent_role,
                market_id=signal.market_id,
                action=signal.action,
                outcome=signal.outcome,
                confidence=adjusted_confidence,
                reasoning=signal.reasoning,
                estimated_probability=signal.estimated_probability,
                position_size=signal.position_size,
                timestamp=signal.timestamp,
                supporting_evidence=signal.supporting_evidence
            )
            weighted_signals.append(weighted_signal)
        
        return weighted_signals
    
    def _generate_consensus_reasoning(self, market: Market,
                                    analysis_results: Dict[str, AgentSignal],
                                    strategy_results: Dict[str, AgentSignal],
                                    consensus: ConsensusDecision) -> str:
        """Generate detailed reasoning for consensus decision"""
        reasoning_parts = []
        
        # Market context
        reasoning_parts.append(f"Market: {market.question[:100]}...")
        reasoning_parts.append(f"Current probability: {market.probability:.2f}")
        
        # Analysis summary
        analysis_scores = []
        for agent_name, signal in analysis_results.items():
            analysis_scores.append(f"{agent_name}: {signal.confidence:.2f}")
        reasoning_parts.append(f"Analysis: {', '.join(analysis_scores)}")
        
        # Strategy summary
        strategy_votes = {}
        for agent_name, signal in strategy_results.items():
            if signal.action != TradingAction.HOLD:
                action_key = signal.action.value
                if action_key not in strategy_votes:
                    strategy_votes[action_key] = []
                strategy_votes[action_key].append(agent_name)
        
        if strategy_votes:
            vote_summary = []
            for action, agents in strategy_votes.items():
                vote_summary.append(f"{action}: {len(agents)} agents")
            reasoning_parts.append(f"Strategy votes: {', '.join(vote_summary)}")
        
        # Consensus details
        reasoning_parts.append(f"Consensus: {consensus.final_action} ({consensus.consensus_confidence:.2f} confidence)")
        
        return " | ".join(reasoning_parts)
    
    async def _validate_with_risk_manager(self, decision: ConsensusDecision) -> Dict[str, Any]:
        """Validate decision with risk manager"""
        if 'Risk Manager' not in self.coordinator.active_agents:
            return {'approved': False, 'reason': 'Risk Manager not available'}
        
        risk_manager = self.coordinator.active_agents['Risk Manager']
        
        # Send decision to risk manager for validation
        message_content = {
            'decision': {
                'market_id': decision.market_id,
                'action': decision.final_action,
                'position_size': decision.position_size,
                'confidence': decision.consensus_confidence,
                'agent_signals': [
                    {
                        'agent': signal.agent_name,
                        'action': signal.action.value,
                        'confidence': signal.confidence
                    }
                    for signal in decision.agent_signals
                ]
            }
        }
        
        try:
            # Send message to risk manager
            await self.coordinator.send_message(
                self.agent_name, 
                'Risk Manager',
                message_content,
                'REQUEST',
                priority=1  # High priority for risk validation
            )
            
            # Wait for response (simplified - in real implementation would use proper async messaging)
            await asyncio.sleep(1)
            
            # Get risk manager's response
            messages = self.coordinator.message_broker.get_messages('Risk Manager')
            
            for message in messages:
                if message.sender_agent == 'Risk Manager' and message.receiver_agent == self.agent_name:
                    return message.content
            
            # Default response if no reply received
            return {'approved': False, 'reason': 'No response from Risk Manager'}
            
        except Exception as e:
            log.error(f"Error validating with risk manager: {str(e)}")
            return {'approved': False, 'reason': f'Error: {str(e)}'}
    
    def update_agent_performance_weights(self, performance_data: Dict[str, Dict[str, float]]):
        """Update agent performance weights for consensus building"""
        for agent_name, metrics in performance_data.items():
            # Use accuracy rate as primary weight
            accuracy = metrics.get('accuracy_rate', 0.5)
            pnl_contribution = metrics.get('total_pnl_contribution', 0)
            
            # Calculate composite weight
            weight = accuracy * 0.7 + min(pnl_contribution / 100, 1.0) * 0.3
            weight = max(0.1, min(2.0, weight))  # Clamp between 0.1 and 2.0
            
            self.agent_performance_weights[agent_name] = weight
            
            log.info(f"Updated performance weight for {agent_name}: {weight:.2f}")
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get status of all active workflows"""
        return {
            'active_workflows': len(self.active_workflows),
            'workflow_details': {
                wid: {
                    'market_id': workflow['market_id'],
                    'stage': workflow['stage'],
                    'status': workflow['status'],
                    'duration': (datetime.now() - workflow['started_at']).total_seconds()
                }
                for wid, workflow in self.active_workflows.items()
            },
            'total_decisions': len(self.decision_history),
            'agent_weights': self.agent_performance_weights
        }
    
    def get_decision_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent decision history"""
        recent_decisions = self.decision_history[-limit:]
        
        return [
            {
                'market_id': decision.market_id,
                'action': decision.final_action,
                'confidence': decision.consensus_confidence,
                'position_size': decision.position_size,
                'risk_approved': decision.risk_approved,
                'timestamp': decision.timestamp.isoformat(),
                'participating_agents': decision.participating_agents
            }
            for decision in recent_decisions
        ]
    
    async def handle_system_alert(self, alert_type: str, details: Dict[str, Any]):
        """Handle system-level alerts"""
        log.warning(f"System alert: {alert_type} - {details}")
        
        if alert_type == 'high_loss':
            # Implement emergency stop logic
            await self._emergency_stop(details)
        elif alert_type == 'agent_failure':
            # Handle agent failure
            await self._handle_agent_failure(details)
        elif alert_type == 'market_anomaly':
            # Handle unusual market conditions
            await self._handle_market_anomaly(details)
    
    async def _emergency_stop(self, details: Dict[str, Any]):
        """Implement emergency trading stop"""
        log.critical(f"Emergency stop triggered: {details}")
        
        # Notify all agents to stop trading
        await self.coordinator.broadcast_message(
            self.agent_name,
            {'action': 'emergency_stop', 'reason': details.get('reason', 'Unknown')},
            topic='emergency'
        )
    
    async def _handle_agent_failure(self, details: Dict[str, Any]):
        """Handle agent failure"""
        failed_agent = details.get('agent_name')
        log.error(f"Agent failure: {failed_agent}")
        
        # Remove failed agent from active workflows
        for workflow_id, workflow in self.active_workflows.items():
            if workflow.get('stage') == 'running':
                # Mark workflow as affected
                workflow['status'] = 'agent_failure'
                workflow['failed_agent'] = failed_agent
    
    async def _handle_market_anomaly(self, details: Dict[str, Any]):
        """Handle market anomaly"""
        market_id = details.get('market_id')
        log.warning(f"Market anomaly detected: {market_id}")
        
        # Cancel any active workflows for this market
        affected_workflows = [
            wid for wid, workflow in self.active_workflows.items()
            if workflow.get('market_id') == market_id
        ]
        
        for workflow_id in affected_workflows:
            self.active_workflows[workflow_id]['status'] = 'market_anomaly'
            log.info(f"Cancelled workflow {workflow_id} due to market anomaly")