"""
Agent communication and coordination layer
"""
import asyncio
import json
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass, asdict
from collections import defaultdict, deque

from src.models import AgentMessage, AgentSignal, ConsensusDecision
from src.utils.logger import get_logger
from src.utils.config import config

log = get_logger(__name__)


@dataclass
class MessageBroker:
    """Simple message broker for agent communication"""
    
    def __init__(self):
        self.message_queues: Dict[str, deque] = defaultdict(deque)
        self.subscribers: Dict[str, List[str]] = defaultdict(list)
        self.message_history: List[AgentMessage] = []
        self.max_history = 1000
        
    def subscribe(self, agent_name: str, topic: str = "*"):
        """Subscribe agent to a topic"""
        self.subscribers[topic].append(agent_name)
        log.info(f"Agent {agent_name} subscribed to topic {topic}")
    
    def unsubscribe(self, agent_name: str, topic: str = "*"):
        """Unsubscribe agent from a topic"""
        if topic in self.subscribers and agent_name in self.subscribers[topic]:
            self.subscribers[topic].remove(agent_name)
            log.info(f"Agent {agent_name} unsubscribed from topic {topic}")
    
    async def publish(self, message: AgentMessage, topic: str = "direct"):
        """Publish message to subscribers"""
        # Add to message history
        self.message_history.append(message)
        if len(self.message_history) > self.max_history:
            self.message_history.pop(0)
        
        # Direct message
        if topic == "direct":
            if message.receiver_agent in self.message_queues:
                self.message_queues[message.receiver_agent].append(message)
                log.debug(f"Message delivered to {message.receiver_agent}")
        else:
            # Broadcast to topic subscribers
            subscribers = self.subscribers.get(topic, []) + self.subscribers.get("*", [])
            for subscriber in subscribers:
                if subscriber != message.sender_agent:  # Don't send to self
                    self.message_queues[subscriber].append(message)
        
        log.debug(f"Message published: {message.sender_agent} -> {message.receiver_agent} ({message.message_type})")
    
    def get_messages(self, agent_name: str, limit: Optional[int] = None) -> List[AgentMessage]:
        """Get messages for an agent"""
        messages = list(self.message_queues[agent_name])
        if limit:
            messages = messages[-limit:]
        
        # Clear retrieved messages
        self.message_queues[agent_name].clear()
        return messages
    
    def get_message_history(self, agent_name: Optional[str] = None, 
                          message_type: Optional[str] = None,
                          limit: int = 100) -> List[AgentMessage]:
        """Get message history with filters"""
        filtered = self.message_history
        
        if agent_name:
            filtered = [m for m in filtered if m.sender_agent == agent_name or m.receiver_agent == agent_name]
        
        if message_type:
            filtered = [m for m in filtered if m.message_type == message_type]
        
        return filtered[-limit:]


class AgentCoordinator:
    """Coordinates agent interactions and workflows"""
    
    def __init__(self):
        self.message_broker = MessageBroker()
        self.active_agents: Dict[str, Any] = {}
        self.workflow_registry: Dict[str, Any] = {}
        self.consensus_builder = ConsensusBuilder()
        self.agent_performance: Dict[str, Dict[str, Any]] = {}
        
    def register_agent(self, agent: Any):
        """Register an agent with the coordinator"""
        self.active_agents[agent.agent_name] = agent
        self.message_broker.subscribe(agent.agent_name)
        self.agent_performance[agent.agent_name] = {
            "messages_sent": 0,
            "messages_received": 0,
            "last_activity": datetime.now()
        }
        log.info(f"Agent {agent.agent_name} registered")
    
    def unregister_agent(self, agent_name: str):
        """Unregister an agent"""
        if agent_name in self.active_agents:
            del self.active_agents[agent_name]
            self.message_broker.unsubscribe(agent_name)
            log.info(f"Agent {agent_name} unregistered")
    
    async def send_message(self, sender: str, receiver: str, content: Dict[str, Any], 
                          message_type: str = "REQUEST", priority: int = 3):
        """Send message between agents"""
        message = AgentMessage(
            sender_agent=sender,
            receiver_agent=receiver,
            message_type=message_type,
            content=content,
            priority=priority
        )
        
        await self.message_broker.publish(message)
        
        # Update metrics
        if sender in self.agent_performance:
            self.agent_performance[sender]["messages_sent"] += 1
            self.agent_performance[sender]["last_activity"] = datetime.now()
    
    async def broadcast_message(self, sender: str, content: Dict[str, Any], 
                               topic: str = "broadcast", priority: int = 3):
        """Broadcast message to all agents"""
        message = AgentMessage(
            sender_agent=sender,
            receiver_agent="ALL",
            message_type="BROADCAST",
            content=content,
            priority=priority
        )
        
        await self.message_broker.publish(message, topic)
        
        # Update metrics
        if sender in self.agent_performance:
            self.agent_performance[sender]["messages_sent"] += len(self.active_agents) - 1
            self.agent_performance[sender]["last_activity"] = datetime.now()
    
    async def process_agent_messages(self, agent_name: str):
        """Process all pending messages for an agent"""
        if agent_name not in self.active_agents:
            return
        
        agent = self.active_agents[agent_name]
        messages = self.message_broker.get_messages(agent_name)
        
        for message in messages:
            try:
                response = await agent.process_message(message)
                if response:
                    await self.message_broker.publish(response)
                
                # Update metrics
                if agent_name in self.agent_performance:
                    self.agent_performance[agent_name]["messages_received"] += 1
                    
            except Exception as e:
                log.error(f"Error processing message for {agent_name}: {str(e)}")
    
    async def run_parallel_analysis(self, market_id: str, analyst_agents: List[str]) -> Dict[str, Any]:
        """Run parallel analysis with multiple agents"""
        tasks = []
        results = {}
        
        for agent_name in analyst_agents:
            if agent_name in self.active_agents:
                agent = self.active_agents[agent_name]
                # In real implementation, would pass market object
                task = asyncio.create_task(agent.analyze_market(None))  # Placeholder
                tasks.append((agent_name, task))
        
        # Wait for all analyses to complete
        for agent_name, task in tasks:
            try:
                result = await task
                results[agent_name] = result
                log.info(f"Analysis complete: {agent_name}")
            except Exception as e:
                log.error(f"Analysis failed for {agent_name}: {str(e)}")
                results[agent_name] = None
        
        return results
    
    async def build_consensus(self, market_id: str, signals: List[AgentSignal]) -> ConsensusDecision:
        """Build consensus from multiple agent signals"""
        return await self.consensus_builder.build_consensus(market_id, signals)
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all registered agents"""
        status = {}
        for agent_name, agent in self.active_agents.items():
            status[agent_name] = agent.get_status()
            status[agent_name]["performance"] = self.agent_performance.get(agent_name, {})
        return status
    
    def get_message_stats(self) -> Dict[str, Any]:
        """Get message statistics"""
        total_messages = len(self.message_broker.message_history)
        message_types = defaultdict(int)
        
        for message in self.message_broker.message_history:
            message_types[message.message_type] += 1
        
        return {
            "total_messages": total_messages,
            "message_types": dict(message_types),
            "queue_sizes": {name: len(queue) for name, queue in self.message_broker.message_queues.items()}
        }


class ConsensusBuilder:
    """Builds consensus from multiple agent signals"""
    
    def __init__(self):
        self.min_consensus_confidence = config.get('trading.min_consensus_confidence', 0.65)
        self.min_agent_agreement = config.get('agents.min_agent_agreement', 2)
        
    async def build_consensus(self, market_id: str, signals: List[AgentSignal]) -> ConsensusDecision:
        """Build consensus decision from agent signals"""
        if not signals:
            raise ValueError("No signals provided for consensus building")
        
        # Filter out HOLD signals for consensus calculation
        trading_signals = [s for s in signals if s.action.value in ["BUY", "SELL"]]
        
        if not trading_signals:
            # All signals are HOLD
            return ConsensusDecision(
                market_id=market_id,
                final_action="HOLD",
                consensus_confidence=1.0,
                participating_agents=[s.agent_name for s in signals],
                agent_signals=signals,
                orchestrator_reasoning="All agents recommend HOLD",
                risk_approved=True,
                position_size=0.0
            )
        
        # Count votes
        votes = defaultdict(list)
        for signal in trading_signals:
            votes[signal.action].append(signal)
        
        # Determine winning action
        winning_action = max(votes.keys(), key=lambda x: len(votes[x]))
        winning_signals = votes[winning_action]
        
        # Calculate consensus confidence
        agreement_ratio = len(winning_signals) / len(trading_signals)
        avg_confidence = sum(s.confidence for s in winning_signals) / len(winning_signals)
        consensus_confidence = agreement_ratio * avg_confidence
        
        # Calculate position size based on confidence and agent recommendations
        avg_position_size = sum(s.position_size or 0 for s in winning_signals) / len(winning_signals)
        final_position_size = avg_position_size * consensus_confidence
        
        # Generate reasoning
        participating_agents = [s.agent_name for s in winning_signals]
        reasoning = f"Consensus: {len(winning_signals)}/{len(trading_signals)} agents agree on {winning_action.value} with {consensus_confidence:.2f} confidence"
        
        return ConsensusDecision(
            market_id=market_id,
            final_action=winning_action.value,
            consensus_confidence=consensus_confidence,
            participating_agents=participating_agents,
            agent_signals=signals,
            orchestrator_reasoning=reasoning,
            risk_approved=consensus_confidence >= self.min_consensus_confidence,
            position_size=final_position_size
        )
    
    def calculate_agreement_score(self, signals: List[AgentSignal]) -> float:
        """Calculate agreement score among signals"""
        if len(signals) < 2:
            return 1.0
        
        trading_signals = [s for s in signals if s.action.value in ["BUY", "SELL"]]
        if not trading_signals:
            return 1.0
        
        votes = defaultdict(int)
        for signal in trading_signals:
            votes[signal.action] += 1
        
        # Calculate entropy-based agreement
        total_votes = len(trading_signals)
        max_votes = max(votes.values())
        agreement = max_votes / total_votes
        
        return agreement
    
    def weight_signals_by_performance(self, signals: List[AgentSignal], 
                                   agent_performance: Dict[str, Dict[str, Any]]) -> List[AgentSignal]:
        """Weight signals by agent historical performance"""
        weighted_signals = []
        
        for signal in signals:
            performance = agent_performance.get(signal.agent_name, {})
            accuracy = performance.get("accuracy_rate", 0.5)
            
            # Create weighted signal
            weighted_signal = AgentSignal(
                agent_name=signal.agent_name,
                agent_role=signal.agent_role,
                market_id=signal.market_id,
                action=signal.action,
                outcome=signal.outcome,
                confidence=signal.confidence * accuracy,  # Weight by accuracy
                reasoning=signal.reasoning,
                estimated_probability=signal.estimated_probability,
                position_size=signal.position_size,
                timestamp=signal.timestamp,
                supporting_evidence=signal.supporting_evidence
            )
            weighted_signals.append(weighted_signal)
        
        return weighted_signals


class WorkflowManager:
    """Manages agent workflows and coordination"""
    
    def __init__(self, coordinator: AgentCoordinator):
        self.coordinator = coordinator
        self.active_workflows: Dict[str, Any] = {}
        self.workflow_templates: Dict[str, Dict[str, Any]] = {}
        
    def register_workflow_template(self, name: str, template: Dict[str, Any]):
        """Register a workflow template"""
        self.workflow_templates[name] = template
        log.info(f"Workflow template '{name}' registered")
    
    async def start_workflow(self, workflow_name: str, market_id: str, **kwargs) -> str:
        """Start a new workflow instance"""
        if workflow_name not in self.workflow_templates:
            raise ValueError(f"Workflow template '{workflow_name}' not found")
        
        template = self.workflow_templates[workflow_name]
        workflow_id = f"{workflow_name}_{market_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create workflow instance
        workflow = {
            "id": workflow_id,
            "name": workflow_name,
            "market_id": market_id,
            "template": template,
            "status": "RUNNING",
            "started_at": datetime.now(),
            "current_stage": 0,
            "results": {},
            "kwargs": kwargs
        }
        
        self.active_workflows[workflow_id] = workflow
        
        # Start workflow execution
        asyncio.create_task(self._execute_workflow(workflow_id))
        
        log.info(f"Workflow started: {workflow_id}")
        return workflow_id
    
    async def _execute_workflow(self, workflow_id: str):
        """Execute a workflow"""
        workflow = self.active_workflows.get(workflow_id)
        if not workflow:
            return
        
        template = workflow["template"]
        stages = template.get("stages", [])
        
        try:
            for i, stage in enumerate(stages):
                workflow["current_stage"] = i
                
                # Execute stage
                stage_result = await self._execute_stage(stage, workflow)
                workflow["results"][stage["name"]] = stage_result
                
                # Check if stage failed
                if not stage_result.get("success", True):
                    workflow["status"] = "FAILED"
                    workflow["error"] = stage_result.get("error", "Stage failed")
                    break
            
            if workflow["status"] == "RUNNING":
                workflow["status"] = "COMPLETED"
                
        except Exception as e:
            workflow["status"] = "FAILED"
            workflow["error"] = str(e)
            log.error(f"Workflow {workflow_id} failed: {str(e)}")
        
        workflow["completed_at"] = datetime.now()
        log.info(f"Workflow {workflow_id} completed with status: {workflow['status']}")
    
    async def _execute_stage(self, stage: Dict[str, Any], workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single workflow stage"""
        stage_type = stage.get("type")
        agent_name = stage.get("agent")
        
        if stage_type == "agent_analysis":
            return await self._execute_agent_analysis(agent_name, workflow)
        elif stage_type == "parallel_analysis":
            agents = stage.get("agents", [])
            return await self._execute_parallel_analysis(agents, workflow)
        elif stage_type == "consensus":
            return await self._execute_consensus(workflow)
        else:
            return {"success": False, "error": f"Unknown stage type: {stage_type}"}
    
    async def _execute_agent_analysis(self, agent_name: str, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent analysis stage"""
        if agent_name not in self.coordinator.active_agents:
            return {"success": False, "error": f"Agent {agent_name} not found"}
        
        agent = self.coordinator.active_agents[agent_name]
        
        try:
            # In real implementation, would pass market object
            signal = await agent.analyze_market(None)  # Placeholder
            return {"success": True, "signal": signal}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_parallel_analysis(self, agents: List[str], workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Execute parallel analysis with multiple agents"""
        results = await self.coordinator.run_parallel_analysis(workflow["market_id"], agents)
        return {"success": True, "results": results}
    
    async def _execute_consensus(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Execute consensus building stage"""
        # Collect signals from previous stages
        signals = []
        for stage_result in workflow["results"].values():
            if "signal" in stage_result:
                signals.append(stage_result["signal"])
            elif "results" in stage_result:
                for agent_signal in stage_result["results"].values():
                    if agent_signal:
                        signals.append(agent_signal)
        
        if not signals:
            return {"success": False, "error": "No signals found for consensus"}
        
        consensus = await self.coordinator.build_consensus(workflow["market_id"], signals)
        return {"success": True, "consensus": consensus}
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow status"""
        return self.active_workflows.get(workflow_id)
    
    def get_active_workflows(self) -> Dict[str, Dict[str, Any]]:
        """Get all active workflows"""
        return self.active_workflows.copy()