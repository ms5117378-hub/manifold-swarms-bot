"""
Base agent class for the Manifold Swarms Trading Bot
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio
import json
from dataclasses import asdict

from swarms import Agent
from swarm_models import OpenAIChat
from src.models import (
    AgentSignal, AgentMessage, Market, TradingAction, 
    AgentRole, AgentPerformance
)
from src.utils.config import config
from src.utils.logger import AgentLogger


class BaseManifoldAgent(ABC):
    """Base class for all Manifold trading agents"""
    
    def __init__(
        self,
        agent_name: str,
        agent_role: AgentRole,
        system_prompt: Optional[str] = None,
        llm_provider: str = "openai",
        model: str = "gpt-4",
        temperature: float = 0.1,
        max_tokens: int = 2000
    ):
        self.agent_name = agent_name
        self.agent_role = agent_role
        self.system_prompt = system_prompt or self.get_default_prompt()
        self.llm_provider = llm_provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize logger
        self.logger = AgentLogger(agent_name)
        
        # Performance tracking
        self.performance = AgentPerformance(
            agent_name=agent_name,
            agent_role=agent_role,
            total_signals=0,
            profitable_signals=0,
            total_pnl_contribution=0.0,
            average_confidence=0.0,
            accuracy_rate=0.0
        )
        
        # Initialize Swarms Agent
        self.swarms_agent = self._create_swarms_agent()
        
        # Message queue for inter-agent communication
        self.message_queue: List[AgentMessage] = []
        
        # State management
        self.is_active = False
        self.last_activity = datetime.now()
        
    def _create_swarms_agent(self) -> Agent:
        """Create the underlying Swarms agent"""
        # Initialize LLM based on provider
        if self.llm_provider == "openai":
            llm = OpenAIChat(
                model_name=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                openai_api_key=config.get('env.openai_api_key')
            )
        else:
            # Add support for other providers as needed
            raise ValueError(f"LLM provider {self.llm_provider} not yet implemented")
        
        return Agent(
            agent_name=self.agent_name,
            system_prompt=self.system_prompt,
            llm=llm,
            max_loops=1,
            autosave=config.get('monitoring.save_agent_states', True),
            dashboard=False,
            verbose=True,
            streaming_on=False,
            saved_state_path=f"agent_states/{self.agent_name.lower().replace(' ', '_')}_state.json"
        )
    
    @abstractmethod
    def get_default_prompt(self) -> str:
        """Get default system prompt for this agent type"""
        pass
    
    @abstractmethod
    async def analyze_market(self, market: Market) -> AgentSignal:
        """Analyze a market and generate a trading signal"""
        pass
    
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process incoming message from another agent"""
        self.logger.communication(message.sender_agent, message.message_type, str(message.content))
        
        # Add to message queue
        self.message_queue.append(message)
        self.last_activity = datetime.now()
        
        # Process based on message type
        if message.message_type == "REQUEST":
            return await self.handle_request(message)
        elif message.message_type == "RESPONSE":
            return await self.handle_response(message)
        elif message.message_type == "NOTIFICATION":
            return await self.handle_notification(message)
        
        return None
    
    async def handle_request(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle request message"""
        # Default implementation - override in subclasses
        return AgentMessage(
            sender_agent=self.agent_name,
            receiver_agent=message.sender_agent,
            message_type="RESPONSE",
            content={"status": "received", "action": "processed"},
            priority=3
        )
    
    async def handle_response(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle response message"""
        # Default implementation - no response needed
        return None
    
    async def handle_notification(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle notification message"""
        # Default implementation - no response needed
        return None
    
    async def send_message(self, target_agent: str, content: Dict[str, Any], 
                          message_type: str = "REQUEST", priority: int = 3) -> None:
        """Send message to another agent"""
        message = AgentMessage(
            sender_agent=self.agent_name,
            receiver_agent=target_agent,
            message_type=message_type,
            content=content,
            priority=priority
        )
        
        # In a real implementation, this would use a message broker
        # For now, we'll log it and assume it's handled by the orchestrator
        self.logger.communication(target_agent, message_type, str(content))
    
    def update_performance(self, signal: AgentSignal, was_profitable: bool, pnl_contribution: float):
        """Update agent performance metrics"""
        self.performance.total_signals += 1
        if was_profitable:
            self.performance.profitable_signals += 1
        self.performance.total_pnl_contribution += pnl_contribution
        
        # Update average confidence
        total_confidence = self.performance.average_confidence * (self.performance.total_signals - 1)
        self.performance.average_confidence = (total_confidence + signal.confidence) / self.performance.total_signals
        
        # Update accuracy rate
        self.performance.accuracy_rate = self.performance.calculate_accuracy()
        self.performance.last_updated = datetime.now()
        
        self.logger.analysis_complete("performance_update", self.performance.accuracy_rate, 
                                     [f"total_signals: {self.performance.total_signals}"])
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary as dictionary"""
        return {
            "agent_name": self.agent_name,
            "agent_role": self.agent_role.value,
            "total_signals": self.performance.total_signals,
            "accuracy_rate": self.performance.accuracy_rate,
            "total_pnl_contribution": self.performance.total_pnl_contribution,
            "average_confidence": self.performance.average_confidence,
            "last_updated": self.performance.last_updated.isoformat()
        }
    
    async def activate(self):
        """Activate the agent"""
        self.is_active = True
        self.logger.logger.info(f"Agent {self.agent_name} activated")
    
    async def deactivate(self):
        """Deactivate the agent"""
        self.is_active = False
        self.logger.logger.info(f"Agent {self.agent_name} deactivated")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            "agent_name": self.agent_name,
            "agent_role": self.agent_role.value,
            "is_active": self.is_active,
            "last_activity": self.last_activity.isoformat(),
            "message_queue_size": len(self.message_queue),
            "performance": self.get_performance_summary()
        }
    
    def save_state(self, filepath: str):
        """Save agent state to file"""
        state = {
            "agent_name": self.agent_name,
            "agent_role": self.agent_role.value,
            "performance": asdict(self.performance),
            "last_activity": self.last_activity.isoformat(),
            "message_queue": [asdict(msg) for msg in self.message_queue[-10:]]  # Save last 10 messages
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        self.logger.logger.info(f"Agent state saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load agent state from file"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Restore performance data
            if "performance" in state:
                perf_data = state["performance"]
                self.performance = AgentPerformance(
                    agent_name=perf_data["agent_name"],
                    agent_role=AgentRole(perf_data["agent_role"]),
                    total_signals=perf_data["total_signals"],
                    profitable_signals=perf_data["profitable_signals"],
                    total_pnl_contribution=perf_data["total_pnl_contribution"],
                    average_confidence=perf_data["average_confidence"],
                    accuracy_rate=perf_data["accuracy_rate"],
                    last_updated=datetime.fromisoformat(perf_data["last_updated"])
                )
            
            # Restore last activity
            if "last_activity" in state:
                self.last_activity = datetime.fromisoformat(state["last_activity"])
            
            self.logger.logger.info(f"Agent state loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(e, f"loading state from {filepath}")


class AnalystAgent(BaseManifoldAgent):
    """Base class for analyst agents"""
    
    def __init__(self, agent_name: str, specialization: str):
        self.specialization = specialization
        super().__init__(agent_name, AgentRole.ANALYST)
    
    @abstractmethod
    async def calculate_score(self, market: Market) -> float:
        """Calculate analysis score for the market"""
        pass
    
    async def analyze_market(self, market: Market) -> AgentSignal:
        """Analyze market and return analytical signal"""
        score = await self.calculate_score(market)
        
        # Generate reasoning based on score
        reasoning = await self.generate_reasoning(market, score)
        
        return AgentSignal(
            agent_name=self.agent_name,
            agent_role=self.agent_role,
            market_id=market.id,
            action=TradingAction.HOLD,  # Analysts typically provide analysis, not direct trades
            outcome="",  # Not applicable for analysts
            confidence=score,
            reasoning=reasoning,
            estimated_probability=market.probability  # Analysts may adjust probability estimates
        )
    
    async def generate_reasoning(self, market: Market, score: float) -> str:
        """Generate reasoning for the analysis score"""
        prompt = f"""
        Analyze this prediction market and provide reasoning for your score:
        
        Market: {market.question}
        Current Probability: {market.probability:.2f}
        Volume: {market.volume:.0f}M$
        Your Analysis Score: {score:.2f}
        
        Provide detailed reasoning for your score in 2-3 sentences.
        """
        
        try:
            response = await self.swarms_agent.run(prompt)
            return response.strip()
        except Exception as e:
            self.logger.error(e, "generating reasoning")
            return f"Analysis score of {score:.2f} based on {self.specialization} factors."


class StrategyAgent(BaseManifoldAgent):
    """Base class for strategy agents"""
    
    def __init__(self, agent_name: str, strategy_type: str):
        self.strategy_type = strategy_type
        super().__init__(agent_name, AgentRole.STRATEGIST)
    
    @abstractmethod
    async def evaluate_opportunity(self, market: Market, analysis_data: Dict[str, float]) -> Dict[str, Any]:
        """Evaluate trading opportunity based on analysis data"""
        pass
    
    async def analyze_market(self, market: Market) -> AgentSignal:
        """Analyze market and return trading signal"""
        # Get analysis data from other agents (in real implementation)
        analysis_data = {
            "fundamental_score": market.fundamental_score or 0.5,
            "technical_score": market.technical_score or 0.5,
            "sentiment_score": market.sentiment_score or 0.5
        }
        
        opportunity = await self.evaluate_opportunity(market, analysis_data)
        
        return AgentSignal(
            agent_name=self.agent_name,
            agent_role=self.agent_role,
            market_id=market.id,
            action=opportunity.get("action", TradingAction.HOLD),
            outcome=opportunity.get("outcome", ""),
            confidence=opportunity.get("confidence", 0.0),
            reasoning=opportunity.get("reasoning", ""),
            estimated_probability=opportunity.get("estimated_probability"),
            position_size=opportunity.get("position_size", 0.0)
        )