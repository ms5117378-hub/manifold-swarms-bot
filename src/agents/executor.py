"""
Trade Executor Agent
"""
from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio
import time

from src.agents.base_agent import BaseManifoldAgent
from src.models import (
    AgentSignal, Market, TradingAction, AgentRole, 
    Trade, Position, AgentMessage
)
from src.manifold.market_fetcher import ManifoldAPIClient
from src.utils.config import config
from src.utils.logger import get_logger, TradingLogger

log = get_logger(__name__)
trading_log = TradingLogger()


class TradeExecutor(BaseManifoldAgent):
    """Executes approved trades through Manifold API with retry logic and error handling"""
    
    def __init__(self):
        super().__init__("Trade Executor", AgentRole.EXECUTOR)
        
        # Execution state
        self.execution_queue: List[Dict[str, Any]] = []
        self.execution_history: List[Dict[str, Any]] = []
        self.active_positions: Dict[str, Position] = {}
        
        # Execution parameters
        self.max_retries = config.get('agents.max_retries', 3)
        self.retry_delay = 1.0  # Base delay in seconds
        self.execution_timeout = 30  # Seconds
        
        # API client
        self.api_client: Optional[ManifoldAPIClient] = None
        
    def get_default_prompt(self) -> str:
        return config.get_agent_prompt('execution') or """
        You are the trade executor responsible for implementing approved trading decisions.
        
        Your responsibilities:
        1. Execute trades through Manifold API
        2. Handle retries and error recovery
        3. Confirm successful trade execution
        4. Update portfolio state
        5. Report execution results
        
        Execution Protocol:
        1. Validate market is still open and liquid
        2. Check price hasn't moved significantly
        3. Execute with optimal timing
        4. Retry up to 3 times with exponential backoff
        5. Log all execution details
        
        Provide:
        - Execution status: SUCCESS, FAILED, PARTIAL
        - Actual execution price
        - Slippage from expected price
        - Transaction details
        - Any issues encountered
        
        Ensure all approved trades are executed efficiently and accurately.
        """
    
    async def analyze_market(self, market: Market) -> AgentSignal:
        """Trade executor doesn't generate signals, only executes trades"""
        return AgentSignal(
            agent_name=self.agent_name,
            agent_role=self.agent_role,
            market_id=market.id,
            action=TradingAction.HOLD,
            outcome="",
            confidence=0.0,
            reasoning="Trade executor only executes approved trades"
        )
    
    async def execute_trade_decision(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an approved trading decision"""
        try:
            market_id = decision.get('market_id')
            action = decision.get('action')
            outcome = decision.get('outcome')
            position_size = decision.get('position_size', 0)
            confidence = decision.get('confidence', 0)
            
            log.info(f"Executing trade: {action} {position_size:.1%} on {market_id}")
            
            # Validate execution parameters
            if not self._validate_execution_parameters(decision):
                return {
                    'status': 'FAILED',
                    'reason': 'Invalid execution parameters',
                    'executed_at': datetime.now().isoformat()
                }
            
            # Add to execution queue
            execution_request = {
                'id': f"exec_{market_id}_{int(time.time())}",
                'market_id': market_id,
                'action': action,
                'outcome': outcome,
                'position_size': position_size,
                'confidence': confidence,
                'requested_at': datetime.now(),
                'status': 'pending',
                'retries': 0
            }
            
            self.execution_queue.append(execution_request)
            
            # Execute the trade
            execution_result = await self._execute_trade_with_retry(execution_request)
            
            # Update execution history
            execution_request.update(execution_result)
            execution_request['completed_at'] = datetime.now()
            self.execution_history.append(execution_request)
            
            # Remove from queue
            self.execution_queue = [req for req in self.execution_queue if req['id'] != execution_request['id']]
            
            return execution_result
            
        except Exception as e:
            log.error(f"Error executing trade decision: {str(e)}")
            return {
                'status': 'FAILED',
                'reason': f'Execution error: {str(e)}',
                'executed_at': datetime.now().isoformat()
            }
    
    def _validate_execution_parameters(self, decision: Dict[str, Any]) -> bool:
        """Validate trade execution parameters"""
        required_fields = ['market_id', 'action', 'outcome', 'position_size']
        
        for field in required_fields:
            if field not in decision:
                log.error(f"Missing required field: {field}")
                return False
        
        # Validate action
        action = decision.get('action')
        if action not in ['BUY', 'SELL']:
            log.error(f"Invalid action: {action}")
            return False
        
        # Validate position size
        position_size = decision.get('position_size', 0)
        if position_size <= 0:
            log.error(f"Invalid position size: {position_size}")
            return False
        
        # Validate outcome
        outcome = decision.get('outcome')
        if outcome not in ['YES', 'NO']:
            log.error(f"Invalid outcome: {outcome}")
            return False
        
        return True
    
    async def _execute_trade_with_retry(self, execution_request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute trade with retry logic"""
        market_id = execution_request['market_id']
        action = execution_request['action']
        outcome = execution_request['outcome']
        position_size = execution_request['position_size']
        
        for attempt in range(self.max_retries + 1):
            try:
                # Initialize API client if needed
                if not self.api_client:
                    self.api_client = ManifoldAPIClient()
                
                # Execute trade
                execution_result = await self._execute_single_trade(execution_request)
                
                if execution_result['status'] == 'SUCCESS':
                    log.info(f"Trade executed successfully on attempt {attempt + 1}")
                    return execution_result
                elif attempt < self.max_retries:
                    # Retry with exponential backoff
                    delay = self.retry_delay * (2 ** attempt)
                    log.warning(f"Trade failed, retrying in {delay}s (attempt {attempt + 1}/{self.max_retries + 1})")
                    await asyncio.sleep(delay)
                    execution_request['retries'] = attempt + 1
                else:
                    log.error(f"Trade failed after {self.max_retries + 1} attempts")
                    return execution_result
                    
            except Exception as e:
                log.error(f"Trade execution error on attempt {attempt + 1}: {str(e)}")
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                    execution_request['retries'] = attempt + 1
                else:
                    return {
                        'status': 'FAILED',
                        'reason': f'Failed after {self.max_retries + 1} attempts: {str(e)}',
                        'attempts': attempt + 1
                    }
        
        return {
            'status': 'FAILED',
            'reason': 'Max retries exceeded',
            'attempts': self.max_retries + 1
        }
    
    async def _execute_single_trade(self, execution_request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single trade attempt"""
        market_id = execution_request['market_id']
        action = execution_request['action']
        outcome = execution_request['outcome']
        position_size = execution_request['position_size']
        
        try:
            async with self.api_client as client:
                # Get current market state
                market_data = await client.get_market(market_id)
                
                # Validate market is still tradable
                if not self._validate_market_state(market_data):
                    return {
                        'status': 'FAILED',
                        'reason': 'Market no longer tradable',
                        'market_state': market_data.get('resolution', 'UNKNOWN')
                    }
                
                # Calculate trade amount
                current_balance = await client.get_user_balance('me')  # Would use actual user ID
                trade_amount = current_balance * position_size
                
                # Execute trade
                start_time = time.time()
                
                if action == 'BUY':
                    if outcome == 'YES':
                        bet_result = await client.place_bet(market_id, 'YES', trade_amount)
                    else:  # outcome == 'NO'
                        bet_result = await client.place_bet(market_id, 'NO', trade_amount)
                else:  # SELL
                    # For selling, we'd need to find existing shares to sell
                    # Simplified implementation
                    bet_result = await client.place_bet(market_id, outcome, trade_amount)
                
                execution_time = time.time() - start_time
                
                # Process execution result
                if bet_result and 'id' in bet_result:
                    # Success
                    trade = Trade(
                        market_id=market_id,
                        market_question=market_data.get('question', ''),
                        action=TradingAction(action),
                        outcome=outcome,
                        amount=trade_amount,
                        price=market_data.get('probability', 0.5),
                        shares=bet_result.get('shares', 0),
                        fee=bet_result.get('fee', 0),
                        executed_at=datetime.now(),
                        transaction_id=bet_result.get('id')
                    )
                    
                    # Log trade
                    trading_log.trade_executed(
                        market_id, action, trade_amount, trade.price
                    )
                    
                    # Update position
                    self._update_position(trade, market_data)
                    
                    return {
                        'status': 'SUCCESS',
                        'trade_id': bet_result['id'],
                        'amount': trade_amount,
                        'price': trade.price,
                        'shares': trade.shares,
                        'fee': trade.fee,
                        'execution_time': execution_time,
                        'slippage': self._calculate_slippage(market_data, trade.price)
                    }
                else:
                    # Failure
                    return {
                        'status': 'FAILED',
                        'reason': bet_result.get('error', 'Unknown API error'),
                        'execution_time': execution_time
                    }
                    
        except Exception as e:
            return {
                'status': 'FAILED',
                'reason': f'API error: {str(e)}',
                'execution_time': 0
            }
    
    def _validate_market_state(self, market_data: Dict[str, Any]) -> bool:
        """Validate that market is still tradable"""
        # Check if market is resolved
        if market_data.get('resolution'):
            return False
        
        # Check if market is closed
        close_time = market_data.get('closeTime')
        if close_time:
            close_datetime = datetime.fromisoformat(close_time.replace('Z', '+00:00'))
            if close_datetime <= datetime.now():
                return False
        
        return True
    
    def _calculate_slippage(self, market_data: Dict[str, Any], execution_price: float) -> float:
        """Calculate price slippage from expected price"""
        expected_price = market_data.get('probability', 0.5)
        slippage = abs(execution_price - expected_price)
        
        return slippage
    
    def _update_position(self, trade: Trade, market_data: Dict[str, Any]):
        """Update position tracking"""
        market_id = trade.market_id
        
        if market_id not in self.active_positions:
            # Create new position
            position = Position(
                market_id=market_id,
                market_question=trade.market_question,
                outcome=trade.outcome,
                initial_stake=trade.amount,
                initial_probability=trade.price,
                current_probability=market_data.get('probability', trade.price),
                shares=trade.shares,
                current_value=trade.calculate_cost(),
                unrealized_pnl=0.0,
                opened_at=trade.executed_at
            )
            self.active_positions[market_id] = position
            
            # Log position opening
            trading_log.position_opened(market_id, trade.amount, trade.outcome)
        else:
            # Update existing position
            position = self.active_positions[market_id]
            
            # Simplified position update (would be more complex in reality)
            position.current_probability = market_data.get('probability', position.current_probability)
            position.current_value = position.shares * position.current_probability
            position.unrealized_pnl = position.calculate_pnl()
            position.last_updated = datetime.now()
    
    async def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific execution"""
        for execution in self.execution_history:
            if execution.get('id') == execution_id:
                return execution
        return None
    
    def get_execution_queue(self) -> List[Dict[str, Any]]:
        """Get current execution queue"""
        return self.execution_queue.copy()
    
    def get_execution_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get execution history"""
        return self.execution_history[-limit:]
    
    def get_active_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get all active positions"""
        return {
            market_id: {
                'market_question': pos.market_question,
                'outcome': pos.outcome,
                'initial_stake': pos.initial_stake,
                'current_value': pos.current_value,
                'unrealized_pnl': pos.calculate_pnl(),
                'pnl_percentage': pos.calculate_pnl_percentage(),
                'shares': pos.shares,
                'current_probability': pos.current_probability,
                'opened_at': pos.opened_at.isoformat(),
                'days_held': (datetime.now() - pos.opened_at).days
            }
            for market_id, pos in self.active_positions.items()
        }
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get execution statistics"""
        successful_executions = [e for e in self.execution_history if e.get('status') == 'SUCCESS']
        failed_executions = [e for e in self.execution_history if e.get('status') == 'FAILED']
        
        total_executed = sum(e.get('amount', 0) for e in successful_executions)
        total_fees = sum(e.get('fee', 0) for e in successful_executions)
        avg_execution_time = sum(e.get('execution_time', 0) for e in successful_executions) / len(successful_executions) if successful_executions else 0
        avg_slippage = sum(e.get('slippage', 0) for e in successful_executions) / len(successful_executions) if successful_executions else 0
        
        return {
            'total_executions': len(self.execution_history),
            'successful_executions': len(successful_executions),
            'failed_executions': len(failed_executions),
            'success_rate': len(successful_executions) / len(self.execution_history) if self.execution_history else 0,
            'total_amount_executed': total_executed,
            'total_fees_paid': total_fees,
            'average_execution_time': avg_execution_time,
            'average_slippage': avg_slippage,
            'queue_length': len(self.execution_queue),
            'active_positions': len(self.active_positions)
        }
    
    async def handle_request(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle execution requests"""
        if message.message_type == "REQUEST":
            content = message.content
            
            if 'execute_decision' in content:
                # Execute trading decision
                execution_result = await self.execute_trade_decision(content['execute_decision'])
                
                return AgentMessage(
                    sender_agent=self.agent_name,
                    receiver_agent=message.sender_agent,
                    message_type="RESPONSE",
                    content={'execution_result': execution_result},
                    priority=1
                )
            
            elif 'execution_status' in content:
                # Get execution status
                execution_id = content['execution_status']
                status = await self.get_execution_status(execution_id)
                
                return AgentMessage(
                    sender_agent=self.agent_name,
                    receiver_agent=message.sender_agent,
                    message_type="RESPONSE",
                    content={'execution_status': status},
                    priority=2
                )
            
            elif 'execution_statistics' in content:
                # Get execution statistics
                stats = self.get_execution_statistics()
                
                return AgentMessage(
                    sender_agent=self.agent_name,
                    receiver_agent=message.sender_agent,
                    message_type="RESPONSE",
                    content={'execution_statistics': stats},
                    priority=3
                )
        
        return None
    
    async def monitor_positions(self):
        """Monitor and update active positions"""
        if not self.api_client:
            return
        
        try:
            async with self.api_client as client:
                for market_id, position in self.active_positions.items():
                    # Get current market data
                    market_data = await client.get_market(market_id)
                    
                    if market_data:
                        # Update position
                        position.current_probability = market_data.get('probability', position.current_probability)
                        position.current_value = position.shares * position.current_probability
                        position.unrealized_pnl = position.calculate_pnl()
                        position.last_updated = datetime.now()
                        
                        # Check if position should be closed (market resolved)
                        if market_data.get('resolution'):
                            await self._close_resolved_position(position, market_data)
                            
        except Exception as e:
            log.error(f"Error monitoring positions: {str(e)}")
    
    async def _close_resolved_position(self, position: Position, market_data: Dict[str, Any]):
        """Close position due to market resolution"""
        resolution = market_data.get('resolution')
        
        # Calculate final P&L
        if resolution == position.outcome:
            final_value = position.shares  # Won
        else:
            final_value = 0  # Lost
        
        final_pnl = final_value - position.initial_stake
        
        # Log position closure
        trading_log.position_closed(
            position.market_id,
            final_pnl,
            final_pnl / position.initial_stake if position.initial_stake > 0 else 0
        )
        
        # Remove from active positions
        if position.market_id in self.active_positions:
            del self.active_positions[position.market_id]
        
        log.info(f"Closed resolved position {position.market_id}: {resolution}, P&L: {final_pnl:.2f}M$")