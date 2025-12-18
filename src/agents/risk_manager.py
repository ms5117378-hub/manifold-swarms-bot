"""
Risk Manager Agent
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio

from src.agents.base_agent import BaseManifoldAgent
from src.models import (
    AgentSignal, Market, TradingAction, AgentRole, 
    Position, RiskMetrics, AgentMessage
)
from src.utils.config import config
from src.utils.logger import get_logger, TradingLogger

log = get_logger(__name__)
trading_log = TradingLogger()


class RiskManager(BaseManifoldAgent):
    """Enforces risk limits, position sizing, and portfolio constraints"""
    
    def __init__(self):
        super().__init__("Risk Manager", AgentRole.RISK_MANAGER)
        
        # Risk limits from configuration
        self.risk_config = config.get_risk_limits()
        
        # Portfolio state
        self.current_balance: float = 1000.0  # Starting balance (would come from API)
        self.active_positions: List[Position] = []
        self.risk_metrics: RiskMetrics = RiskMetrics(
            total_exposure=0.0,
            exposure_percentage=0.0,
            position_count=0,
            correlation_risk=0.0,
            liquidity_risk=0.0,
            concentration_risk=0.0,
            var_95=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0
        )
        
        # Risk monitoring
        self.risk_breaches: List[Dict[str, Any]] = []
        self.last_risk_check = datetime.now()
        
    def get_default_prompt(self) -> str:
        return config.get_agent_prompt('risk_management') or """
        You are the risk manager for the trading system, responsible for capital preservation.
        
        Your responsibilities:
        1. Enforce portfolio risk limits and constraints
        2. Validate all trading decisions
        3. Calculate optimal position sizes
        4. Monitor portfolio exposure and correlations
        5. Implement stop-loss and take-profit rules
        
        Risk Rules:
        - Maximum total exposure: 80% of balance
        - Maximum single position: 10% of balance
        - Maximum 5 active positions
        - Minimum 10% balance reserve
        - Stop loss at -15%, take profit at +25%
        
        Provide:
        - Risk approval: APPROVED, APPROVED_REDUCED, REJECTED
        - Recommended position size
        - Risk score: 0.0 to 1.0
        - Specific concerns or conditions
        - Monitoring requirements
        
        Never approve trades that violate risk parameters. Capital preservation is paramount.
        """
    
    async def analyze_market(self, market: Market) -> AgentSignal:
        """Risk manager doesn't generate trading signals, but validates them"""
        return AgentSignal(
            agent_name=self.agent_name,
            agent_role=self.agent_role,
            market_id=market.id,
            action=TradingAction.HOLD,
            outcome="",
            confidence=0.0,
            reasoning="Risk manager validates decisions rather than generating signals"
        )
    
    async def validate_trading_decision(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a trading decision against risk parameters"""
        try:
            market_id = decision.get('market_id')
            action = decision.get('action')
            requested_position_size = decision.get('position_size', 0)
            confidence = decision.get('confidence', 0)
            agent_signals = decision.get('agent_signals', [])
            
            log.info(f"Validating trading decision: {action} on {market_id}")
            
            # Update portfolio state
            await self._update_portfolio_state()
            
            # Calculate risk metrics
            risk_score = self._calculate_risk_score(decision)
            
            # Check all risk constraints
            constraint_checks = await self._check_risk_constraints(decision)
            
            # Determine position size
            recommended_size = self._calculate_optimal_position_size(
                decision, constraint_checks, risk_score
            )
            
            # Make final decision
            if not constraint_checks['all_passed']:
                return {
                    'approved': False,
                    'reason': constraint_checks['failure_reason'],
                    'risk_score': risk_score,
                    'adjusted_position_size': 0,
                    'risk_breaches': constraint_checks['breaches']
                }
            elif recommended_size < requested_position_size:
                return {
                    'approved': True,
                    'reduced': True,
                    'reason': f"Position size reduced from {requested_position_size:.1%} to {recommended_size:.1%} due to risk constraints",
                    'risk_score': risk_score,
                    'adjusted_position_size': recommended_size,
                    'risk_breaches': []
                }
            else:
                return {
                    'approved': True,
                    'reduced': False,
                    'reason': "Trade approved within risk parameters",
                    'risk_score': risk_score,
                    'adjusted_position_size': recommended_size,
                    'risk_breaches': []
                }
                
        except Exception as e:
            log.error(f"Error validating trading decision: {str(e)}")
            return {
                'approved': False,
                'reason': f'Validation error: {str(e)}',
                'risk_score': 1.0,  # High risk on error
                'adjusted_position_size': 0,
                'risk_breaches': [{'type': 'system_error', 'details': str(e)}]
            }
    
    async def _update_portfolio_state(self):
        """Update current portfolio state"""
        # In a real implementation, would fetch from API
        # For now, simulate portfolio state
        
        self.last_risk_check = datetime.now()
        
        # Calculate current exposure
        total_exposure = sum(pos.current_value for pos in self.active_positions)
        self.risk_metrics.total_exposure = total_exposure
        self.risk_metrics.exposure_percentage = total_exposure / self.current_balance if self.current_balance > 0 else 0
        self.risk_metrics.position_count = len(self.active_positions)
        
        # Calculate other risk metrics
        self.risk_metrics.concentration_risk = self._calculate_concentration_risk()
        self.risk_metrics.liquidity_risk = self._calculate_liquidity_risk()
        self.risk_metrics.correlation_risk = self._calculate_correlation_risk()
        
        # Log portfolio summary
        trading_log.portfolio_summary(
            self.current_balance,
            self.risk_metrics.exposure_percentage,
            self.risk_metrics.position_count,
            self._calculate_daily_pnl()
        )
    
    def _calculate_risk_score(self, decision: Dict[str, Any]) -> float:
        """Calculate overall risk score for the decision"""
        risk_factors = []
        
        # Position size risk
        position_size = decision.get('position_size', 0)
        max_position = config.get('risk.max_single_position', 0.10)
        position_risk = position_size / max_position
        risk_factors.append(position_risk)
        
        # Confidence risk (lower confidence = higher risk)
        confidence = decision.get('confidence', 0)
        confidence_risk = 1 - confidence
        risk_factors.append(confidence_risk)
        
        # Portfolio concentration risk
        if self.risk_metrics.position_count >= 4:
            risk_factors.append(0.8)  # High concentration risk
        elif self.risk_metrics.position_count >= 2:
            risk_factors.append(0.5)  # Medium concentration risk
        else:
            risk_factors.append(0.2)  # Low concentration risk
        
        # Exposure risk
        exposure_risk = self.risk_metrics.exposure_percentage / config.get('risk.max_total_exposure', 0.80)
        risk_factors.append(exposure_risk)
        
        # Agent agreement risk
        agent_signals = decision.get('agent_signals', [])
        if len(agent_signals) < 2:
            risk_factors.append(0.7)  # Low agreement
        elif len(agent_signals) < 3:
            risk_factors.append(0.4)  # Medium agreement
        else:
            risk_factors.append(0.2)  # Good agreement
        
        # Calculate weighted average
        weights = [0.3, 0.2, 0.2, 0.2, 0.1]  # Position size most important
        risk_score = sum(risk * weight for risk, weight in zip(risk_factors, weights))
        
        return max(0.0, min(1.0, risk_score))
    
    async def _check_risk_constraints(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Check all risk constraints"""
        checks = {
            'all_passed': True,
            'failure_reason': '',
            'breaches': []
        }
        
        # 1. Maximum total exposure
        max_total_exposure = config.get('risk.max_total_exposure', 0.80)
        if self.risk_metrics.exposure_percentage >= max_total_exposure:
            checks['all_passed'] = False
            checks['failure_reason'] = f"Total exposure ({self.risk_metrics.exposure_percentage:.1%}) exceeds limit ({max_total_exposure:.1%})"
            checks['breaches'].append({
                'type': 'total_exposure',
                'current': self.risk_metrics.exposure_percentage,
                'limit': max_total_exposure
            })
        
        # 2. Maximum position count
        max_positions = config.get('trading.max_active_positions', 5)
        if self.risk_metrics.position_count >= max_positions:
            checks['all_passed'] = False
            checks['failure_reason'] = f"Position count ({self.risk_metrics.position_count}) exceeds limit ({max_positions})"
            checks['breaches'].append({
                'type': 'position_count',
                'current': self.risk_metrics.position_count,
                'limit': max_positions
            })
        
        # 3. Minimum balance reserve
        min_reserve = config.get('risk.min_balance_reserve', 0.10)
        available_balance = self.current_balance - self.risk_metrics.total_exposure
        reserve_percentage = available_balance / self.current_balance if self.current_balance > 0 else 0
        
        if reserve_percentage < min_reserve:
            checks['all_passed'] = False
            checks['failure_reason'] = f"Balance reserve ({reserve_percentage:.1%}) below minimum ({min_reserve:.1%})"
            checks['breaches'].append({
                'type': 'balance_reserve',
                'current': reserve_percentage,
                'limit': min_reserve
            })
        
        # 4. Single position size limit
        max_single_position = config.get('risk.max_single_position', 0.10)
        requested_size = decision.get('position_size', 0)
        
        if requested_size > max_single_position:
            checks['all_passed'] = False
            checks['failure_reason'] = f"Position size ({requested_size:.1%}) exceeds limit ({max_single_position:.1%})"
            checks['breaches'].append({
                'type': 'position_size',
                'current': requested_size,
                'limit': max_single_position
            })
        
        # 5. Minimum consensus confidence
        min_confidence = config.get('trading.min_consensus_confidence', 0.65)
        decision_confidence = decision.get('confidence', 0)
        
        if decision_confidence < min_confidence:
            checks['all_passed'] = False
            checks['failure_reason'] = f"Decision confidence ({decision_confidence:.2f}) below minimum ({min_confidence:.2f})"
            checks['breaches'].append({
                'type': 'confidence_threshold',
                'current': decision_confidence,
                'limit': min_confidence
            })
        
        return checks
    
    def _calculate_optimal_position_size(self, decision: Dict[str, Any], 
                                      constraint_checks: Dict[str, Any], 
                                      risk_score: float) -> float:
        """Calculate optimal position size considering all constraints"""
        requested_size = decision.get('position_size', 0)
        
        # Start with requested size and apply constraints
        optimal_size = requested_size
        
        # 1. Apply maximum position size limit
        max_single_position = config.get('risk.max_single_position', 0.10)
        optimal_size = min(optimal_size, max_single_position)
        
        # 2. Apply portfolio concentration limits
        if self.risk_metrics.position_count >= 3:
            optimal_size = min(optimal_size, 0.05)  # 5% max with 3+ positions
        elif self.risk_metrics.position_count >= 2:
            optimal_size = min(optimal_size, 0.07)  # 7% max with 2 positions
        
        # 3. Apply total exposure limits
        available_exposure = config.get('risk.max_total_exposure', 0.80) - self.risk_metrics.exposure_percentage
        if available_exposure > 0:
            optimal_size = min(optimal_size, available_exposure)
        else:
            optimal_size = 0  # No room for new positions
        
        # 4. Apply risk score adjustment
        if risk_score > 0.7:
            optimal_size *= 0.5  # Reduce size for high risk
        elif risk_score > 0.5:
            optimal_size *= 0.75  # Reduce size for medium risk
        
        # 5. Apply confidence adjustment
        confidence = decision.get('confidence', 0)
        confidence_multiplier = max(0.5, confidence)  # Minimum 50% of calculated size
        optimal_size *= confidence_multiplier
        
        # Ensure minimum position size for meaningful trades
        min_position = 0.01  # 1% minimum
        optimal_size = max(optimal_size, min_position if requested_size > 0 else 0)
        
        return optimal_size
    
    def _calculate_concentration_risk(self) -> float:
        """Calculate portfolio concentration risk"""
        if not self.active_positions:
            return 0.0
        
        # Calculate concentration using Herfindahl-Hirschman Index
        total_value = sum(pos.current_value for pos in self.active_positions)
        if total_value == 0:
            return 0.0
        
        hhi = sum((pos.current_value / total_value) ** 2 for pos in self.active_positions)
        
        # Normalize to 0-1 scale
        return (hhi - 1/len(self.active_positions)) / (1 - 1/len(self.active_positions)) if len(self.active_positions) > 1 else 0.0
    
    def _calculate_liquidity_risk(self) -> float:
        """Calculate liquidity risk based on position characteristics"""
        if not self.active_positions:
            return 0.0
        
        liquidity_scores = []
        
        for position in self.active_positions:
            # Simplified liquidity scoring based on volume
            # In a real implementation, would use more sophisticated metrics
            if hasattr(position, 'volume'):
                volume = position.volume
                if volume >= 1000:
                    liquidity_scores.append(0.1)  # Low risk
                elif volume >= 500:
                    liquidity_scores.append(0.3)  # Low-medium risk
                elif volume >= 100:
                    liquidity_scores.append(0.6)  # Medium-high risk
                else:
                    liquidity_scores.append(0.9)  # High risk
            else:
                liquidity_scores.append(0.5)  # Default medium risk
        
        return sum(liquidity_scores) / len(liquidity_scores)
    
    def _calculate_correlation_risk(self) -> float:
        """Calculate correlation risk between positions"""
        # Simplified correlation risk calculation
        # In a real implementation, would calculate actual correlations
        
        if len(self.active_positions) <= 1:
            return 0.0
        
        # Assume some correlation based on position count
        # More positions = higher chance of correlation
        return min(0.8, len(self.active_positions) * 0.15)
    
    def _calculate_daily_pnl(self) -> float:
        """Calculate daily P&L"""
        # Simplified daily P&L calculation
        # In a real implementation, would track actual daily changes
        return sum(pos.calculate_pnl() for pos in self.active_positions)
    
    async def monitor_positions(self):
        """Monitor existing positions for risk management"""
        await self._update_portfolio_state()
        
        positions_to_close = []
        
        for position in self.active_positions:
            # Check stop-loss
            if position.calculate_pnl_percentage() <= config.get('risk.stop_loss_threshold', -0.15):
                positions_to_close.append({
                    'position': position,
                    'reason': 'stop_loss',
                    'pnl': position.calculate_pnl(),
                    'pnl_percent': position.calculate_pnl_percentage()
                })
                trading_log.position_closed(
                    position.market_id,
                    position.calculate_pnl(),
                    position.calculate_pnl_percentage()
                )
            
            # Check take-profit
            elif position.calculate_pnl_percentage() >= config.get('risk.take_profit_threshold', 0.25):
                positions_to_close.append({
                    'position': position,
                    'reason': 'take_profit',
                    'pnl': position.calculate_pnl(),
                    'pnl_percent': position.calculate_pnl_percentage()
                })
                trading_log.position_closed(
                    position.market_id,
                    position.calculate_pnl(),
                    position.calculate_pnl_percentage()
                )
        
        # Close positions that hit limits
        for close_info in positions_to_close:
            await self._close_position(close_info['position'], close_info['reason'])
        
        if positions_to_close:
            log.info(f"Closed {len(positions_to_close)} positions due to risk limits")
    
    async def _close_position(self, position: Position, reason: str):
        """Close a position (would execute trade in real implementation)"""
        # Remove from active positions
        self.active_positions = [p for p in self.active_positions if p.market_id != position.market_id]
        
        # Log the closure
        log.info(f"Closed position {position.market_id} due to {reason}: P&L {position.calculate_pnl():.2f}M$")
        
        # Update balance (simplified)
        self.current_balance += position.current_value
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary"""
        return {
            'current_balance': self.current_balance,
            'total_exposure': self.risk_metrics.total_exposure,
            'exposure_percentage': self.risk_metrics.exposure_percentage,
            'active_positions': self.risk_metrics.position_count,
            'available_balance': self.current_balance - self.risk_metrics.total_exposure,
            'reserve_percentage': (self.current_balance - self.risk_metrics.total_exposure) / self.current_balance if self.current_balance > 0 else 0,
            'concentration_risk': self.risk_metrics.concentration_risk,
            'liquidity_risk': self.risk_metrics.liquidity_risk,
            'correlation_risk': self.risk_metrics.correlation_risk,
            'risk_breaches_today': len(self.risk_breaches),
            'last_risk_check': self.last_risk_check.isoformat()
        }
    
    def get_position_details(self) -> List[Dict[str, Any]]:
        """Get details of all active positions"""
        return [
            {
                'market_id': pos.market_id,
                'market_question': pos.market_question,
                'outcome': pos.outcome,
                'initial_stake': pos.initial_stake,
                'current_value': pos.current_value,
                'unrealized_pnl': pos.calculate_pnl(),
                'pnl_percentage': pos.calculate_pnl_percentage(),
                'opened_at': pos.opened_at.isoformat(),
                'days_held': (datetime.now() - pos.opened_at).days
            }
            for pos in self.active_positions
        ]
    
    async def handle_request(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle risk management requests"""
        if message.message_type == "REQUEST":
            content = message.content
            
            if 'decision' in content:
                # Validate trading decision
                validation_result = await self.validate_trading_decision(content['decision'])
                
                return AgentMessage(
                    sender_agent=self.agent_name,
                    receiver_agent=message.sender_agent,
                    message_type="RESPONSE",
                    content=validation_result,
                    priority=1
                )
            
            elif 'portfolio_status' in content:
                # Return portfolio status
                portfolio_status = self.get_risk_summary()
                
                return AgentMessage(
                    sender_agent=self.agent_name,
                    receiver_agent=message.sender_agent,
                    message_type="RESPONSE",
                    content={'portfolio_status': portfolio_status},
                    priority=2
                )
        
        return None