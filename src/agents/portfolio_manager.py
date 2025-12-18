"""
Portfolio Manager Agent
"""
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import asyncio

from src.agents.base_agent import BaseManifoldAgent
from src.models import (
    AgentSignal, Market, TradingAction, AgentRole, 
    Position, AgentPerformance, AgentMessage
)
from src.utils.config import config
from src.utils.logger import get_logger, TradingLogger

log = get_logger(__name__)
trading_log = TradingLogger()


class PortfolioManager(BaseManifoldAgent):
    """Monitors all positions, calculates P&L, suggests rebalancing, generates performance reports"""
    
    def __init__(self):
        super().__init__("Portfolio Manager", AgentRole.PORTFOLIO_MANAGER)
        
        # Portfolio state
        self.current_balance: float = 1000.0  # Starting balance
        self.initial_balance: float = 1000.0
        self.active_positions: List[Position] = []
        self.closed_positions: List[Position] = []
        self.daily_pnl_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.agent_contributions: Dict[str, Dict[str, float]] = {}
        self.portfolio_metrics: Dict[str, float] = {}
        self.last_performance_update = datetime.now()
        
        # Rebalancing parameters
        self.rebalance_threshold = 0.10  # 10% deviation triggers rebalance
        self.max_position_weight = 0.15  # 15% max weight per position
        
    def get_default_prompt(self) -> str:
        return config.get_agent_prompt('portfolio_management') or """
        You are the portfolio manager responsible for overall performance and position monitoring.
        
        Your responsibilities:
        1. Monitor all active positions and P&L
        2. Generate performance reports
        3. Suggest portfolio rebalancing
        4. Track agent performance contributions
        5. Identify optimization opportunities
        
        Monitoring Tasks:
        1. Hourly position reviews
        2. Daily performance summaries
        3. Weekly agent performance analysis
        4. Monthly strategy effectiveness review
        5. Risk exposure calculations
        
        Provide:
        - Portfolio P&L and metrics
        - Position performance breakdown
        - Agent contribution rankings
        - Risk exposure analysis
        - Rebalancing recommendations
        
        Focus on maximizing risk-adjusted returns while maintaining portfolio discipline.
        """
    
    async def analyze_market(self, market: Market) -> AgentSignal:
        """Portfolio manager doesn't generate trading signals"""
        return AgentSignal(
            agent_name=self.agent_name,
            agent_role=self.agent_role,
            market_id=market.id,
            action=TradingAction.HOLD,
            outcome="",
            confidence=0.0,
            reasoning="Portfolio manager monitors positions rather than generating signals"
        )
    
    async def update_portfolio_state(self, positions: List[Position], trades: List[Dict[str, Any]]):
        """Update portfolio state with new positions and trades"""
        try:
            # Update active positions
            self.active_positions = positions
            
            # Process new trades
            for trade in trades:
                await self._process_trade(trade)
            
            # Calculate portfolio metrics
            await self._calculate_portfolio_metrics()
            
            # Check for rebalancing opportunities
            rebalance_recommendations = await self._analyze_rebalancing_needs()
            
            # Update agent contributions
            await self._update_agent_contributions(trades)
            
            # Generate performance report
            if self._should_generate_report():
                await self._generate_performance_report()
            
            log.info(f"Portfolio updated: {len(self.active_positions)} positions, "
                     f"Total value: {self._calculate_total_value():.2f}M$")
            
            return {
                'portfolio_value': self._calculate_total_value(),
                'total_pnl': self._calculate_total_pnl(),
                'active_positions': len(self.active_positions),
                'rebalance_recommendations': rebalance_recommendations
            }
            
        except Exception as e:
            log.error(f"Error updating portfolio state: {str(e)}")
            return None
    
    async def _process_trade(self, trade: Dict[str, Any]):
        """Process a new trade and update portfolio"""
        trade_type = trade.get('type', 'unknown')
        
        if trade_type == 'open_position':
            # New position opened
            position = Position(
                market_id=trade.get('market_id'),
                market_question=trade.get('market_question', ''),
                outcome=trade.get('outcome'),
                initial_stake=trade.get('amount'),
                initial_probability=trade.get('price'),
                current_probability=trade.get('price'),
                shares=trade.get('shares', 0),
                current_value=trade.get('amount'),
                unrealized_pnl=0.0,
                opened_at=datetime.fromisoformat(trade.get('executed_at', datetime.now().isoformat()))
            )
            self.active_positions.append(position)
            
        elif trade_type == 'close_position':
            # Position closed
            market_id = trade.get('market_id')
            for i, pos in enumerate(self.active_positions):
                if pos.market_id == market_id:
                    # Update final P&L
                    pos.current_value = trade.get('close_value', 0)
                    pos.unrealized_pnl = pos.calculate_pnl()
                    
                    # Move to closed positions
                    self.closed_positions.append(pos)
                    self.active_positions.pop(i)
                    break
    
    async def _calculate_portfolio_metrics(self):
        """Calculate comprehensive portfolio metrics"""
        total_value = self._calculate_total_value()
        total_pnl = self._calculate_total_pnl()
        total_return = self._calculate_total_return()
        
        # Calculate risk metrics
        max_drawdown = self._calculate_max_drawdown()
        volatility = self._calculate_portfolio_volatility()
        sharpe_ratio = self._calculate_sharpe_ratio()
        
        # Position concentration
        concentration = self._calculate_concentration_metrics()
        
        # Update portfolio metrics
        self.portfolio_metrics = {
            'total_value': total_value,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'daily_return': self._calculate_daily_return(),
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'position_count': len(self.active_positions),
            'cash_balance': self.current_balance,
            'exposure_percentage': (total_value - self.current_balance) / total_value if total_value > 0 else 0,
            'concentration_metrics': concentration,
            'last_updated': datetime.now().isoformat()
        }
        
        # Log portfolio summary
        trading_log.portfolio_summary(
            self.current_balance,
            self.portfolio_metrics['exposure_percentage'],
            len(self.active_positions),
            self._calculate_daily_pnl()
        )
    
    def _calculate_total_value(self) -> float:
        """Calculate total portfolio value"""
        positions_value = sum(pos.current_value for pos in self.active_positions)
        return self.current_balance + positions_value
    
    def _calculate_total_pnl(self) -> float:
        """Calculate total portfolio P&L"""
        active_pnl = sum(pos.unrealized_pnl for pos in self.active_positions)
        closed_pnl = sum(pos.calculate_pnl() for pos in self.closed_positions)
        return active_pnl + closed_pnl
    
    def _calculate_total_return(self) -> float:
        """Calculate total portfolio return"""
        if self.initial_balance == 0:
            return 0.0
        return (self._calculate_total_value() - self.initial_balance) / self.initial_balance
    
    def _calculate_daily_return(self) -> float:
        """Calculate daily return"""
        today = datetime.now().date()
        
        # Find today's starting value
        today_start_value = self.initial_balance
        for day_pnl in self.daily_pnl_history:
            if day_pnl['date'] == today.isoformat():
                today_start_value = day_pnl['start_value']
                break
        
        current_value = self._calculate_total_value()
        return (current_value - today_start_value) / today_start_value if today_start_value > 0 else 0.0
    
    def _calculate_daily_pnl(self) -> float:
        """Calculate daily P&L"""
        today = datetime.now().date()
        
        # Sum P&L changes for today
        daily_pnl = 0.0
        for pos in self.active_positions:
            # Simplified daily P&L calculation
            # In reality, would track actual daily changes
            daily_pnl += pos.unrealized_pnl * 0.01  # Placeholder
        
        return daily_pnl
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if not self.daily_pnl_history:
            return 0.0
        
        peak_value = self.initial_balance
        max_drawdown = 0.0
        
        for day_pnl in self.daily_pnl_history:
            current_value = day_pnl['end_value']
            
            if current_value > peak_value:
                peak_value = current_value
            
            drawdown = (peak_value - current_value) / peak_value
            max_drawdown = max(max_drawdown, drawdown)
        
        # Check current drawdown
        current_value = self._calculate_total_value()
        if current_value > peak_value:
            peak_value = current_value
        
        current_drawdown = (peak_value - current_value) / peak_value
        max_drawdown = max(max_drawdown, current_drawdown)
        
        return max_drawdown
    
    def _calculate_portfolio_volatility(self) -> float:
        """Calculate portfolio volatility (standard deviation of daily returns)"""
        if len(self.daily_pnl_history) < 2:
            return 0.0
        
        daily_returns = []
        for day_pnl in self.daily_pnl_history:
            daily_returns.append(day_pnl['daily_return'])
        
        # Calculate standard deviation
        mean_return = sum(daily_returns) / len(daily_returns)
        variance = sum((r - mean_return) ** 2 for r in daily_returns) / len(daily_returns)
        
        return variance ** 0.5  # Standard deviation
    
    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        volatility = self._calculate_portfolio_volatility()
        if volatility == 0:
            return 0.0
        
        # Annualized returns and volatility
        annual_return = self._calculate_total_return() * 365  # Simplified
        annual_volatility = volatility * (365 ** 0.5)
        
        return (annual_return - risk_free_rate) / annual_volatility
    
    def _calculate_concentration_metrics(self) -> Dict[str, float]:
        """Calculate portfolio concentration metrics"""
        if not self.active_positions:
            return {'herfindahl_index': 0, 'max_position_weight': 0, 'effective_positions': 0}
        
        total_value = self._calculate_total_value() - self.current_balance
        if total_value == 0:
            return {'herfindahl_index': 0, 'max_position_weight': 0, 'effective_positions': 0}
        
        # Calculate position weights
        weights = [pos.current_value / total_value for pos in self.active_positions]
        
        # Herfindahl-Hirschman Index
        hhi = sum(w ** 2 for w in weights)
        
        # Maximum position weight
        max_weight = max(weights) if weights else 0
        
        # Effective number of positions
        effective_positions = 1 / hhi if hhi > 0 else 0
        
        return {
            'herfindahl_index': hhi,
            'max_position_weight': max_weight,
            'effective_positions': effective_positions
        }
    
    async def _analyze_rebalancing_needs(self) -> List[Dict[str, Any]]:
        """Analyze if portfolio needs rebalancing"""
        recommendations = []
        
        if not self.active_positions:
            return recommendations
        
        total_value = self._calculate_total_value() - self.current_balance
        if total_value == 0:
            return recommendations
        
        # Check position weight limits
        for pos in self.active_positions:
            weight = pos.current_value / total_value
            
            if weight > self.max_position_weight:
                recommendations.append({
                    'type': 'reduce_position',
                    'market_id': pos.market_id,
                    'current_weight': weight,
                    'target_weight': self.max_position_weight,
                    'reason': f'Position weight ({weight:.1%}) exceeds maximum ({self.max_position_weight:.1%})'
                })
        
        # Check underweight positions
        for pos in self.active_positions:
            weight = pos.current_value / total_value
            
            if weight < self.rebalance_threshold and pos.unrealized_pnl < 0:
                recommendations.append({
                    'type': 'consider_closing',
                    'market_id': pos.market_id,
                    'current_weight': weight,
                    'pnl': pos.unrealized_pnl,
                    'pnl_percent': pos.calculate_pnl_percentage(),
                    'reason': f'Underweight position ({weight:.1%}) with losses ({pos.calculate_pnl_percentage():.1%})'
                })
        
        # Check overall portfolio balance
        concentration = self._calculate_concentration_metrics()
        if concentration['max_position_weight'] > 0.25:
            recommendations.append({
                'type': 'diversify_portfolio',
                'current_concentration': concentration['max_position_weight'],
                'reason': f'Portfolio too concentrated ({concentration["max_position_weight"]:.1%} in single position)'
            })
        
        return recommendations
    
    async def _update_agent_contributions(self, trades: List[Dict[str, Any]]):
        """Update agent performance contributions"""
        for trade in trades:
            agent_name = trade.get('agent_name')
            if not agent_name:
                continue
            
            pnl = trade.get('pnl', 0)
            
            if agent_name not in self.agent_contributions:
                self.agent_contributions[agent_name] = {
                    'total_trades': 0,
                    'profitable_trades': 0,
                    'total_pnl': 0.0,
                    'win_rate': 0.0,
                    'average_pnl': 0.0
                }
            
            # Update agent stats
            stats = self.agent_contributions[agent_name]
            stats['total_trades'] += 1
            stats['total_pnl'] += pnl
            
            if pnl > 0:
                stats['profitable_trades'] += 1
            
            # Calculate derived metrics
            stats['win_rate'] = stats['profitable_trades'] / stats['total_trades']
            stats['average_pnl'] = stats['total_pnl'] / stats['total_trades']
    
    def _should_generate_report(self) -> bool:
        """Check if should generate performance report"""
        now = datetime.now()
        
        # Generate report daily
        if now.date() != self.last_performance_update.date():
            return True
        
        return False
    
    async def _generate_performance_report(self):
        """Generate comprehensive performance report"""
        try:
            report = {
                'report_date': datetime.now().isoformat(),
                'portfolio_metrics': self.portfolio_metrics,
                'position_breakdown': self._get_position_breakdown(),
                'agent_performance': self.agent_contributions,
                'risk_metrics': self._calculate_risk_metrics(),
                'recommendations': await self._analyze_rebalancing_needs()
            }
            
            # Log report
            log.info(f"Performance Report Generated:")
            log.info(f"  Total Return: {self._calculate_total_return():.2%}")
            log.info(f"  Sharpe Ratio: {self._calculate_sharpe_ratio():.2f}")
            log.info(f"  Max Drawdown: {self._calculate_max_drawdown():.2%}")
            log.info(f"  Active Positions: {len(self.active_positions)}")
            
            # Update daily P&L history
            self.daily_pnl_history.append({
                'date': datetime.now().date().isoformat(),
                'start_value': self._calculate_total_value() - self._calculate_daily_pnl(),
                'end_value': self._calculate_total_value(),
                'daily_return': self._calculate_daily_return(),
                'daily_pnl': self._calculate_daily_pnl()
            })
            
            # Keep only last 90 days
            if len(self.daily_pnl_history) > 90:
                self.daily_pnl_history = self.daily_pnl_history[-90:]
            
            self.last_performance_update = datetime.now()
            
            return report
            
        except Exception as e:
            log.error(f"Error generating performance report: {str(e)}")
            return None
    
    def _get_position_breakdown(self) -> List[Dict[str, Any]]:
        """Get detailed breakdown of all positions"""
        positions = []
        
        for pos in self.active_positions:
            positions.append({
                'market_id': pos.market_id,
                'market_question': pos.market_question,
                'outcome': pos.outcome,
                'initial_stake': pos.initial_stake,
                'current_value': pos.current_value,
                'unrealized_pnl': pos.unrealized_pnl,
                'pnl_percentage': pos.calculate_pnl_percentage(),
                'weight': pos.current_value / self._calculate_total_value() if self._calculate_total_value() > 0 else 0,
                'days_held': (datetime.now() - pos.opened_at).days,
                'current_probability': pos.current_probability
            })
        
        return positions
    
    def _calculate_risk_metrics(self) -> Dict[str, Any]:
        """Calculate detailed risk metrics"""
        return {
            'value_at_risk_95': self._calculate_var_95(),
            'expected_shortfall': self._calculate_expected_shortfall(),
            'beta_exposure': self._calculate_beta_exposure(),
            'sector_exposure': self._calculate_sector_exposure(),
            'correlation_risk': self._calculate_correlation_risk()
        }
    
    def _calculate_var_95(self) -> float:
        """Calculate Value at Risk at 95% confidence level"""
        # Simplified VaR calculation
        # In reality, would use historical simulation or parametric methods
        volatility = self._calculate_portfolio_volatility()
        portfolio_value = self._calculate_total_value()
        
        # 95% VaR ≈ 1.65 * σ * portfolio_value
        return 1.65 * volatility * portfolio_value
    
    def _calculate_expected_shortfall(self) -> float:
        """Calculate Expected Shortfall (CVaR)"""
        # Simplified Expected Shortfall
        # In reality, would use more sophisticated methods
        var_95 = self._calculate_var_95()
        return var_95 * 1.2  # Approximation
    
    def _calculate_beta_exposure(self) -> float:
        """Calculate portfolio beta exposure"""
        # Simplified beta calculation
        # In reality, would calculate against market benchmark
        return 1.0  # Placeholder
    
    def _calculate_sector_exposure(self) -> Dict[str, float]:
        """Calculate sector exposure"""
        # Simplified sector exposure
        # In reality, would categorize markets by sector
        return {'technology': 0.3, 'finance': 0.2, 'other': 0.5}  # Placeholder
    
    def _calculate_correlation_risk(self) -> float:
        """Calculate correlation risk"""
        # Simplified correlation risk
        # In reality, would calculate actual correlations
        return 0.5  # Placeholder
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary"""
        return {
            'current_balance': self.current_balance,
            'total_value': self._calculate_total_value(),
            'total_pnl': self._calculate_total_pnl(),
            'total_return': self._calculate_total_return(),
            'active_positions': len(self.active_positions),
            'portfolio_metrics': self.portfolio_metrics,
            'agent_contributions': self.agent_contributions,
            'last_updated': datetime.now().isoformat()
        }
    
    async def handle_request(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle portfolio management requests"""
        if message.message_type == "REQUEST":
            content = message.content
            
            if 'portfolio_summary' in content:
                # Return portfolio summary
                summary = self.get_portfolio_summary()
                
                return AgentMessage(
                    sender_agent=self.agent_name,
                    receiver_agent=message.sender_agent,
                    message_type="RESPONSE",
                    content={'portfolio_summary': summary},
                    priority=2
                )
            
            elif 'performance_report' in content:
                # Generate performance report
                report = await self._generate_performance_report()
                
                return AgentMessage(
                    sender_agent=self.agent_name,
                    receiver_agent=message.sender_agent,
                    message_type="RESPONSE",
                    content={'performance_report': report},
                    priority=3
                )
            
            elif 'rebalancing_recommendations' in content:
                # Get rebalancing recommendations
                recommendations = await self._analyze_rebalancing_needs()
                
                return AgentMessage(
                    sender_agent=self.agent_name,
                    receiver_agent=message.sender_agent,
                    message_type="RESPONSE",
                    content={'rebalancing_recommendations': recommendations},
                    priority=2
                )
        
        return None