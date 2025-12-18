"""
Mean Reversion Trader Strategy Agent
"""
from typing import Dict, Any, List, Tuple
import statistics
from datetime import datetime, timedelta

from src.agents.base_agent import StrategyAgent
from src.models import Market, TradingAction
from src.utils.config import config
from src.utils.logger import get_logger

log = get_logger(__name__)


class MeanReversionTrader(StrategyAgent):
    """Finds markets that have moved too far from mean, takes contrarian positions on overreactions"""
    
    def __init__(self):
        super().__init__("Mean Reversion Trader", "mean_reversion_trading")
        self.min_deviation_threshold = 0.15  # 15% minimum deviation from mean
        self.price_history: Dict[str, List[Tuple[datetime, float]]] = {}
        self.probability_means: Dict[str, float] = {}
        
    def get_default_prompt(self) -> str:
        return config.get_agent_prompt('mean_reversion') or """
        You are a mean reversion trader for prediction markets specializing in contrarian opportunities.
        
        Strategy:
        1. Identify markets that have moved too far from fair value
        2. Take contrarian positions on overreactions
        3. Use statistical measures of deviation
        4. Patient waiting for reversion to mean
        5. Risk management through position sizing
        
        Provide:
        - Deviation from mean percentage
        - Reversion probability: 0.0 to 1.0
        - Contrarian signal strength
        - Recommended position size: 0-10% of capital
        - Expected reversion timeframe
        
        Look for extreme movements and bet on rationality returning to the market.
        """
    
    async def evaluate_opportunity(self, market: Market, analysis_data: Dict[str, float]) -> Dict[str, Any]:
        """Evaluate mean reversion opportunity"""
        
        # Get analysis scores
        fundamental_score = analysis_data.get('fundamental_score', 0.5)
        technical_score = analysis_data.get('technical_score', 0.5)
        sentiment_score = analysis_data.get('sentiment_score', 0.5)
        
        # Update price history and calculate statistics
        self._update_price_history(market)
        self._update_probability_mean(market)
        
        # Calculate mean reversion metrics
        current_prob = market.probability
        historical_mean = self.probability_means.get(market.id, 0.5)
        deviation = current_prob - historical_mean
        deviation_percent = abs(deviation) / max(historical_mean, 1 - historical_mean, 0.01)
        
        # Calculate z-score (how many standard deviations from mean)
        z_score = self._calculate_z_score(market)
        
        # Calculate reversion probability
        reversion_probability = self._calculate_reversion_probability(z_score, deviation_percent)
        
        # Identify overreaction type
        overreaction_type = self._identify_overreaction_type(deviation, z_score)
        
        # Determine trading action
        if deviation_percent < self.min_deviation_threshold:
            action = TradingAction.HOLD
            thesis = f"Insufficient deviation ({deviation_percent:.1%} < {self.min_deviation_threshold:.1%})"
            confidence = 0.3
        elif deviation > 0:  # Market above mean - expect reversion down
            action = TradingAction.SELL
            thesis = f"Overreaction to upside: {deviation_percent:.1%} above mean"
            confidence = min(0.9, reversion_probability)
        else:  # Market below mean - expect reversion up
            action = TradingAction.BUY
            thesis = f"Overreaction to downside: {deviation_percent:.1%} below mean"
            confidence = min(0.9, reversion_probability)
        
        # Calculate position size based on deviation magnitude
        position_size = self._calculate_mean_reversion_position_size(
            deviation_percent, z_score, confidence, fundamental_score
        )
        
        # Determine outcome
        outcome = "YES" if action == TradingAction.BUY else "NO"
        
        # Expected reversion timeframe
        reversion_timeframe = self._estimate_reversion_timeframe(market, z_score)
        
        # Risk assessment
        risk_factors = self._assess_mean_reversion_risks(market, z_score, deviation_percent)
        
        return {
            "action": action,
            "outcome": outcome,
            "confidence": confidence,
            "reasoning": thesis,
            "estimated_probability": historical_mean,  # Expect reversion to mean
            "position_size": position_size,
            "deviation_percent": deviation_percent,
            "z_score": z_score,
            "historical_mean": historical_mean,
            "reversion_probability": reversion_probability,
            "overreaction_type": overreaction_type,
            "reversion_timeframe": reversion_timeframe,
            "risk_factors": risk_factors,
            "contrarian_strength": self._calculate_contrarian_strength(z_score)
        }
    
    def _update_price_history(self, market: Market):
        """Update price history for mean reversion analysis"""
        now = datetime.now()
        
        if market.id not in self.price_history:
            self.price_history[market.id] = []
        
        # Add current price point
        self.price_history[market.id].append((now, market.probability))
        
        # Keep only last 100 data points for mean calculation
        if len(self.price_history[market.id]) > 100:
            self.price_history[market.id] = self.price_history[market.id][-100:]
    
    def _update_probability_mean(self, market: Market):
        """Update historical mean probability"""
        if market.id not in self.price_history or len(self.price_history[market.id]) < 5:
            # Use current probability as mean for new markets
            self.probability_means[market.id] = market.probability
            return
        
        prices = [price for _, price in self.price_history[market.id]]
        
        # Calculate exponential moving average for more responsive mean
        if len(prices) >= 2:
            alpha = 0.1  # Smoothing factor
            current_mean = self.probability_means.get(market.id, prices[0])
            new_mean = alpha * prices[-1] + (1 - alpha) * current_mean
            self.probability_means[market.id] = new_mean
        else:
            self.probability_means[market.id] = statistics.mean(prices)
    
    def _calculate_z_score(self, market: Market) -> float:
        """Calculate z-score (standard deviations from mean)"""
        if market.id not in self.price_history or len(self.price_history[market.id]) < 5:
            return 0.0
        
        prices = [price for _, price in self.price_history[market.id][-20:]]
        
        if len(prices) < 5:
            return 0.0
        
        mean_price = statistics.mean(prices)
        std_dev = statistics.stdev(prices) if len(prices) > 1 else 0.01
        
        if std_dev == 0:
            return 0.0
        
        current_price = market.probability
        z_score = (current_price - mean_price) / std_dev
        
        return z_score
    
    def _calculate_reversion_probability(self, z_score: float, deviation_percent: float) -> float:
        """Calculate probability of mean reversion"""
        # Higher z-scores and larger deviations increase reversion probability
        z_score_factor = min(abs(z_score) / 3, 1.0)  # Cap at 3 standard deviations
        deviation_factor = min(deviation_percent / 0.5, 1.0)  # Cap at 50% deviation
        
        # Combine factors
        reversion_prob = (z_score_factor * 0.6 + deviation_factor * 0.4)
        
        # Base probability adjustment (markets tend to revert)
        base_prob = 0.6
        
        final_prob = base_prob + (reversion_prob * 0.3)
        
        return min(0.95, max(0.3, final_prob))
    
    def _identify_overreaction_type(self, deviation: float, z_score: float) -> str:
        """Identify type of overreaction"""
        abs_z = abs(z_score)
        
        if abs_z >= 3:
            return "Extreme overreaction"
        elif abs_z >= 2:
            return "Strong overreaction"
        elif abs_z >= 1.5:
            return "Moderate overreaction"
        elif abs_z >= 1:
            return "Mild overreaction"
        else:
            return "Normal fluctuation"
    
    def _calculate_mean_reversion_position_size(self, deviation_percent: float, z_score: float,
                                               confidence: float, fundamental_score: float) -> float:
        """Calculate position size based on mean reversion opportunity"""
        
        # Base position on deviation magnitude
        deviation_factor = min(deviation_percent / 0.3, 1.0)  # Cap at 30% deviation
        
        # Z-score factor (higher z-score = higher conviction)
        z_score_factor = min(abs(z_score) / 2.5, 1.0)
        
        # Base position size
        base_size = 0.08  # 8% base size for strong mean reversion
        
        # Adjust by factors
        adjusted_size = base_size * deviation_factor * z_score_factor * confidence
        
        # Fundamental adjustment (avoid mean reversion on fundamentally broken markets)
        fundamental_multiplier = max(0.3, fundamental_score)  # Minimum 30% if fundamentals are poor
        
        final_size = adjusted_size * fundamental_multiplier
        
        # Apply limits
        max_position = config.get('trading.max_position_size', 0.10)
        final_size = min(final_size, max_position)
        
        # Minimum position for significant deviations
        min_position = 0.02 if abs(z_score) > 1.5 else 0.01
        final_size = max(final_size, min_position if deviation_percent > 0.2 else 0)
        
        return final_size
    
    def _estimate_reversion_timeframe(self, market: Market, z_score: float) -> str:
        """Estimate timeframe for mean reversion"""
        abs_z = abs(z_score)
        
        # Higher z-scores tend to revert faster (more extreme mispricing)
        if abs_z >= 3:
            return "Very short-term (1-3 days)"
        elif abs_z >= 2:
            return "Short-term (3-7 days)"
        elif abs_z >= 1.5:
            return "Medium-term (1-2 weeks)"
        else:
            return "Longer-term (2-4 weeks)"
    
    def _assess_mean_reversion_risks(self, market: Market, z_score: float, 
                                    deviation_percent: float) -> List[str]:
        """Assess risks specific to mean reversion trading"""
        risks = []
        
        # Trend continuation risk
        if abs(z_score) < 1:
            risks.append("Low deviation - may not revert")
        
        # Structural change risk
        if market.close_time:
            days_to_close = (market.close_time - datetime.now()).days
            if days_to_close > 90:
                risks.append("Long timeframe - fundamentals may change")
        
        # Low volume risk
        if market.volume < 100:
            risks.append("Low volume - reversion may be slow")
        
        # Extreme probability risk
        if market.probability >= 0.95 or market.probability <= 0.05:
            risks.append("Extreme probability - potential new information")
        
        # One-way market risk
        pool = market.pool
        if pool:
            yes_pool = pool.get('YES', 0)
            no_pool = pool.get('NO', 0)
            total_pool = yes_pool + no_pool
            
            if total_pool > 0:
                balance_ratio = min(yes_pool, no_pool) / max(yes_pool, no_pool)
                if balance_ratio < 0.1:
                    risks.append("Highly imbalanced market - may reflect new information")
        
        return risks
    
    def _calculate_contrarian_strength(self, z_score: float) -> float:
        """Calculate strength of contrarian signal"""
        # Higher z-scores indicate stronger contrarian opportunities
        abs_z = abs(z_score)
        
        if abs_z >= 3:
            return 1.0  # Very strong contrarian signal
        elif abs_z >= 2.5:
            return 0.9
        elif abs_z >= 2:
            return 0.8
        elif abs_z >= 1.5:
            return 0.6
        elif abs_z >= 1:
            return 0.4
        else:
            return 0.2  # Weak contrarian signal
    
    def identify_momentum_trap(self, market: Market) -> bool:
        """Identify if market might be in momentum trend rather than mean reversion"""
        if market.id not in self.price_history or len(self.price_history[market.id]) < 10:
            return False
        
        prices = [price for _, price in self.price_history[market.id][-15:]]
        
        # Check for consistent directional movement
        if len(prices) >= 5:
            # Calculate trend consistency
            trend_directions = []
            for i in range(1, len(prices)):
                if prices[i] > prices[i-1]:
                    trend_directions.append(1)
                elif prices[i] < prices[i-1]:
                    trend_directions.append(-1)
                else:
                    trend_directions.append(0)
            
            # If 80%+ of changes are in same direction, it's momentum not mean reversion
            if trend_directions:
                consistency = abs(statistics.mean(trend_directions))
                return consistency >= 0.8
        
        return False
    
    def calculate_mean_reversion_metrics(self, market: Market) -> Dict[str, float]:
        """Calculate detailed mean reversion metrics"""
        current_prob = market.probability
        historical_mean = self.probability_means.get(market.id, 0.5)
        z_score = self._calculate_z_score(market)
        
        # Calculate deviation metrics
        absolute_deviation = abs(current_prob - historical_mean)
        percent_deviation = absolute_deviation / max(historical_mean, 1 - historical_mean, 0.01)
        
        # Calculate reversion target
        reversion_target = historical_mean
        
        # Potential profit if reversion occurs
        if current_prob > historical_mean:
            potential_profit = (current_prob - reversion_target) / current_prob
        else:
            potential_profit = (reversion_target - current_prob) / (1 - current_prob)
        
        return {
            "current_probability": current_prob,
            "historical_mean": historical_mean,
            "absolute_deviation": absolute_deviation,
            "percent_deviation": percent_deviation,
            "z_score": z_score,
            "reversion_target": reversion_target,
            "potential_profit": potential_profit,
            "contrarian_strength": self._calculate_contrarian_strength(z_score)
        }
    
    async def generate_mean_reversion_thesis(self, market: Market, opportunity: Dict[str, Any]) -> str:
        """Generate detailed mean reversion thesis"""
        action = opportunity['action']
        deviation_percent = opportunity['deviation_percent']
        z_score = opportunity['z_score']
        overreaction_type = opportunity['overreaction_type']
        
        thesis_parts = []
        
        if action == TradingAction.BUY:
            thesis_parts.append(f"BUY: Market oversold by {deviation_percent:.1%}")
        elif action == TradingAction.SELL:
            thesis_parts.append(f"SELL: Market overbought by {deviation_percent:.1%}")
        else:
            thesis_parts.append(f"HOLD: Insufficient deviation ({deviation_percent:.1%})")
        
        # Add statistical context
        thesis_parts.append(f"{overreaction_type} (Z-score: {z_score:.2f})")
        
        # Add reversion expectation
        reversion_prob = opportunity['reversion_probability']
        thesis_parts.append(f"Reversion probability: {reversion_prob:.1%}")
        
        # Add timeframe
        thesis_parts.append(f"Expected timeframe: {opportunity['reversion_timeframe']}")
        
        # Add risk factors
        risks = opportunity['risk_factors']
        if risks:
            thesis_parts.append(f"Risks: {', '.join(risks[:2])}")
        
        # Check for momentum trap
        if self.identify_momentum_trap(market):
            thesis_parts.append("⚠️ Potential momentum trap - contrarian approach risky")
        
        return " | ".join(thesis_parts)