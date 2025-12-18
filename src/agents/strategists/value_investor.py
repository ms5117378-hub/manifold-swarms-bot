"""
Value Investor Strategy Agent
"""
from typing import Dict, Any
import statistics

from src.agents.base_agent import StrategyAgent
from src.models import Market, TradingAction
from src.utils.config import config
from src.utils.logger import get_logger

log = get_logger(__name__)


class ValueInvestor(StrategyAgent):
    """Finds mispriced markets by comparing current probability with estimated true probability"""
    
    def __init__(self):
        super().__init__("Value Investor", "value_investing")
        self.min_mispricing_threshold = 0.15  # 15% minimum mispricing
        
    def get_default_prompt(self) -> str:
        return config.get_agent_prompt('value_investor') or """
        You are a value investor for prediction markets specializing in finding mispriced opportunities.
        
        Strategy:
        1. Compare current probability with estimated true probability
        2. Identify markets with significant mispricing (>15% difference)
        3. Focus on fundamental value and long-term resolution
        4. Patient approach waiting for the right opportunities
        5. Size positions based on conviction level
        
        Provide:
        - Estimated true probability: 0.0 to 1.0
        - Mispricing percentage
        - Investment thesis
        - Recommended position size: 0-10% of capital
        - Time horizon for resolution
        
        Look for markets where the crowd is wrong and fundamental analysis reveals true value.
        """
    
    async def evaluate_opportunity(self, market: Market, analysis_data: Dict[str, float]) -> Dict[str, Any]:
        """Evaluate value investing opportunity"""
        
        # Get analysis scores from other agents
        fundamental_score = analysis_data.get('fundamental_score', 0.5)
        technical_score = analysis_data.get('technical_score', 0.5)
        sentiment_score = analysis_data.get('sentiment_score', 0.5)
        
        # Calculate estimated true probability
        true_probability = self._calculate_true_probability(market, analysis_data)
        
        # Calculate mispricing
        current_prob = market.probability
        mispricing = abs(true_probability - current_prob)
        mispricing_percent = mispricing / max(current_prob, 1 - current_prob, 0.01)
        
        # Determine investment action
        if mispricing_percent < self.min_mispricing_threshold:
            action = TradingAction.HOLD
            thesis = f"Insufficient mispricing ({mispricing_percent:.1%} < {self.min_mispricing_threshold:.1%})"
            confidence = 0.3
        elif true_probability > current_prob:
            action = TradingAction.BUY
            thesis = f"Undervalued: True prob {true_probability:.2f} > Market {current_prob:.2f}"
            confidence = min(0.9, mispricing_percent * 2)  # Higher confidence for larger mispricing
        else:
            action = TradingAction.SELL
            thesis = f"Overvalued: True prob {true_probability:.2f} < Market {current_prob:.2f}"
            confidence = min(0.9, mispricing_percent * 2)
        
        # Calculate position size based on conviction
        position_size = self._calculate_position_size(mispricing_percent, confidence, fundamental_score)
        
        # Determine outcome (YES/NO based on true probability)
        outcome = "YES" if true_probability > 0.5 else "NO"
        
        return {
            "action": action,
            "outcome": outcome,
            "confidence": confidence,
            "reasoning": thesis,
            "estimated_probability": true_probability,
            "position_size": position_size,
            "mispricing_percent": mispricing_percent,
            "time_horizon": self._estimate_time_horizon(market),
            "fundamental_strength": fundamental_score,
            "margin_of_safety": mispricing_percent
        }
    
    def _calculate_true_probability(self, market: Market, analysis_data: Dict[str, float]) -> float:
        """Calculate estimated true probability using fundamental analysis"""
        
        # Base probability from market
        base_prob = market.probability
        
        # Adjust based on fundamental analysis
        fundamental_score = analysis_data.get('fundamental_score', 0.5)
        technical_score = analysis_data.get('technical_score', 0.5)
        sentiment_score = analysis_data.get('sentiment_score', 0.5)
        
        # Weight fundamental analysis more heavily for value investing
        weights = {
            'fundamental': 0.5,
            'technical': 0.2,
            'sentiment': 0.2,
            'current': 0.1
        }
        
        # Calculate adjustments
        fundamental_adjustment = (fundamental_score - 0.5) * 0.4  # Max 20% adjustment
        technical_adjustment = (technical_score - 0.5) * 0.2      # Max 10% adjustment
        sentiment_adjustment = (sentiment_score - 0.5) * 0.2     # Max 10% adjustment
        
        # Apply adjustments
        adjusted_prob = base_prob + fundamental_adjustment + technical_adjustment + sentiment_adjustment
        
        # Consider market efficiency factors
        efficiency_adjustment = self._calculate_efficiency_adjustment(market)
        adjusted_prob += efficiency_adjustment
        
        # Ensure probability stays within bounds
        return max(0.05, min(0.95, adjusted_prob))
    
    def _calculate_efficiency_adjustment(self, market: Market) -> float:
        """Calculate adjustment based on market efficiency"""
        adjustment = 0.0
        
        # Low volume markets may be less efficient
        if market.volume < 100:
            adjustment += 0.05  # Bias toward mean reversion
        elif market.volume < 500:
            adjustment += 0.02
        
        # Markets with few traders may be less efficient
        pool = market.pool
        if pool:
            total_pool = sum(pool.values())
            if total_pool < 200:
                adjustment += 0.03
            elif total_pool < 500:
                adjustment += 0.01
        
        # New markets may be less efficiently priced
        market_age_days = (market.created_time - market.created_time).days
        if market_age_days < 7:
            adjustment += 0.02
        
        return adjustment
    
    def _calculate_position_size(self, mispricing_percent: float, confidence: float, 
                                fundamental_score: float) -> float:
        """Calculate optimal position size using Kelly criterion with safety factors"""
        
        # Kelly criterion: f = (bp - q) / b
        # where b = odds, p = probability of winning, q = probability of losing
        
        # Simplified Kelly using mispricing as edge
        edge = mispricing_percent
        odds = 1.0  # Binary markets have roughly 1:1 odds
        
        # Conservative Kelly (quarter Kelly for safety)
        kelly_fraction = (edge * odds - (1 - edge)) / odds * 0.25
        
        # Adjust for confidence and fundamental strength
        confidence_multiplier = confidence
        fundamental_multiplier = fundamental_score
        
        # Calculate final position size
        position_size = kelly_fraction * confidence_multiplier * fundamental_multiplier
        
        # Apply maximum position size limit
        max_position = config.get('trading.max_position_size', 0.10)
        position_size = min(position_size, max_position)
        
        # Minimum position size for meaningful trades
        min_position = 0.01  # 1% minimum
        position_size = max(position_size, min_position if edge > 0.2 else 0)
        
        return position_size
    
    def _estimate_time_horizon(self, market: Market) -> str:
        """Estimate optimal holding period"""
        if not market.close_time:
            return "Long-term (no close date)"
        
        days_to_close = (market.close_time - market.created_time).days
        
        if days_to_close <= 7:
            return "Very short-term (1 week)"
        elif days_to_close <= 30:
            return "Short-term (1 month)"
        elif days_to_close <= 90:
            return "Medium-term (3 months)"
        elif days_to_close <= 365:
            return "Long-term (1 year)"
        else:
            return "Very long-term (1+ year)"
    
    def calculate_margin_of_safety(self, market: Market, true_probability: float) -> float:
        """Calculate margin of safety as percentage"""
        current_prob = market.probability
        mispricing = abs(true_probability - current_prob)
        
        # Margin of safety relative to current price
        if true_probability > current_prob:
            return mispricing / current_prob
        else:
            return mispricing / (1 - current_prob)
    
    def identify_value_traps(self, market: Market, analysis_data: Dict[str, float]) -> bool:
        """Identify potential value traps"""
        
        fundamental_score = analysis_data.get('fundamental_score', 0.5)
        technical_score = analysis_data.get('technical_score', 0.5)
        
        # Red flags for value traps
        red_flags = []
        
        # Poor fundamentals
        if fundamental_score < 0.3:
            red_flags.append("Poor fundamentals")
        
        # Deteriorating technicals
        if technical_score < 0.3:
            red_flags.append("Poor technicals")
        
        # Very low volume (possible illiquidity trap)
        if market.volume < 50:
            red_flags.append("Very low volume")
        
        # Extreme probabilities (possible crowd madness)
        if market.probability >= 0.95 or market.probability <= 0.05:
            red_flags.append("Extreme probability")
        
        # Ambiguous resolution criteria
        if not market.description or len(market.description) < 50:
            red_flags.append("Unclear resolution")
        
        return len(red_flags) >= 2  # Value trap if 2+ red flags
    
    async def generate_investment_thesis(self, market: Market, opportunity: Dict[str, Any]) -> str:
        """Generate detailed investment thesis"""
        action = opportunity['action']
        mispricing = opportunity['mispricing_percent']
        confidence = opportunity['confidence']
        fundamental_strength = opportunity['fundamental_strength']
        
        thesis_parts = []
        
        if action == TradingAction.BUY:
            thesis_parts.append(f"BUY recommendation: Market appears undervalued by {mispricing:.1%}")
        elif action == TradingAction.SELL:
            thesis_parts.append(f"SELL recommendation: Market appears overvalued by {mispricing:.1%}")
        else:
            thesis_parts.append(f"HOLD: Insufficient mispricing ({mispricing:.1%})")
        
        # Add fundamental assessment
        if fundamental_strength >= 0.7:
            thesis_parts.append("Strong fundamental support")
        elif fundamental_strength >= 0.5:
            thesis_parts.append("Adequate fundamentals")
        else:
            thesis_parts.append("Weak fundamentals - caution advised")
        
        # Add confidence rationale
        if confidence >= 0.7:
            thesis_parts.append("High conviction based on significant mispricing")
        elif confidence >= 0.5:
            thesis_parts.append("Moderate conviction")
        else:
            thesis_parts.append("Low conviction - prefer to wait")
        
        # Add time horizon
        thesis_parts.append(f"Time horizon: {opportunity['time_horizon']}")
        
        # Risk factors
        if self.identify_value_traps(market, {'fundamental_score': fundamental_strength}):
            thesis_parts.append("⚠️ Potential value trap - additional due diligence required")
        
        return " | ".join(thesis_parts)