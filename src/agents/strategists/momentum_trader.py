"""
Momentum Trader Strategy Agent
"""
from typing import Dict, Any, List, Tuple
import statistics
from datetime import datetime, timedelta

from src.agents.base_agent import StrategyAgent
from src.models import Market, TradingAction
from src.utils.config import config
from src.utils.logger import get_logger

log = get_logger(__name__)


class MomentumTrader(StrategyAgent):
    """Identifies strong trends and momentum signals, trades in direction of established trends"""
    
    def __init__(self):
        super().__init__("Momentum Trader", "momentum_trading")
        self.min_momentum_threshold = 0.3  # Minimum momentum score to trade
        self.price_history: Dict[str, List[Tuple[datetime, float]]] = {}
        
    def get_default_prompt(self) -> str:
        return config.get_agent_prompt('momentum_trader') or """
        You are a momentum trader for prediction markets specializing in trend following.
        
        Strategy:
        1. Identify strong price trends and momentum
        2. Trade in direction of established momentum
        3. Use technical indicators for entry/exit timing
        4. Scale into winning positions
        5. Cut losses quickly on trend reversals
        
        Provide:
        - Momentum score: 0.0 to 1.0
        - Trend direction and strength
        - Entry signal strength
        - Recommended position size: 0-10% of capital
        - Exit strategy
        
        Focus on markets showing clear momentum and ride trends for maximum profit.
        """
    
    async def evaluate_opportunity(self, market: Market, analysis_data: Dict[str, float]) -> Dict[str, Any]:
        """Evaluate momentum trading opportunity"""
        
        # Get analysis scores
        fundamental_score = analysis_data.get('fundamental_score', 0.5)
        technical_score = analysis_data.get('technical_score', 0.5)
        sentiment_score = analysis_data.get('sentiment_score', 0.5)
        
        # Update price history
        self._update_price_history(market)
        
        # Calculate momentum indicators
        momentum_score = self._calculate_momentum_score(market)
        trend_direction = self._identify_trend_direction(market)
        trend_strength = self._calculate_trend_strength(market)
        
        # Volume confirmation
        volume_confirmation = self._analyze_volume_momentum(market)
        
        # Overall momentum signal
        overall_momentum = (momentum_score + trend_strength + volume_confirmation) / 3
        
        # Determine trading action
        if overall_momentum < self.min_momentum_threshold:
            action = TradingAction.HOLD
            thesis = f"Weak momentum ({overall_momentum:.2f} < {self.min_momentum_threshold})"
            confidence = 0.3
        elif trend_direction == "bullish":
            action = TradingAction.BUY
            thesis = f"Strong bullish momentum (score: {overall_momentum:.2f})"
            confidence = min(0.9, overall_momentum)
        elif trend_direction == "bearish":
            action = TradingAction.SELL
            thesis = f"Strong bearish momentum (score: {overall_momentum:.2f})"
            confidence = min(0.9, overall_momentum)
        else:
            action = TradingAction.HOLD
            thesis = f"No clear trend (sideways momentum)"
            confidence = 0.4
        
        # Calculate position size based on momentum strength
        position_size = self._calculate_momentum_position_size(overall_momentum, confidence, technical_score)
        
        # Determine outcome
        outcome = "YES" if trend_direction == "bullish" else "NO"
        
        # Exit strategy
        exit_strategy = self._determine_exit_strategy(market, trend_direction, trend_strength)
        
        return {
            "action": action,
            "outcome": outcome,
            "confidence": confidence,
            "reasoning": thesis,
            "estimated_probability": self._estimate_momentum_probability(market, trend_direction),
            "position_size": position_size,
            "momentum_score": overall_momentum,
            "trend_direction": trend_direction,
            "trend_strength": trend_strength,
            "volume_confirmation": volume_confirmation,
            "exit_strategy": exit_strategy,
            "entry_timing": self._assess_entry_timing(market),
            "risk_factors": self._identify_momentum_risks(market)
        }
    
    def _update_price_history(self, market: Market):
        """Update price history for momentum analysis"""
        now = datetime.now()
        
        if market.id not in self.price_history:
            self.price_history[market.id] = []
        
        # Add current price point
        self.price_history[market.id].append((now, market.probability))
        
        # Keep only last 50 data points
        if len(self.price_history[market.id]) > 50:
            self.price_history[market.id] = self.price_history[market.id][-50:]
    
    def _calculate_momentum_score(self, market: Market) -> float:
        """Calculate momentum score based on price changes"""
        if market.id not in self.price_history or len(self.price_history[market.id]) < 3:
            return 0.3  # Default score for insufficient data
        
        prices = [price for _, price in self.price_history[market.id][-10:]]
        
        if len(prices) < 3:
            return 0.3
        
        # Calculate recent price changes
        short_term_change = prices[-1] - prices[-3] if len(prices) >= 3 else 0
        medium_term_change = prices[-1] - prices[-5] if len(prices) >= 5 else 0
        long_term_change = prices[-1] - prices[-10] if len(prices) >= 10 else 0
        
        # Weight changes (more weight on recent changes)
        weighted_change = (short_term_change * 0.5 + medium_term_change * 0.3 + long_term_change * 0.2)
        
        # Calculate momentum score (0-1)
        momentum_score = min(1.0, abs(weighted_change) * 5)  # Scale to 0-1
        
        return momentum_score
    
    def _identify_trend_direction(self, market: Market) -> str:
        """Identify trend direction"""
        if market.id not in self.price_history or len(self.price_history[market.id]) < 5:
            return "neutral"
        
        prices = [price for _, price in self.price_history[market.id][-10:]]
        
        if len(prices) < 5:
            return "neutral"
        
        # Calculate moving averages
        short_ma = statistics.mean(prices[-3:])  # 3-period MA
        long_ma = statistics.mean(prices[-10:]) if len(prices) >= 10 else statistics.mean(prices)
        
        # Determine trend based on moving averages and recent price action
        if short_ma > long_ma and prices[-1] > short_ma:
            return "bullish"
        elif short_ma < long_ma and prices[-1] < short_ma:
            return "bearish"
        else:
            return "neutral"
    
    def _calculate_trend_strength(self, market: Market) -> float:
        """Calculate trend strength"""
        if market.id not in self.price_history or len(self.price_history[market.id]) < 5:
            return 0.3
        
        prices = [price for _, price in self.price_history[market.id][-20:]]
        
        if len(prices) < 5:
            return 0.3
        
        # Calculate price variance (proxy for trend strength)
        price_variance = statistics.variance(prices) if len(prices) > 1 else 0
        
        # Calculate trend consistency
        trend_directions = []
        for i in range(1, len(prices)):
            if prices[i] > prices[i-1]:
                trend_directions.append(1)
            elif prices[i] < prices[i-1]:
                trend_directions.append(-1)
            else:
                trend_directions.append(0)
        
        # Consistency is high when most changes are in same direction
        if trend_directions:
            consistency = abs(statistics.mean(trend_directions))
        else:
            consistency = 0
        
        # Combine variance and consistency
        trend_strength = (min(price_variance * 10, 0.5) + consistency) / 2
        
        return min(1.0, max(0.0, trend_strength))
    
    def _analyze_volume_momentum(self, market: Market) -> float:
        """Analyze volume as momentum confirmation"""
        # In a real implementation, would analyze volume history
        # For now, use current volume as proxy
        
        volume = market.volume
        
        # Volume scoring
        if volume >= 1000:
            return 1.0  # Very strong volume confirmation
        elif volume >= 500:
            return 0.8
        elif volume >= 200:
            return 0.6
        elif volume >= 100:
            return 0.4
        else:
            return 0.2  # Weak volume confirmation
    
    def _calculate_momentum_position_size(self, momentum_score: float, confidence: float, 
                                        technical_score: float) -> float:
        """Calculate position size based on momentum strength"""
        
        # Base position size on momentum strength
        base_size = momentum_score * 0.15  # Max 15% for strongest momentum
        
        # Adjust for confidence and technical confirmation
        confidence_multiplier = confidence
        technical_multiplier = technical_score
        
        # Calculate final position size
        position_size = base_size * confidence_multiplier * technical_multiplier
        
        # Apply limits
        max_position = config.get('trading.max_position_size', 0.10)
        position_size = min(position_size, max_position)
        
        # Minimum position for meaningful momentum trades
        min_position = 0.02 if momentum_score > 0.5 else 0.01
        position_size = max(position_size, min_position if momentum_score > 0.4 else 0)
        
        return position_size
    
    def _estimate_momentum_probability(self, market: Market, trend_direction: str) -> float:
        """Estimate probability based on momentum"""
        current_prob = market.probability
        
        if trend_direction == "bullish":
            # Bullish momentum suggests higher probability
            return min(0.95, current_prob + 0.1)
        elif trend_direction == "bearish":
            # Bearish momentum suggests lower probability
            return max(0.05, current_prob - 0.1)
        else:
            # Neutral trend - probability stays similar
            return current_prob
    
    def _determine_exit_strategy(self, market: Market, trend_direction: str, 
                                trend_strength: float) -> str:
        """Determine exit strategy for momentum trade"""
        
        if trend_direction == "neutral":
            return "No position - no clear trend"
        
        # Base exit strategy on trend strength
        if trend_strength >= 0.8:
            return f"Strong {trend_direction} trend - hold until momentum weakens or 25% profit target"
        elif trend_strength >= 0.6:
            return f"Moderate {trend_direction} trend - tight stop loss, 15% profit target"
        else:
            return f"Weak {trend_direction} trend - quick exit, 10% profit target"
    
    def _assess_entry_timing(self, market: Market) -> str:
        """Assess timing for market entry"""
        
        # Check if market is at extreme (potential reversal point)
        current_prob = market.probability
        
        if current_prob >= 0.95:
            return "Poor timing - market at extreme high, potential reversal"
        elif current_prob <= 0.05:
            return "Poor timing - market at extreme low, potential reversal"
        elif 0.2 <= current_prob <= 0.8:
            return "Good timing - market in normal range"
        else:
            return "Moderate timing - market approaching extremes"
    
    def _identify_momentum_risks(self, market: Market) -> List[str]:
        """Identify risks specific to momentum trading"""
        risks = []
        
        # Low volume risk
        if market.volume < 100:
            risks.append("Low volume - momentum may not be sustainable")
        
        # Time to close risk
        if market.close_time:
            days_to_close = (market.close_time - datetime.now()).days
            if days_to_close < 3:
                risks.append("Very close to resolution - momentum may reverse quickly")
            elif days_to_close < 7:
                risks.append("Close to resolution - reduced time for momentum to play out")
        
        # Extreme probability risk
        if market.probability >= 0.9 or market.probability <= 0.1:
            risks.append("Extreme probability - high reversal risk")
        
        # Sudden news risk
        if not market.description or len(market.description) < 50:
            risks.append("Limited information - higher surprise risk")
        
        return risks
    
    def calculate_momentum_indicators(self, market: Market) -> Dict[str, float]:
        """Calculate detailed momentum indicators"""
        if market.id not in self.price_history or len(self.price_history[market.id]) < 5:
            return {"rsi": 0.5, "macd": 0.0, "rate_of_change": 0.0}
        
        prices = [price for _, price in self.price_history[market.id][-20:]]
        
        # RSI (Relative Strength Index) - simplified
        if len(prices) >= 14:
            gains = []
            losses = []
            for i in range(1, len(prices)):
                change = prices[i] - prices[i-1]
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(change))
            
            avg_gain = statistics.mean(gains[-14:]) if gains else 0
            avg_loss = statistics.mean(losses[-14:]) if losses else 0
            
            if avg_loss == 0:
                rsi = 1.0
            else:
                rs = avg_gain / avg_loss
                rsi = 1 - (1 / (1 + rs))
        else:
            rsi = 0.5
        
        # Rate of Change
        if len(prices) >= 5:
            roc = (prices[-1] - prices[-5]) / prices[-5]
        else:
            roc = 0.0
        
        # MACD (simplified)
        if len(prices) >= 12:
            short_ma = statistics.mean(prices[-5:])
            long_ma = statistics.mean(prices[-12:])
            macd = short_ma - long_ma
        else:
            macd = 0.0
        
        return {
            "rsi": rsi,
            "macd": macd,
            "rate_of_change": roc
        }
    
    async def generate_momentum_thesis(self, market: Market, opportunity: Dict[str, Any]) -> str:
        """Generate detailed momentum trading thesis"""
        action = opportunity['action']
        momentum_score = opportunity['momentum_score']
        trend_direction = opportunity['trend_direction']
        trend_strength = opportunity['trend_strength']
        
        thesis_parts = []
        
        if action == TradingAction.BUY:
            thesis_parts.append(f"BUY: Strong {trend_direction} momentum detected")
        elif action == TradingAction.SELL:
            thesis_parts.append(f"SELL: Strong {trend_direction} momentum detected")
        else:
            thesis_parts.append(f"HOLD: Insufficient momentum (score: {momentum_score:.2f})")
        
        # Add momentum characteristics
        thesis_parts.append(f"Momentum score: {momentum_score:.2f}, Trend strength: {trend_strength:.2f}")
        
        # Add volume confirmation
        volume_conf = opportunity['volume_confirmation']
        if volume_conf >= 0.7:
            thesis_parts.append("Strong volume confirmation")
        elif volume_conf >= 0.5:
            thesis_parts.append("Moderate volume confirmation")
        else:
            thesis_parts.append("Weak volume confirmation - caution")
        
        # Add exit strategy
        thesis_parts.append(f"Exit plan: {opportunity['exit_strategy']}")
        
        # Add risk assessment
        risks = opportunity['risk_factors']
        if risks:
            thesis_parts.append(f"Risks: {', '.join(risks[:2])}")
        
        return " | ".join(thesis_parts)