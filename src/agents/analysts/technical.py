"""
Technical Analyst Agent
"""
from typing import Dict, Any, List, Tuple
import statistics
from datetime import datetime, timedelta

from src.agents.base_agent import AnalystAgent
from src.models import Market
from src.utils.config import config
from src.utils.logger import get_logger

log = get_logger(__name__)


class TechnicalAnalyst(AnalystAgent):
    """Analyzes technical indicators: price patterns, volume, momentum, liquidity"""
    
    def __init__(self):
        super().__init__("Technical Analyst", "technical_analysis")
        # In a real implementation, would store historical price data
        self.price_history: Dict[str, List[Tuple[datetime, float]]] = {}
        self.volume_history: Dict[str, List[Tuple[datetime, float]]] = {}
    
    def get_default_prompt(self) -> str:
        return config.get_agent_prompt('technical') or """
        You are a technical analyst for prediction markets specializing in price action and market dynamics.
        
        Analyze:
        1. Price momentum and trends
        2. Volume patterns and liquidity
        3. Market depth and order flow
        4. Historical price movements
        5. Support and resistance levels
        
        Provide:
        - Technical score: 0.0 to 1.0
        - Momentum indicator: positive, neutral, negative
        - Trend strength: weak, moderate, strong
        - Key technical signals
        - Price targets and stop levels
        
        Focus on identifying profitable trading opportunities through technical analysis.
        """
    
    async def calculate_score(self, market: Market) -> float:
        """Calculate technical analysis score"""
        score_components = {}
        
        # 1. Volume analysis (0-1)
        volume_score = self._analyze_volume(market)
        score_components['volume'] = volume_score
        
        # 2. Liquidity analysis (0-1)
        liquidity_score = self._analyze_liquidity(market)
        score_components['liquidity'] = liquidity_score
        
        # 3. Price momentum (0-1) - simplified since we don't have historical data
        momentum_score = self._analyze_momentum(market)
        score_components['momentum'] = momentum_score
        
        # 4. Market depth (0-1)
        depth_score = self._analyze_market_depth(market)
        score_components['depth'] = depth_score
        
        # 5. Probability stability (0-1)
        stability_score = self._analyze_probability_stability(market)
        score_components['stability'] = stability_score
        
        # 6. Time-based patterns (0-1)
        time_score = self._analyze_time_patterns(market)
        score_components['time_patterns'] = time_score
        
        # Calculate weighted average
        weights = {
            'volume': 0.20,
            'liquidity': 0.20,
            'momentum': 0.20,
            'depth': 0.15,
            'stability': 0.15,
            'time_patterns': 0.10
        }
        
        total_score = sum(score_components[component] * weight 
                         for component, weight in weights.items())
        
        # Store current data point for future analysis
        self._update_history(market)
        
        return max(0.0, min(1.0, total_score))
    
    def _analyze_volume(self, market: Market) -> float:
        """Analyze trading volume"""
        volume = market.volume
        
        # Volume scoring based on thresholds
        if volume >= 1000:  # M$1000+
            return 1.0
        elif volume >= 500:  # M$500+
            return 0.8
        elif volume >= 200:  # M$200+
            return 0.6
        elif volume >= 100:  # M$100+
            return 0.4
        elif volume >= 50:   # M$50+
            return 0.2
        else:
            return 0.1
    
    def _analyze_liquidity(self, market: Market) -> float:
        """Analyze market liquidity through pool metrics"""
        pool = market.pool
        
        if not pool or market.outcome_type != 'BINARY':
            return 0.5  # Default score for non-binary or missing data
        
        # For binary markets, check YES and NO pool balance
        yes_pool = pool.get('YES', 0)
        no_pool = pool.get('NO', 0)
        total_pool = yes_pool + no_pool
        
        if total_pool == 0:
            return 0.1
        
        # Liquidity is better when pools are balanced
        balance_ratio = min(yes_pool, no_pool) / max(yes_pool, no_pool) if max(yes_pool, no_pool) > 0 else 0
        
        # Score based on total pool size and balance
        if total_pool >= 1000 and balance_ratio >= 0.3:
            return 1.0
        elif total_pool >= 500 and balance_ratio >= 0.2:
            return 0.8
        elif total_pool >= 200 and balance_ratio >= 0.15:
            return 0.6
        elif total_pool >= 100 and balance_ratio >= 0.1:
            return 0.4
        else:
            return 0.2
    
    def _analyze_momentum(self, market: Market) -> float:
        """Analyze price momentum - simplified without historical data"""
        # In a real implementation, would analyze historical price changes
        # For now, use probability position as a proxy
        
        probability = market.probability
        
        # Extreme probabilities might indicate strong momentum
        if probability >= 0.9 or probability <= 0.1:
            return 0.8  # Strong momentum at extremes
        elif probability >= 0.75 or probability <= 0.25:
            return 0.6  # Moderate momentum
        elif probability >= 0.6 or probability <= 0.4:
            return 0.4  # Weak momentum
        else:
            return 0.2  # Neutral/no momentum
    
    def _analyze_market_depth(self, market: Market) -> float:
        """Analyze market depth and order book strength"""
        # Simplified depth analysis using pool size
        pool = market.pool
        
        if not pool:
            return 0.3
        
        total_pool = sum(pool.values())
        
        # Depth scoring
        if total_pool >= 2000:
            return 1.0
        elif total_pool >= 1000:
            return 0.8
        elif total_pool >= 500:
            return 0.6
        elif total_pool >= 200:
            return 0.4
        else:
            return 0.2
    
    def _analyze_probability_stability(self, market: Market) -> float:
        """Analyze probability stability - simplified"""
        # In a real implementation, would analyze historical probability variance
        # For now, assume mid-range probabilities are more stable
        
        probability = market.probability
        
        # Mid-range probabilities tend to be more stable
        if 0.3 <= probability <= 0.7:
            return 0.8
        elif 0.2 <= probability <= 0.8:
            return 0.6
        else:
            return 0.4  # Extreme probabilities can be volatile
    
    def _analyze_time_patterns(self, market: Market) -> float:
        """Analyze time-based patterns"""
        if not market.close_time:
            return 0.5
        
        now = datetime.now()
        total_duration = (market.close_time - market.created_time).days
        elapsed = (now - market.created_time).days
        progress = elapsed / total_duration if total_duration > 0 else 0
        
        # Different patterns at different stages
        if progress < 0.1:  # Just opened
            return 0.6  # Early stage, less predictable
        elif progress < 0.3:  # Early stage
            return 0.7
        elif progress < 0.7:  # Mid stage
            return 0.9  # Sweet spot for technical analysis
        elif progress < 0.9:  # Late stage
            return 0.7
        else:  # Very close to close
            return 0.5  # Less reliable technical signals
    
    def _update_history(self, market: Market):
        """Update historical data for future analysis"""
        now = datetime.now()
        
        # Initialize history if needed
        if market.id not in self.price_history:
            self.price_history[market.id] = []
            self.volume_history[market.id] = []
        
        # Add current data point
        self.price_history[market.id].append((now, market.probability))
        self.volume_history[market.id].append((now, market.volume))
        
        # Keep only last 100 data points
        if len(self.price_history[market.id]) > 100:
            self.price_history[market.id] = self.price_history[market.id][-100:]
            self.volume_history[market.id] = self.volume_history[market.id][-100:]
    
    def calculate_momentum_indicator(self, market: Market) -> str:
        """Calculate momentum direction"""
        if market.id not in self.price_history or len(self.price_history[market.id]) < 2:
            return "neutral"
        
        # Get recent price changes
        recent_prices = [price for _, price in self.price_history[market.id][-10:]]
        
        if len(recent_prices) < 2:
            return "neutral"
        
        # Calculate trend
        if len(recent_prices) >= 3:
            recent_change = recent_prices[-1] - recent_prices[-2]
            avg_change = statistics.mean([recent_prices[i] - recent_prices[i-1] 
                                        for i in range(1, len(recent_prices))])
            
            if recent_change > avg_change * 1.2:
                return "positive"
            elif recent_change < avg_change * 0.8:
                return "negative"
        
        return "neutral"
    
    def calculate_trend_strength(self, market: Market) -> str:
        """Calculate trend strength"""
        if market.id not in self.price_history or len(self.price_history[market.id]) < 5:
            return "weak"
        
        # Calculate price variance as proxy for trend strength
        recent_prices = [price for _, price in self.price_history[market.id][-20:]]
        
        if len(recent_prices) < 5:
            return "weak"
        
        price_variance = statistics.variance(recent_prices) if len(recent_prices) > 1 else 0
        
        # Higher variance suggests stronger trends
        if price_variance > 0.05:
            return "strong"
        elif price_variance > 0.02:
            return "moderate"
        else:
            return "weak"
    
    def identify_support_resistance(self, market: Market) -> Dict[str, float]:
        """Identify support and resistance levels"""
        if market.id not in self.price_history or len(self.price_history[market.id]) < 10:
            return {"support": 0.3, "resistance": 0.7}
        
        recent_prices = [price for _, price in self.price_history[market.id][-50:]]
        current_prob = market.probability
        
        # Simple support/resistance based on recent highs and lows
        recent_high = max(recent_prices)
        recent_low = min(recent_prices)
        
        # Support is recent low or current if lower
        support = min(current_prob, recent_low)
        
        # Resistance is recent high or current if higher
        resistance = max(current_prob, recent_high)
        
        return {
            "support": max(0.1, support),
            "resistance": min(0.9, resistance)
        }
    
    async def generate_reasoning(self, market: Market, score: float) -> str:
        """Generate detailed reasoning for technical score"""
        # Calculate individual components
        volume_score = self._analyze_volume(market)
        liquidity_score = self._analyze_liquidity(market)
        momentum_score = self._analyze_momentum(market)
        depth_score = self._analyze_market_depth(market)
        stability_score = self._analyze_probability_stability(market)
        time_score = self._analyze_time_patterns(market)
        
        # Get additional technical indicators
        momentum = self.calculate_momentum_indicator(market)
        trend_strength = self.calculate_trend_strength(market)
        levels = self.identify_support_resistance(market)
        
        reasoning_parts = []
        
        # Volume analysis
        if volume_score >= 0.8:
            reasoning_parts.append("Very high volume indicates strong interest")
        elif volume_score >= 0.6:
            reasoning_parts.append("Good volume supports trading")
        elif volume_score >= 0.4:
            reasoning_parts.append("Moderate volume")
        else:
            reasoning_parts.append("Low volume may limit trading opportunities")
        
        # Liquidity analysis
        if liquidity_score >= 0.8:
            reasoning_parts.append("Excellent liquidity with balanced pools")
        elif liquidity_score >= 0.6:
            reasoning_parts.append("Good liquidity")
        else:
            reasoning_parts.append("Limited liquidity")
        
        # Momentum
        if momentum != "neutral":
            reasoning_parts.append(f"{momentum.capitalize()} momentum detected ({trend_strength} trend)")
        
        # Market depth
        if depth_score >= 0.8:
            reasoning_parts.append("Deep market with strong order book")
        elif depth_score >= 0.6:
            reasoning_parts.append("Adequate market depth")
        else:
            reasoning_parts.append("Shallow market")
        
        # Support/Resistance
        reasoning_parts.append(f"Support: {levels['support']:.2f}, Resistance: {levels['resistance']:.2f}")
        
        # Combine into final reasoning
        if score >= 0.8:
            return f"Excellent technical setup: {', '.join(reasoning_parts)}. Score: {score:.2f}"
        elif score >= 0.6:
            return f"Good technical conditions: {', '.join(reasoning_parts)}. Score: {score:.2f}"
        elif score >= 0.4:
            return f"Moderate technical quality: {', '.join(reasoning_parts)}. Score: {score:.2f}"
        else:
            return f"Poor technical conditions: {', '.join(reasoning_parts)}. Score: {score:.2f}"