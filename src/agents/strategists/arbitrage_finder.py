"""
Arbitrage Finder Strategy Agent
"""
from typing import Dict, Any, List, Tuple, Optional
import statistics
from datetime import datetime

from src.agents.base_agent import StrategyAgent
from src.models import Market, TradingAction
from src.utils.config import config
from src.utils.logger import get_logger

log = get_logger(__name__)


class ArbitrageFinder(StrategyAgent):
    """Identifies arbitrage opportunities across related markets or inconsistent probabilities"""
    
    def __init__(self):
        super().__init__("Arbitrage Finder", "arbitrage_trading")
        self.related_markets_cache: Dict[str, List[str]] = {}
        self.probability_inconsistencies: Dict[str, Dict[str, float]] = {}
        
    def get_default_prompt(self) -> str:
        return config.get_agent_prompt('arbitrage_finder') or """
        You are an arbitrage specialist for prediction markets finding risk-free opportunities.
        
        Strategy:
        1. Identify arbitrage across related markets
        2. Find inconsistent probabilities
        3. Exploit market inefficiencies
        4. Hedge positions to minimize risk
        5. Quick execution to capture opportunities
        
        Provide:
        - Arbitrage opportunity identification
        - Risk-free profit percentage
        - Required hedge positions
        - Execution urgency
        - Capital requirements
        
        Focus on finding guaranteed profits through market inconsistencies and correlations.
        """
    
    async def evaluate_opportunity(self, market: Market, analysis_data: Dict[str, float]) -> Dict[str, Any]:
        """Evaluate arbitrage opportunities"""
        
        # Get analysis scores
        fundamental_score = analysis_data.get('fundamental_score', 0.5)
        technical_score = analysis_data.get('technical_score', 0.5)
        sentiment_score = analysis_data.get('sentiment_score', 0.5)
        
        # Find related markets
        related_markets = self._find_related_markets(market)
        
        # Identify arbitrage opportunities
        arbitrage_opportunities = []
        
        # 1. Cross-market arbitrage
        cross_market_arb = self._identify_cross_market_arbitrage(market, related_markets)
        if cross_market_arb:
            arbitrage_opportunities.append(cross_market_arb)
        
        # 2. Probability inconsistency arbitrage
        prob_arb = self._identify_probability_inconsistency(market)
        if prob_arb:
            arbitrage_opportunities.append(prob_arb)
        
        # 3. Temporal arbitrage (if we had historical data)
        temporal_arb = self._identify_temporal_arbitrage(market)
        if temporal_arb:
            arbitrage_opportunities.append(temporal_arb)
        
        # 4. Logical arbitrage (mutually exclusive outcomes)
        logical_arb = self._identify_logical_arbitrage(market, related_markets)
        if logical_arb:
            arbitrage_opportunities.append(logical_arb)
        
        # Select best arbitrage opportunity
        best_arb = self._select_best_arbitrage(arbitrage_opportunities)
        
        if not best_arb:
            return {
                "action": TradingAction.HOLD,
                "outcome": "",
                "confidence": 0.2,
                "reasoning": "No profitable arbitrage opportunities found",
                "estimated_probability": market.probability,
                "position_size": 0.0,
                "arbitrage_type": "none",
                "opportunities_found": len(arbitrage_opportunities)
            }
        
        # Calculate position requirements
        position_requirements = self._calculate_arbitrage_positions(best_arb, market)
        
        # Determine action based on arbitrage type
        action, outcome = self._determine_arbitrage_action(best_arb, market)
        
        # Calculate confidence based on arbitrage quality
        confidence = min(0.95, best_arb.get('profit_percentage', 0) * 5)  # Higher profit = higher confidence
        
        return {
            "action": action,
            "outcome": outcome,
            "confidence": confidence,
            "reasoning": best_arb.get('description', ''),
            "estimated_probability": market.probability,
            "position_size": position_requirements.get('primary_position', 0.05),
            "arbitrage_type": best_arb.get('type', 'unknown'),
            "profit_percentage": best_arb.get('profit_percentage', 0),
            "hedge_positions": position_requirements.get('hedge_positions', []),
            "execution_urgency": best_arb.get('urgency', 'medium'),
            "capital_required": position_requirements.get('total_capital', 0),
            "risk_level": best_arb.get('risk_level', 'low'),
            "related_markets": best_arb.get('related_markets', [])
        }
    
    def _find_related_markets(self, market: Market) -> List[str]:
        """Find markets related to the given market"""
        # In a real implementation, would use semantic similarity and API calls
        # For now, use simple keyword matching
        
        related_keywords = self._extract_related_keywords(market)
        related_markets = []
        
        # Mock related markets based on keywords
        if 'election' in market.question.lower():
            related_markets.extend([
                "mock_presidential_election_winner",
                "mock_senate_control",
                "mock_voter_turnout"
            ])
        elif 'economy' in market.question.lower() or 'gdp' in market.question.lower():
            related_markets.extend([
                "mock_gdp_growth",
                "mock_unemployment_rate",
                "mock_inflation_rate"
            ])
        elif 'stock' in market.question.lower() or 'market' in market.question.lower():
            related_markets.extend([
                "mock_sp500_return",
                "mock_interest_rates",
                "mock_recession_probability"
            ])
        
        return related_markets
    
    def _extract_related_keywords(self, market: Market) -> List[str]:
        """Extract keywords for finding related markets"""
        question_words = market.question.lower().split()
        description_words = market.description.lower().split() if market.description else []
        tags = [tag.lower() for tag in market.tags]
        
        all_words = question_words + description_words + tags
        
        # Filter for meaningful keywords
        meaningful_words = [
            word for word in all_words 
            if len(word) > 3 and word not in ['will', 'the', 'this', 'that', 'with', 'from']
        ]
        
        return list(set(meaningful_words))[:5]  # Return top 5 unique keywords
    
    def _identify_cross_market_arbitrage(self, market: Market, related_markets: List[str]) -> Optional[Dict[str, Any]]:
        """Identify arbitrage opportunities across related markets"""
        # In a real implementation, would fetch actual related market data
        # For now, simulate potential arbitrage scenarios
        
        arbitrage_opportunities = []
        
        # Simulate finding inconsistent probabilities across related markets
        for related_market_id in related_markets:
            # Mock related market probability
            related_prob = self._get_mock_probability(related_market_id, market.probability)
            
            # Check for arbitrage
            if self._has_probability_arbitrage(market.probability, related_prob):
                profit_pct = abs(market.probability - related_prob) * 0.5  # Simplified profit calculation
                
                arbitrage_opportunities.append({
                    'type': 'cross_market',
                    'related_market': related_market_id,
                    'market1_prob': market.probability,
                    'market2_prob': related_prob,
                    'profit_percentage': profit_pct,
                    'description': f'Cross-market arbitrage with {related_market_id}',
                    'urgency': 'high' if profit_pct > 0.1 else 'medium',
                    'risk_level': 'low'
                })
        
        # Return best opportunity
        return max(arbitrage_opportunities, key=lambda x: x['profit_percentage']) if arbitrage_opportunities else None
    
    def _identify_probability_inconsistency(self, market: Market) -> Optional[Dict[str, Any]]:
        """Identify internal probability inconsistencies"""
        current_prob = market.probability
        
        # Check for logical inconsistencies
        inconsistencies = []
        
        # 1. Extreme probability with low volume (possible inefficiency)
        if (current_prob >= 0.9 or current_prob <= 0.1) and market.volume < 200:
            inconsistencies.append({
                'type': 'volume_probability_mismatch',
                'severity': 'medium',
                'description': f'Extreme probability ({current_prob:.2f}) with low volume ({market.volume:.0f}M$)'
            })
        
        # 2. Probability vs time inconsistency
        if market.close_time:
            days_to_close = (market.close_time - datetime.now()).days
            if days_to_close > 90 and current_prob >= 0.8:
                inconsistencies.append({
                    'type': 'time_probability_mismatch',
                    'severity': 'low',
                    'description': f'High probability ({current_prob:.2f}) far from resolution ({days_to_close} days)'
                })
        
        # 3. Pool imbalance inconsistency
        pool = market.pool
        if pool and market.outcome_type == 'BINARY':
            yes_pool = pool.get('YES', 0)
            no_pool = pool.get('NO', 0)
            total_pool = yes_pool + no_pool
            
            if total_pool > 0:
                pool_implied_prob = yes_pool / total_pool
                prob_diff = abs(current_prob - pool_implied_prob)
                
                if prob_diff > 0.1:  # 10% difference
                    inconsistencies.append({
                        'type': 'pool_price_mismatch',
                        'severity': 'high',
                        'description': f'Price ({current_prob:.2f}) differs from pool implied ({pool_implied_prob:.2f})',
                        'profit_percentage': prob_diff * 0.3
                    })
        
        # Return most severe inconsistency
        if inconsistencies:
            best = max(inconsistencies, key=lambda x: {'high': 3, 'medium': 2, 'low': 1}[x['severity']])
            
            return {
                'type': 'probability_inconsistency',
                'subtype': best['type'],
                'description': best['description'],
                'profit_percentage': best.get('profit_percentage', 0.05),
                'urgency': 'high' if best['severity'] == 'high' else 'medium',
                'risk_level': 'low'
            }
        
        return None
    
    def _identify_temporal_arbitrage(self, market: Market) -> Optional[Dict[str, Any]]:
        """Identify temporal arbitrage opportunities (would need historical data)"""
        # Placeholder for temporal arbitrage detection
        # In a real implementation, would analyze historical price patterns
        
        return None
    
    def _identify_logical_arbitrage(self, market: Market, related_markets: List[str]) -> Optional[Dict[str, Any]]:
        """Identify logical arbitrage from mutually exclusive outcomes"""
        # In a real implementation, would identify markets that are logically related
        # For example: "Party A wins election" vs "Party B wins election"
        
        # Placeholder implementation
        if 'election' in market.question.lower() and len(related_markets) >= 2:
            # Simulate finding mutually exclusive outcomes with inconsistent probabilities
            total_prob = market.probability + 0.6  # Mock related market probability
            
            if total_prob > 1.0:  # Arbitrage opportunity
                profit_pct = (total_prob - 1.0) * 0.8
                
                return {
                    'type': 'logical_arbitrage',
                    'description': f'Mutually exclusive outcomes sum to {total_prob:.1%} > 100%',
                    'profit_percentage': profit_pct,
                    'urgency': 'high',
                    'risk_level': 'very_low',
                    'related_markets': related_markets[:2]
                }
        
        return None
    
    def _select_best_arbitrage(self, opportunities: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Select the best arbitrage opportunity from available options"""
        if not opportunities:
            return None
        
        # Score opportunities by profit, urgency, and risk
        def score_opportunity(opp):
            profit_score = opp.get('profit_percentage', 0) * 10
            urgency_score = {'high': 3, 'medium': 2, 'low': 1}.get(opp.get('urgency', 'medium'), 2)
            risk_score = {'very_low': 4, 'low': 3, 'medium': 2, 'high': 1}.get(opp.get('risk_level', 'medium'), 2)
            
            return profit_score + urgency_score + risk_score
        
        return max(opportunities, key=score_opportunity)
    
    def _calculate_arbitrage_positions(self, arbitrage: Dict[str, Any], market: Market) -> Dict[str, Any]:
        """Calculate position requirements for arbitrage"""
        arb_type = arbitrage.get('type')
        
        if arb_type == 'cross_market':
            # Need positions in both markets
            primary_position = 0.05  # 5% in primary market
            hedge_position = 0.05    # 5% in related market
            total_capital = primary_position + hedge_position
            
            return {
                'primary_position': primary_position,
                'hedge_positions': [{'market': arbitrage.get('related_market'), 'size': hedge_position}],
                'total_capital': total_capital
            }
        
        elif arb_type == 'logical_arbitrage':
            # Need positions across multiple related markets
            primary_position = 0.04
            hedge_positions = [
                {'market': related_market, 'size': 0.03} 
                for related_market in arbitrage.get('related_markets', [])[:2]
            ]
            total_capital = primary_position + sum(hp['size'] for hp in hedge_positions)
            
            return {
                'primary_position': primary_position,
                'hedge_positions': hedge_positions,
                'total_capital': total_capital
            }
        
        else:
            # Simple single market arbitrage
            position_size = 0.06  # 6% for inconsistency arbitrage
            return {
                'primary_position': position_size,
                'hedge_positions': [],
                'total_capital': position_size
            }
    
    def _determine_arbitrage_action(self, arbitrage: Dict[str, Any], market: Market) -> Tuple[TradingAction, str]:
        """Determine the action for arbitrage opportunity"""
        arb_type = arbitrage.get('type')
        
        if arb_type == 'cross_market':
            # Action depends on relative probabilities
            market1_prob = arbitrage.get('market1_prob', market.probability)
            market2_prob = arbitrage.get('market2_prob', 0.5)
            
            if market1_prob > market2_prob:
                return TradingAction.SELL, "NO"  # Sell overpriced
            else:
                return TradingAction.BUY, "YES"  # Buy underpriced
        
        elif arb_type == 'probability_inconsistency':
            subtype = arbitrage.get('subtype')
            
            if subtype == 'pool_price_mismatch':
                # Trade toward pool implied probability
                pool = market.pool
                if pool:
                    yes_pool = pool.get('YES', 0)
                    no_pool = pool.get('NO', 0)
                    total_pool = yes_pool + no_pool
                    
                    if total_pool > 0:
                        pool_implied_prob = yes_pool / total_pool
                        
                        if market.probability > pool_implied_prob:
                            return TradingAction.SELL, "NO"
                        else:
                            return TradingAction.BUY, "YES"
        
        # Default action based on current probability
        if market.probability > 0.5:
            return TradingAction.SELL, "NO"
        else:
            return TradingAction.BUY, "YES"
    
    def _has_probability_arbitrage(self, prob1: float, prob2: float, threshold: float = 0.1) -> bool:
        """Check if there's arbitrage opportunity between two probabilities"""
        return abs(prob1 - prob2) > threshold
    
    def _get_mock_probability(self, market_id: str, reference_prob: float) -> float:
        """Get mock probability for related market (placeholder)"""
        # In a real implementation, would fetch actual market data
        # For now, create probabilities that sometimes create arbitrage
        
        import random
        
        # 30% chance of creating arbitrage opportunity
        if random.random() < 0.3:
            # Create significant difference
            return max(0.1, min(0.9, reference_prob + random.uniform(-0.3, 0.3)))
        else:
            # Create small difference
            return max(0.1, min(0.9, reference_prob + random.uniform(-0.05, 0.05)))
    
    def calculate_arbitrage_metrics(self, market: Market) -> Dict[str, Any]:
        """Calculate detailed arbitrage metrics for a market"""
        metrics = {
            'market_id': market.id,
            'current_probability': market.probability,
            'volume': market.volume,
            'pool_balance': 0.0,
            'implied_volatility': 0.0,
            'arbitrage_potential': 0.0
        }
        
        # Calculate pool balance
        pool = market.pool
        if pool and market.outcome_type == 'BINARY':
            yes_pool = pool.get('YES', 0)
            no_pool = pool.get('NO', 0)
            total_pool = yes_pool + no_pool
            
            if total_pool > 0:
                metrics['pool_balance'] = min(yes_pool, no_pool) / max(yes_pool, no_pool)
                
                # Calculate implied probability from pool
                pool_implied_prob = yes_pool / total_pool
                prob_diff = abs(market.probability - pool_implied_prob)
                metrics['arbitrage_potential'] = prob_diff
        
        # Mock implied volatility (would need options data in real implementation)
        if market.volume > 0:
            metrics['implied_volatility'] = min(1.0, market.volume / 1000)
        
        return metrics
    
    async def generate_arbitrage_thesis(self, market: Market, opportunity: Dict[str, Any]) -> str:
        """Generate detailed arbitrage thesis"""
        arb_type = opportunity.get('arbitrage_type', 'unknown')
        profit_pct = opportunity.get('profit_percentage', 0)
        urgency = opportunity.get('urgency', 'medium')
        risk_level = opportunity.get('risk_level', 'medium')
        
        thesis_parts = []
        
        if arb_type == 'none':
            return "No profitable arbitrage opportunities identified"
        
        # Main arbitrage description
        thesis_parts.append(f"{arb_type.replace('_', ' ').title()}: {opportunity.get('reasoning', '')}")
        
        # Profit potential
        thesis_parts.append(f"Expected profit: {profit_pct:.1%}")
        
        # Risk assessment
        thesis_parts.append(f"Risk level: {risk_level}")
        
        # Urgency
        thesis_parts.append(f"Execution urgency: {urgency}")
        
        # Capital requirements
        capital = opportunity.get('capital_required', 0)
        if capital > 0:
            thesis_parts.append(f"Capital required: {capital:.1%} of portfolio")
        
        # Related markets
        related = opportunity.get('related_markets', [])
        if related:
            thesis_parts.append(f"Related markets: {len(related)}")
        
        return " | ".join(thesis_parts)