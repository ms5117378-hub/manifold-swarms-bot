"""
Fundamental Analyst Agent
"""
from typing import Dict, Any
import re
from datetime import datetime

from src.agents.base_agent import AnalystAgent
from src.models import Market
from src.utils.config import config
from src.utils.logger import get_logger

log = get_logger(__name__)


class FundamentalAnalyst(AnalystAgent):
    """Analyzes market fundamentals: question clarity, resolution criteria, creator reputation"""
    
    def __init__(self):
        super().__init__("Fundamental Analyst", "fundamental_analysis")
    
    def get_default_prompt(self) -> str:
        return config.get_agent_prompt('fundamental') or """
        You are a fundamental analyst for prediction markets specializing in question quality assessment.
        
        Analyze markets for:
        1. Question clarity and resolution criteria
        2. Reasonable timeframe for resolution
        3. Market creator reputation and track record
        4. Potential ambiguities or edge cases
        5. Information availability and researchability
        
        Provide:
        - Quality score: 0.0 to 1.0
        - Confidence in your analysis: 0.0 to 1.0
        - Detailed reasoning
        - Key factors affecting resolution
        
        Focus on markets created by MikhailTal and assess their fundamental characteristics.
        """
    
    async def calculate_score(self, market: Market) -> float:
        """Calculate fundamental quality score for the market"""
        score_components = {}
        
        # 1. Question clarity (0-1)
        clarity_score = self._assess_question_clarity(market.question, market.description)
        score_components['clarity'] = clarity_score
        
        # 2. Resolution criteria (0-1)
        resolution_score = self._assess_resolution_criteria(market.description, market.outcome_type)
        score_components['resolution'] = resolution_score
        
        # 3. Timeframe appropriateness (0-1)
        timeframe_score = self._assess_timeframe(market.created_time, market.close_time)
        score_components['timeframe'] = timeframe_score
        
        # 4. Creator reputation (0-1) - MikhailTal gets high base score
        reputation_score = self._assess_creator_reputation(market.creator_id)
        score_components['reputation'] = reputation_score
        
        # 5. Researchability (0-1)
        research_score = self._assess_researchability(market.question, market.tags)
        score_components['researchability'] = research_score
        
        # 6. Market liquidity (0-1)
        liquidity_score = self._assess_liquidity(market.volume, market.pool)
        score_components['liquidity'] = liquidity_score
        
        # Calculate weighted average
        weights = {
            'clarity': 0.25,
            'resolution': 0.25,
            'timeframe': 0.15,
            'reputation': 0.15,
            'researchability': 0.10,
            'liquidity': 0.10
        }
        
        total_score = sum(score_components[component] * weight 
                         for component, weight in weights.items())
        
        # Log component scores for debugging
        self.logger.logger.debug(f"Fundamental analysis for {market.id}: {score_components}")
        
        return max(0.0, min(1.0, total_score))  # Ensure score is between 0 and 1
    
    def _assess_question_clarity(self, question: str, description: str) -> float:
        """Assess question clarity"""
        clarity_score = 0.5  # Base score
        
        # Check for clear yes/no structure
        if any(word in question.lower() for word in ['will', 'is', 'are', 'does', 'did', 'can', 'could', 'should']):
            clarity_score += 0.1
        
        # Check for specific metrics or dates
        if re.search(r'\d{4}', question) or re.search(r'\$', question):
            clarity_score += 0.1
        
        # Check for ambiguous terms
        ambiguous_terms = ['maybe', 'possibly', 'likely', 'probably', 'might', 'could be']
        if any(term in question.lower() for term in ambiguous_terms):
            clarity_score -= 0.2
        
        # Check description length
        if description and len(description) > 50:
            clarity_score += 0.1
        
        # Check for clear resolution criteria
        if description and any(word in description.lower() for word in 
                             ['resolve', 'criteria', 'determined', 'based on']):
            clarity_score += 0.1
        
        return max(0.0, min(1.0, clarity_score))
    
    def _assess_resolution_criteria(self, description: str, outcome_type: str) -> float:
        """Assess clarity of resolution criteria"""
        resolution_score = 0.3  # Base score
        
        if not description:
            return 0.1  # Very low score without description
        
        # Look for resolution indicators
        resolution_indicators = [
            'will resolve', 'based on', 'according to', 'determined by',
            'criteria', 'resolution', 'source', 'official', 'data'
        ]
        
        found_indicators = sum(1 for indicator in resolution_indicators 
                              if indicator in description.lower())
        resolution_score += min(0.4, found_indicators * 0.1)
        
        # Look for specific sources
        sources = ['reuters', 'bloomberg', 'official', 'government', 'census', 'nasa']
        found_sources = sum(1 for source in sources if source in description.lower())
        resolution_score += min(0.2, found_sources * 0.05)
        
        # Check for specific dates or timeframes
        if re.search(r'\b(20\d{2})\b', description):
            resolution_score += 0.1
        
        return max(0.0, min(1.0, resolution_score))
    
    def _assess_timeframe(self, created_time: datetime, close_time: datetime) -> float:
        """Assess if timeframe is reasonable"""
        if not close_time:
            return 0.3  # Lower score for no close time
        
        total_duration = (close_time - created_time).days
        elapsed = (datetime.now() - created_time).days
        remaining = max(0, (close_time - datetime.now()).days)
        
        # Ideal timeframe: 1 week to 6 months
        if 7 <= total_duration <= 180:
            timeframe_score = 0.8
        elif 1 <= total_duration <= 365:
            timeframe_score = 0.6
        else:
            timeframe_score = 0.4
        
        # Adjust based on remaining time
        if remaining < 1:  # Less than 1 day
            timeframe_score -= 0.3
        elif remaining < 7:  # Less than 1 week
            timeframe_score -= 0.1
        
        # Penalize if too much time has passed without resolution
        if elapsed > total_duration * 0.8:  # More than 80% of time passed
            timeframe_score -= 0.2
        
        return max(0.0, min(1.0, timeframe_score))
    
    def _assess_creator_reputation(self, creator_id: str) -> float:
        """Assess creator reputation"""
        # MikhailTal gets high reputation score
        if creator_id == "MikhailTal":
            return 0.9
        
        # For other creators, would need to look up their track record
        # For now, return neutral score
        return 0.5
    
    def _assess_researchability(self, question: str, tags: list) -> float:
        """Assess how researchable the question is"""
        research_score = 0.3  # Base score
        
        # Check for researchable topics
        researchable_topics = [
            'election', 'stock', 'price', 'gdp', 'inflation', 'unemployment',
            'covid', 'weather', 'temperature', 'sports', 'game', 'movie'
        ]
        
        found_topics = sum(1 for topic in researchable_topics 
                          if topic in question.lower())
        research_score += min(0.4, found_topics * 0.1)
        
        # Check for helpful tags
        helpful_tags = ['politics', 'economics', 'sports', 'science', 'tech']
        found_helpful_tags = sum(1 for tag in helpful_tags if tag in tags)
        research_score += min(0.2, found_helpful_tags * 0.05)
        
        # Penalize very specific or obscure topics
        obscure_indicators = ['specific person', 'niche', 'obscure', 'local']
        if any(indicator in question.lower() for indicator in obscure_indicators):
            research_score -= 0.2
        
        return max(0.0, min(1.0, research_score))
    
    def _assess_liquidity(self, volume: float, pool: dict) -> float:
        """Assess market liquidity"""
        if volume >= 1000:  # M$1000+
            return 1.0
        elif volume >= 500:  # M$500+
            return 0.8
        elif volume >= 100:  # M$100+
            return 0.6
        elif volume >= 50:   # M$50+
            return 0.4
        else:
            return 0.2
    
    async def generate_reasoning(self, market: Market, score: float) -> str:
        """Generate detailed reasoning for the fundamental score"""
        # Calculate individual components for reasoning
        clarity = self._assess_question_clarity(market.question, market.description)
        resolution = self._assess_resolution_criteria(market.description, market.outcome_type)
        timeframe = self._assess_timeframe(market.created_time, market.close_time)
        reputation = self._assess_creator_reputation(market.creator_id)
        researchability = self._assess_researchability(market.question, market.tags)
        liquidity = self._assess_liquidity(market.volume, market.pool)
        
        reasoning_parts = []
        
        if clarity >= 0.7:
            reasoning_parts.append("Question is clear and well-structured")
        elif clarity >= 0.5:
            reasoning_parts.append("Question has moderate clarity")
        else:
            reasoning_parts.append("Question lacks clarity")
        
        if resolution >= 0.7:
            reasoning_parts.append("Strong resolution criteria")
        elif resolution >= 0.5:
            reasoning_parts.append("Adequate resolution criteria")
        else:
            reasoning_parts.append("Weak or missing resolution criteria")
        
        if timeframe >= 0.7:
            reasoning_parts.append("Appropriate timeframe")
        elif timeframe >= 0.5:
            reasoning_parts.append("Reasonable timeframe")
        else:
            reasoning_parts.append("Problematic timeframe")
        
        if reputation >= 0.8:
            reasoning_parts.append("Excellent creator reputation")
        elif reputation >= 0.6:
            reasoning_parts.append("Good creator reputation")
        else:
            reasoning_parts.append("Unknown creator reputation")
        
        if researchability >= 0.7:
            reasoning_parts.append("Highly researchable")
        elif researchability >= 0.5:
            reasoning_parts.append("Moderately researchable")
        else:
            reasoning_parts.append("Difficult to research")
        
        if liquidity >= 0.7:
            reasoning_parts.append("Good liquidity")
        elif liquidity >= 0.5:
            reasoning_parts.append("Adequate liquidity")
        else:
            reasoning_parts.append("Low liquidity")
        
        # Combine into final reasoning
        if score >= 0.8:
            return f"Excellent fundamental quality: {', '.join(reasoning_parts)}. Score: {score:.2f}"
        elif score >= 0.6:
            return f"Good fundamental quality: {', '.join(reasoning_parts)}. Score: {score:.2f}"
        elif score >= 0.4:
            return f"Moderate fundamental quality: {', '.join(reasoning_parts)}. Score: {score:.2f}"
        else:
            return f"Poor fundamental quality: {', '.join(reasoning_parts)}. Score: {score:.2f}"