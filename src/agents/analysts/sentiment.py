"""
Sentiment Analyst Agent
"""
from typing import Dict, Any, List
import asyncio
import aiohttp
from datetime import datetime

from src.agents.base_agent import AnalystAgent
from src.models import Market
from src.utils.config import config
from src.utils.logger import get_logger

log = get_logger(__name__)


class SentimentAnalyst(AnalystAgent):
    """Analyzes market sentiment through web search, news, social media, and related markets"""
    
    def __init__(self):
        super().__init__("Sentiment Analyst", "sentiment_analysis")
        self.web_search_enabled = True
        self.related_markets_cache: Dict[str, List[str]] = {}
        
    def get_default_prompt(self) -> str:
        return config.get_agent_prompt('sentiment') or """
        You are a sentiment analyst for prediction markets specializing in market psychology and information flow.
        
        Analyze:
        1. Web search for related news and events
        2. Social media sentiment and discussions
        3. Related market correlations
        4. Expert opinions and predictions
        5. Information asymmetry opportunities
        
        Provide:
        - Sentiment score: 0.0 to 1.0
        - Sentiment direction: bullish, bearish, neutral
        - Information advantage assessment
        - Key sentiment drivers
        - Contrarian opportunities
        
        Use web search capabilities to gather real-time information and identify sentiment edges.
        """
    
    async def calculate_score(self, market: Market) -> float:
        """Calculate sentiment analysis score"""
        score_components = {}
        
        # 1. News sentiment (0-1)
        news_score = await self._analyze_news_sentiment(market)
        score_components['news'] = news_score
        
        # 2. Social media sentiment (0-1)
        social_score = await self._analyze_social_sentiment(market)
        score_components['social'] = social_score
        
        # 3. Related markets sentiment (0-1)
        related_score = self._analyze_related_markets(market)
        score_components['related'] = related_score
        
        # 4. Expert opinion sentiment (0-1)
        expert_score = await self._analyze_expert_opinions(market)
        score_components['expert'] = expert_score
        
        # 5. Information freshness (0-1)
        freshness_score = self._analyze_information_freshness(market)
        score_components['freshness'] = freshness_score
        
        # 6. Contrarian opportunity (0-1)
        contrarian_score = self._identify_contrarian_opportunities(market)
        score_components['contrarian'] = contrarian_score
        
        # Calculate weighted average
        weights = {
            'news': 0.25,
            'social': 0.20,
            'related': 0.20,
            'expert': 0.15,
            'freshness': 0.10,
            'contrarian': 0.10
        }
        
        total_score = sum(score_components[component] * weight 
                         for component, weight in weights.items())
        
        return max(0.0, min(1.0, total_score))
    
    async def _analyze_news_sentiment(self, market: Market) -> float:
        """Analyze news sentiment for the market topic"""
        if not self.web_search_enabled:
            return 0.5  # Neutral score if web search disabled
        
        try:
            # Extract key terms from market question
            key_terms = self._extract_key_terms(market.question)
            
            # Search for recent news
            search_results = await self._web_search(key_terms, days_back=7)
            
            if not search_results:
                return 0.5  # Neutral if no news found
            
            # Analyze sentiment of search results
            positive_count = 0
            negative_count = 0
            total_relevance = 0
            
            for result in search_results:
                sentiment = self._analyze_text_sentiment(result.get('snippet', ''))
                relevance = self._calculate_relevance(result.get('snippet', ''), key_terms)
                
                total_relevance += relevance
                if sentiment > 0.1:
                    positive_count += 1
                elif sentiment < -0.1:
                    negative_count += 1
            
            if total_relevance == 0:
                return 0.5
            
            # Calculate sentiment score
            total_sentiment_articles = positive_count + negative_count
            if total_sentiment_articles == 0:
                return 0.5
            
            positive_ratio = positive_count / total_sentiment_articles
            
            # Return score based on sentiment strength and relevance
            return 0.5 + (positive_ratio - 0.5) * min(total_relevance / 10, 1.0)
            
        except Exception as e:
            log.error(f"Error analyzing news sentiment: {str(e)}")
            return 0.5
    
    async def _analyze_social_sentiment(self, market: Market) -> float:
        """Analyze social media sentiment - simplified implementation"""
        # In a real implementation, would integrate with Twitter API, Reddit API, etc.
        # For now, simulate based on market characteristics
        
        # Markets with more tags tend to have more social discussion
        tag_factor = min(len(market.tags) / 5, 1.0) * 0.2
        
        # Higher volume markets tend to have more social interest
        volume_factor = min(market.volume / 500, 1.0) * 0.3
        
        # Base sentiment score
        base_score = 0.5 + tag_factor + volume_factor
        
        return max(0.0, min(1.0, base_score))
    
    def _analyze_related_markets(self, market: Market) -> float:
        """Analyze sentiment from related markets"""
        # In a real implementation, would find and analyze related markets
        # For now, use a simplified approach based on tags and keywords
        
        related_score = 0.5  # Base score
        
        # Markets with common tags might have correlated sentiment
        common_tags = ['politics', 'economics', 'sports', 'technology', 'science']
        market_tags = set(market.tags)
        
        for tag in common_tags:
            if tag in market_tags:
                related_score += 0.1
        
        # Markets about well-known topics tend to have more related information
        well_known_topics = ['election', 'president', 'stock market', 'gdp', 'covid', 'climate']
        question_lower = market.question.lower()
        
        for topic in well_known_topics:
            if topic in question_lower:
                related_score += 0.1
        
        return max(0.0, min(1.0, related_score))
    
    async def _analyze_expert_opinions(self, market: Market) -> float:
        """Analyze expert opinions and predictions"""
        if not self.web_search_enabled:
            return 0.5
        
        try:
            # Search for expert opinions
            key_terms = self._extract_key_terms(market.question)
            expert_terms = key_terms + ['expert', 'analyst', 'prediction', 'forecast']
            
            search_results = await self._web_search(expert_terms, days_back=30)
            
            if not search_results:
                return 0.5
            
            # Look for expert sources
            expert_sources = ['bloomberg', 'reuters', 'wsj', 'economist', 'nate silver', 'five thirty eight']
            expert_count = 0
            total_relevance = 0
            
            for result in search_results:
                source = result.get('host_name', '').lower()
                snippet = result.get('snippet', '').lower()
                
                # Check if source is expert or contains expert terms
                is_expert = any(expert in source for expert in expert_sources) or \
                           any(expert in snippet for expert in expert_terms)
                
                if is_expert:
                    expert_count += 1
                    relevance = self._calculate_relevance(snippet, key_terms)
                    total_relevance += relevance
            
            if expert_count == 0:
                return 0.5
            
            # Score based on number of expert opinions and relevance
            expert_score = 0.5 + min(expert_count / 10, 0.5) * min(total_relevance / 5, 1.0)
            
            return max(0.0, min(1.0, expert_score))
            
        except Exception as e:
            log.error(f"Error analyzing expert opinions: {str(e)}")
            return 0.5
    
    def _analyze_information_freshness(self, market: Market) -> float:
        """Analyze freshness of available information"""
        now = datetime.now()
        
        # Check if market is about recent events
        question_lower = market.question.lower()
        
        # Look for time indicators
        recent_indicators = ['2024', 'this year', 'current', 'today', 'this week', 'this month']
        has_recent_indicator = any(indicator in question_lower for indicator in recent_indicators)
        
        # Check market age
        market_age_days = (now - market.created_time).days
        
        freshness_score = 0.5  # Base score
        
        if has_recent_indicator:
            freshness_score += 0.3
        
        if market_age_days <= 7:
            freshness_score += 0.2
        elif market_age_days <= 30:
            freshness_score += 0.1
        
        return max(0.0, min(1.0, freshness_score))
    
    def _identify_contrarian_opportunities(self, market: Market) -> float:
        """Identify contrarian opportunities"""
        # Look for markets where crowd might be wrong
        probability = market.probability
        
        contrarian_score = 0.5
        
        # Extreme probabilities might offer contrarian opportunities
        if probability >= 0.95 or probability <= 0.05:
            contrarian_score += 0.3  # Very extreme - likely overconfident
        elif probability >= 0.85 or probability <= 0.15:
            contrarian_score += 0.2  # Quite extreme
        elif probability >= 0.75 or probability <= 0.25:
            contrarian_score += 0.1  # Somewhat extreme
        
        # Markets with low volume but extreme probabilities might be inefficient
        if market.volume < 100 and (probability >= 0.8 or probability <= 0.2):
            contrarian_score += 0.2
        
        return max(0.0, min(1.0, contrarian_score))
    
    def _extract_key_terms(self, question: str) -> List[str]:
        """Extract key terms from market question"""
        # Simple keyword extraction - in real implementation would use NLP
        import re
        
        # Remove common words
        stop_words = {'will', 'the', 'be', 'is', 'are', 'by', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'and', 'or', 'but'}
        
        # Extract words
        words = re.findall(r'\b\w+\b', question.lower())
        key_terms = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Return top terms by length (longer terms are often more specific)
        return sorted(set(key_terms), key=len, reverse=True)[:5]
    
    async def _web_search(self, terms: List[str], days_back: int = 7) -> List[Dict[str, Any]]:
        """Perform web search - placeholder implementation"""
        # In a real implementation, would use web search API
        # For now, return mock results
        
        if not terms:
            return []
        
        # Mock search results based on terms
        mock_results = []
        
        # Simulate finding relevant results
        if any(term in ['election', 'politics', 'president'] for term in terms):
            mock_results.append({
                'title': 'Latest Election Polls and Analysis',
                'snippet': 'Recent polls show tight race with momentum shifting in key battleground states',
                'host_name': 'politico.com',
                'rank': 1
            })
        
        if any(term in ['economy', 'gdp', 'inflation'] for term in terms):
            mock_results.append({
                'title': 'Economic Indicators Show Mixed Signals',
                'snippet': 'GDP growth slows while inflation remains above target levels',
                'host_name': 'bloomberg.com',
                'rank': 2
            })
        
        if any(term in ['covid', 'health', 'pandemic'] for term in terms):
            mock_results.append({
                'title': 'Health Officials Update Pandemic Guidance',
                'snippet': 'New guidelines reflect changing understanding of virus transmission',
                'host_name': 'cdc.gov',
                'rank': 3
            })
        
        return mock_results
    
    def _analyze_text_sentiment(self, text: str) -> float:
        """Analyze sentiment of text - simplified implementation"""
        # Simple sentiment analysis based on keywords
        positive_words = ['good', 'great', 'excellent', 'positive', 'growth', 'success', 'win', 'gain', 'increase']
        negative_words = ['bad', 'terrible', 'negative', 'decline', 'failure', 'lose', 'decrease', 'crisis', 'risk']
        
        text_lower = text.lower()
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total_words = len(text.split())
        
        if total_words == 0:
            return 0.0
        
        # Calculate sentiment score
        sentiment = (positive_count - negative_count) / max(total_words, 1)
        
        return max(-1.0, min(1.0, sentiment * 10))  # Scale to -1 to 1 range
    
    def _calculate_relevance(self, text: str, key_terms: List[str]) -> float:
        """Calculate relevance of text to key terms"""
        if not key_terms:
            return 0.0
        
        text_lower = text.lower()
        term_count = sum(1 for term in key_terms if term in text_lower)
        
        return term_count / len(key_terms)
    
    def get_sentiment_direction(self, score: float) -> str:
        """Convert sentiment score to direction"""
        if score >= 0.6:
            return "bullish"
        elif score <= 0.4:
            return "bearish"
        else:
            return "neutral"
    
    async def generate_reasoning(self, market: Market, score: float) -> str:
        """Generate detailed reasoning for sentiment score"""
        # Get sentiment direction
        direction = self.get_sentiment_direction(score)
        
        # Analyze key factors
        key_terms = self._extract_key_terms(market.question)
        
        reasoning_parts = []
        
        # Information availability
        if len(key_terms) >= 3:
            reasoning_parts.append("Rich information environment with multiple key factors")
        elif len(key_terms) >= 2:
            reasoning_parts.append("Moderate information availability")
        else:
            reasoning_parts.append("Limited information sources")
        
        # Market characteristics
        if market.volume >= 500:
            reasoning_parts.append("High volume indicates strong market interest")
        elif market.volume >= 100:
            reasoning_parts.append("Moderate market interest")
        else:
            reasoning_parts.append("Low market interest may limit information flow")
        
        # Tag analysis
        if market.tags:
            reasoning_parts.append(f"Topic categories: {', '.join(market.tags[:3])}")
        
        # Contrarian potential
        if market.probability >= 0.9 or market.probability <= 0.1:
            reasoning_parts.append("Extreme probability suggests potential contrarian opportunity")
        
        # Time sensitivity
        if market.close_time:
            days_to_close = (market.close_time - datetime.now()).days
            if days_to_close <= 7:
                reasoning_parts.append("Time-sensitive information critical")
            elif days_to_close <= 30:
                reasoning_parts.append("Moderate time sensitivity")
            else:
                reasoning_parts.append("Long timeframe allows information accumulation")
        
        # Combine into final reasoning
        if score >= 0.8:
            return f"Strong {direction} sentiment: {', '.join(reasoning_parts)}. Score: {score:.2f}"
        elif score >= 0.6:
            return f"Moderate {direction} sentiment: {', '.join(reasoning_parts)}. Score: {score:.2f}"
        elif score >= 0.4:
            return f"Neutral sentiment: {', '.join(reasoning_parts)}. Score: {score:.2f}"
        else:
            return f"Weak sentiment indicators: {', '.join(reasoning_parts)}. Score: {score:.2f}"