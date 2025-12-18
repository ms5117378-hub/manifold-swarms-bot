"""
Market discovery and API integration for Manifold Markets
"""
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
from dataclasses import asdict

from src.models import Market, MarketResolution
from src.utils.config import config
from src.utils.logger import get_logger

log = get_logger(__name__)


class ManifoldAPIClient:
    """Client for Manifold Markets API"""
    
    def __init__(self):
        self.api_key = config.get('env.manifold_api_key')
        self.base_url = config.get('env.manifold_base_url', 'https://manifold.markets/api/v0')
        self.target_user = config.get('trading.target_user', 'MikhailTal')
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers={'Authorization': f'Key {self.api_key}'} if self.api_key else {}
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make API request with error handling"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    log.error(f"API request failed: {response.status} - {error_text}")
                    raise Exception(f"API request failed: {response.status}")
                    
        except aiohttp.ClientError as e:
            log.error(f"Network error: {str(e)}")
            raise
        except Exception as e:
            log.error(f"Request error: {str(e)}")
            raise
    
    async def get_user_markets(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get markets created by specific user"""
        params = {
            'userId': user_id,
            'limit': limit
        }
        
        response = await self._make_request('markets', params)
        return response
    
    async def get_market(self, market_id: str) -> Dict[str, Any]:
        """Get specific market details"""
        return await self._make_request(f'market/{market_id}')
    
    async def get_market_orders(self, market_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get order book for a market"""
        params = {'limit': limit}
        return await self._make_request(f'market/{market_id}/orders', params)
    
    async def get_market_bets(self, market_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent bets for a market"""
        params = {'limit': limit}
        return await self._make_request(f'market/{market_id}/bets', params)
    
    async def place_bet(self, market_id: str, outcome: str, amount: float, 
                      limit_prob: Optional[float] = None) -> Dict[str, Any]:
        """Place a bet on a market"""
        if not self.api_key:
            raise ValueError("API key required for placing bets")
        
        data = {
            'outcome': outcome,
            'amount': amount
        }
        
        if limit_prob is not None:
            data['limitProb'] = limit_prob
        
        async with self.session.post(f"{self.base_url}/bet", json=data) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Bet placement failed: {response.status} - {error_text}")
    
    async def get_user_balance(self, user_id: str) -> float:
        """Get user's M$ balance"""
        response = await self._make_request(f'user/{user_id}')
        return response.get('balance', 0.0)
    
    async def get_user_positions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user's open positions"""
        return await self._make_request(f'user/{user_id}/positions')


class MarketDiscovery:
    """Discovers and filters markets from Manifold"""
    
    def __init__(self):
        self.api_client = ManifoldAPIClient()
        self.target_user = config.get('trading.target_user', 'MikhailTal')
        self.market_cache: Dict[str, Market] = {}
        self.last_update: Optional[datetime] = None
        
    async def discover_markets(self, force_refresh: bool = False) -> List[Market]:
        """Discover markets from target user"""
        try:
            async with self.api_client as client:
                # Get markets from target user
                raw_markets = await client.get_user_markets(self.target_user, limit=100)
                
                # Convert to Market objects and filter
                markets = []
                for raw_market in raw_markets:
                    market = self._parse_market(raw_market)
                    if self._is Tradable(market):
                        markets.append(market)
                        self.market_cache[market.id] = market
                
                self.last_update = datetime.now()
                log.info(f"Discovered {len(markets)} tradable markets from {self.target_user}")
                return markets
                
        except Exception as e:
            log.error(f"Error discovering markets: {str(e)}")
            return []
    
    def _parse_market(self, raw_market: Dict[str, Any]) -> Market:
        """Parse raw market data into Market object"""
        # Handle different outcome types
        outcome_type = raw_market.get('outcomeType', 'BINARY')
        mechanism = raw_market.get('mechanism', 'cpmm-1')
        
        # Parse pool data
        pool = raw_market.get('pool', {})
        if isinstance(pool, dict):
            pool = {k: float(v) for k, v in pool.items()}
        
        # Parse probability
        if outcome_type == 'BINARY':
            probability = raw_market.get('probability', 0.0)
        else:
            # For multi-choice markets, use the first outcome's probability
            probability = raw_market.get('probability', 0.0)
        
        # Parse dates
        created_time = datetime.fromisoformat(raw_market['createdTime'].replace('Z', '+00:00'))
        close_time = None
        if raw_market.get('closeTime'):
            close_time = datetime.fromisoformat(raw_market['closeTime'].replace('Z', '+00:00'))
        
        return Market(
            id=raw_market['id'],
            creator_id=raw_market['creatorId'],
            question=raw_market['question'],
            description=raw_market.get('description', ''),
            created_time=created_time,
            close_time=close_time,
            resolution=raw_market.get('resolution'),
            probability=float(probability),
            volume=float(raw_market.get('volume', 0.0)),
            pool=pool,
            outcome_type=outcome_type,
            mechanism=mechanism,
            tags=raw_market.get('tags', []),
            url=f"https://manifold.markets/{raw_market['id']}"
        )
    
    def _is_tradable(self, market: Market) -> bool:
        """Check if market is tradable based on criteria"""
        # Must be created by target user
        if market.creator_id != self.target_user:
            return False
        
        # Must be open (not resolved)
        if market.resolution is not None:
            return False
        
        # Must have sufficient volume
        min_volume = config.get('risk.min_volume_requirement', 50)
        if market.volume < min_volume:
            return False
        
        # Must not be closing too soon
        min_hours = config.get('risk.min_market_hours', 6)
        if market.close_time:
            hours_to_close = (market.close_time - datetime.now()).total_seconds() / 3600
            if hours_to_close < min_hours:
                return False
        
        # Must not be too old
        max_age = config.get('risk.max_market_age', 365)
        market_age = (datetime.now() - market.created_time).days
        if market_age > max_age:
            return False
        
        # Filter out banned tags
        banned_tags = config.get('risk.banned_tags', [])
        if any(tag in market.tags for tag in banned_tags):
            return False
        
        return True
    
    async def update_market(self, market_id: str) -> Optional[Market]:
        """Update specific market data"""
        try:
            async with self.api_client as client:
                raw_market = await client.get_market(market_id)
                market = self._parse_market(raw_market)
                
                if self._is_tradable(market):
                    self.market_cache[market_id] = market
                    return market
                else:
                    # Remove from cache if no longer tradable
                    self.market_cache.pop(market_id, None)
                    return None
                    
        except Exception as e:
            log.error(f"Error updating market {market_id}: {str(e)}")
            return None
    
    def get_cached_market(self, market_id: str) -> Optional[Market]:
        """Get market from cache"""
        return self.market_cache.get(market_id)
    
    def get_all_cached_markets(self) -> List[Market]:
        """Get all cached markets"""
        return list(self.market_cache.values())
    
    def get_markets_needing_update(self, minutes: int = 5) -> List[str]:
        """Get markets that need updating based on age"""
        if not self.last_update:
            return list(self.market_cache.keys())
        
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [
            market_id for market_id, market in self.market_cache.items()
            if market.last_updated < cutoff_time
        ]
    
    async def refresh_markets(self, market_ids: List[str] = None) -> int:
        """Refresh specific markets or all if none specified"""
        if market_ids is None:
            market_ids = list(self.market_cache.keys())
        
        updated_count = 0
        for market_id in market_ids:
            market = await self.update_market(market_id)
            if market:
                updated_count += 1
        
        log.info(f"Refreshed {updated_count}/{len(market_ids)} markets")
        return updated_count
    
    def get_market_statistics(self) -> Dict[str, Any]:
        """Get statistics about cached markets"""
        if not self.market_cache:
            return {}
        
        markets = list(self.market_cache.values())
        
        # Basic stats
        total_volume = sum(m.volume for m in markets)
        avg_probability = sum(m.probability for m in markets) / len(markets)
        
        # By tags
        tag_counts = {}
        for market in markets:
            for tag in market.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        # By time to close
        open_markets = [m for m in markets if m.close_time]
        if open_markets:
            avg_hours_to_close = sum(
                (m.close_time - datetime.now()).total_seconds() / 3600 
                for m in open_markets
            ) / len(open_markets)
        else:
            avg_hours_to_close = 0
        
        return {
            'total_markets': len(markets),
            'total_volume': total_volume,
            'average_probability': avg_probability,
            'average_hours_to_close': avg_hours_to_close,
            'tag_distribution': tag_counts,
            'last_update': self.last_update.isoformat() if self.last_update else None
        }


class MarketMonitor:
    """Monitors markets for changes and opportunities"""
    
    def __init__(self, discovery: MarketDiscovery):
        self.discovery = discovery
        self.monitoring_active = False
        self.check_interval = config.get('trading.check_interval', 300)  # 5 minutes
        
    async def start_monitoring(self):
        """Start continuous market monitoring"""
        self.monitoring_active = True
        log.info(f"Started market monitoring with {self.check_interval}s interval")
        
        while self.monitoring_active:
            try:
                # Discover new markets
                await self.discovery.discover_markets()
                
                # Update existing markets
                market_ids = self.discovery.get_markets_needing_update(
                    minutes=self.check_interval // 60
                )
                if market_ids:
                    await self.discovery.refresh_markets(market_ids)
                
                # Log statistics
                stats = self.discovery.get_market_statistics()
                log.info(f"Market update: {stats.get('total_markets', 0)} markets, "
                        f"{stats.get('total_volume', 0):.0f}M$ total volume")
                
                # Wait for next check
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                log.error(f"Error in market monitoring: {str(e)}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    def stop_monitoring(self):
        """Stop market monitoring"""
        self.monitoring_active = False
        log.info("Stopped market monitoring")
    
    async def check_single_market(self, market_id: str) -> Optional[Market]:
        """Check a single market for updates"""
        return await self.discovery.update_market(market_id)
    
    def get_high_volume_markets(self, min_volume: float = 100) -> List[Market]:
        """Get markets with high volume"""
        return [
            market for market in self.discovery.get_all_cached_markets()
            if market.volume >= min_volume
        ]
    
    def get_fast_closing_markets(self, hours: int = 24) -> List[Market]:
        """Get markets closing soon"""
        cutoff_time = datetime.now() + timedelta(hours=hours)
        return [
            market for market in self.discovery.get_all_cached_markets()
            if market.close_time and market.close_time <= cutoff_time
        ]
    
    def get_volatile_markets(self, threshold: float = 0.1) -> List[Market]:
        """Get markets with high volatility (placeholder - would need historical data)"""
        # This is a placeholder implementation
        # In reality, would analyze price movements over time
        return self.discovery.get_all_cached_markets()[:5]  # Return first 5 as example