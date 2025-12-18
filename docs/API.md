# API Documentation

## Overview

The Manifold Swarms Trading Bot provides both internal APIs for agent communication and external APIs for monitoring and control.

## Internal APIs

### Agent Communication API

#### Message Structure

```python
{
    "sender_agent": str,
    "receiver_agent": str,
    "message_type": str,  # "REQUEST", "RESPONSE", "NOTIFICATION"
    "content": Dict[str, Any],
    "priority": int,  # 1-5, 1 being highest
    "timestamp": datetime,
    "requires_response": bool,
    "correlation_id": Optional[str]
}
```

#### Agent Endpoints

##### Risk Manager
```python
# Validate trading decision
{
    "decision": {
        "market_id": str,
        "action": str,  # "BUY", "SELL", "HOLD"
        "position_size": float,
        "confidence": float,
        "agent_signals": List[Dict]
    }
}

# Response
{
    "approved": bool,
    "reduced": bool,
    "reason": str,
    "risk_score": float,
    "adjusted_position_size": float,
    "risk_breaches": List[Dict]
}
```

##### Trade Executor
```python
# Execute trade
{
    "execute_decision": {
        "market_id": str,
        "action": str,
        "outcome": str,  # "YES", "NO"
        "position_size": float,
        "confidence": float
    }
}

# Response
{
    "status": str,  # "SUCCESS", "FAILED", "PARTIAL"
    "trade_id": str,
    "amount": float,
    "price": float,
    "shares": float,
    "fee": float,
    "execution_time": float,
    "slippage": float
}
```

##### Portfolio Manager
```python
# Get portfolio summary
{
    "portfolio_summary": {}
}

# Response
{
    "current_balance": float,
    "total_value": float,
    "total_pnl": float,
    "total_return": float,
    "active_positions": int,
    "portfolio_metrics": Dict,
    "agent_contributions": Dict
}
```

### Workflow Management API

#### Workflow Execution
```python
# Execute market analysis
{
    "market_id": str,
    "workflow_type": str,  # "sequential", "parallel", "consensus", "hierarchical"
}

# Response
{
    "workflow_id": str,
    "workflow_type": str,
    "market_id": str,
    "result": Dict,
    "duration": float,
    "status": str,  # "completed", "failed", "running"
    "started_at": str,
    "completed_at": str
}
```

#### Workflow Status
```python
# Get workflow status
{
    "workflow_id": str
}

# Response
{
    "workflow_id": str,
    "status": str,
    "progress": float,
    "current_stage": str,
    "agent_status": Dict,
    "error": Optional[str]
}
```

## External APIs

### REST API

#### Base URL
```
http://localhost:8000/api/v1
```

#### Authentication
```python
# API Key in header
headers = {
    "X-API-Key": "your-api-key"
}
```

#### Endpoints

##### System Status
```http
GET /api/v1/status

Response:
{
    "status": "running",  # "running", "paused", "stopped"
    "uptime": str,
    "version": str,
    "active_agents": int,
    "active_workflows": int,
    "last_update": str
}
```

##### Portfolio Information
```http
GET /api/v1/portfolio

Response:
{
    "current_balance": float,
    "total_value": float,
    "total_pnl": float,
    "total_return": float,
    "active_positions": int,
    "positions": [
        {
            "market_id": str,
            "market_question": str,
            "outcome": str,
            "initial_stake": float,
            "current_value": float,
            "unrealized_pnl": float,
            "pnl_percentage": float,
            "days_held": int
        }
    ]
}
```

##### Agent Performance
```http
GET /api/v1/agents/performance

Response:
{
    "agents": [
        {
            "agent_name": str,
            "agent_role": str,
            "total_signals": int,
            "accuracy_rate": float,
            "total_pnl_contribution": float,
            "average_confidence": float,
            "last_updated": str
        }
    ]
}
```

##### Market Analysis
```http
GET /api/v1/markets/analysis

Query Parameters:
- limit: int (default: 10)
- sort: str (default: "probability")
- filter: str (optional)

Response:
{
    "markets": [
        {
            "id": str,
            "question": str,
            "probability": float,
            "volume": float,
            "fundamental_score": float,
            "technical_score": float,
            "sentiment_score": float,
            "last_updated": str
        }
    ],
    "total_count": int
}
```

##### Workflow Execution
```http
POST /api/v1/workflows/execute

Request Body:
{
    "market_id": str,
    "workflow_type": str,
    "force": bool  # Optional, bypass safety checks
}

Response:
{
    "workflow_id": str,
    "status": str,
    "message": str
}
```

##### Trading History
```http
GET /api/v1/trades/history

Query Parameters:
- limit: int (default: 50)
- start_date: str (ISO format)
- end_date: str (ISO format)
- market_id: str (optional)

Response:
{
    "trades": [
        {
            "trade_id": str,
            "market_id": str,
            "market_question": str,
            "action": str,
            "outcome": str,
            "amount": float,
            "price": float,
            "shares": float,
            "fee": float,
            "executed_at": str,
            "pnl": float
        }
    ],
    "total_count": int
}
```

### WebSocket API

#### Connection
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

// Authentication
ws.send(JSON.stringify({
    type: 'auth',
    api_key: 'your-api-key'
}));
```

#### Real-time Updates

##### Portfolio Updates
```javascript
// Subscribe to portfolio updates
ws.send(JSON.stringify({
    type: 'subscribe',
    channel: 'portfolio'
}));

// Receive updates
{
    "type": "portfolio_update",
    "data": {
        "total_value": float,
        "total_pnl": float,
        "active_positions": int,
        "timestamp": str
    }
}
```

##### Agent Status
```javascript
// Subscribe to agent status
ws.send(JSON.stringify({
    type: 'subscribe',
    channel: 'agents'
}));

// Receive updates
{
    "type": "agent_status",
    "data": {
        "agent_name": str,
        "status": str,  # "active", "idle", "error"
        "last_activity": str,
        "current_task": str
    }
}
```

##### Workflow Updates
```javascript
// Subscribe to workflow updates
ws.send(JSON.stringify({
    type: 'subscribe',
    channel: 'workflows'
}));

// Receive updates
{
    "type": "workflow_update",
    "data": {
        "workflow_id": str,
        "status": str,
        "progress": float,
        "current_stage": str,
        "timestamp": str
    }
}
```

##### Trade Execution
```javascript
// Subscribe to trade executions
ws.send(JSON.stringify({
    type: 'subscribe',
    channel: 'trades'
}));

// Receive updates
{
    "type": "trade_executed",
    "data": {
        "trade_id": str,
        "market_id": str,
        "action": str,
        "amount": float,
        "price": float,
        "status": str,
        "timestamp": str
    }
}
```

## Error Handling

### Error Response Format

```python
{
    "error": {
        "code": str,
        "message": str,
        "details": Dict[str, Any],
        "timestamp": str
    }
}
```

### Common Error Codes

| Code | Description |
|------|-------------|
| INVALID_API_KEY | API key is invalid or missing |
| INSUFFICIENT_PERMISSIONS | API key lacks required permissions |
| INVALID_REQUEST | Request format is invalid |
| RESOURCE_NOT_FOUND | Requested resource does not exist |
| RATE_LIMIT_EXCEEDED | API rate limit exceeded |
| INTERNAL_ERROR | Internal server error |
| VALIDATION_ERROR | Request validation failed |
| RISK_LIMIT_EXCEEDED | Trading request exceeds risk limits |
| MARKET_CLOSED | Market is no longer tradable |
| INSUFFICIENT_BALANCE | Insufficient balance for trade |

## Rate Limiting

### API Limits
- **REST API**: 100 requests per minute
- **WebSocket API**: 1000 messages per hour
- **Bulk Operations**: 10 requests per minute

### Rate Limit Headers
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640995200
```

## Data Models

### Market Model
```python
{
    "id": str,
    "creator_id": str,
    "question": str,
    "description": str,
    "created_time": datetime,
    "close_time": Optional[datetime],
    "resolution": Optional[str],
    "probability": float,
    "volume": float,
    "pool": Dict[str, float],
    "outcome_type": str,
    "mechanism": str,
    "tags": List[str],
    "url": str,
    "fundamental_score": Optional[float],
    "technical_score": Optional[float],
    "sentiment_score": Optional[float],
    "agent_analyses": Dict[str, Any]
}
```

### Agent Signal Model
```python
{
    "agent_name": str,
    "agent_role": str,
    "market_id": str,
    "action": str,  # "BUY", "SELL", "HOLD"
    "outcome": str,
    "confidence": float,  # 0.0 to 1.0
    "reasoning": str,
    "estimated_probability": Optional[float],
    "position_size": Optional[float],
    "timestamp": datetime,
    "supporting_evidence": List[str]
}
```

### Consensus Decision Model
```python
{
    "market_id": str,
    "final_action": str,  # "BUY", "SELL", "HOLD"
    "consensus_confidence": float,
    "participating_agents": List[str],
    "agent_signals": List[AgentSignal],
    "orchestrator_reasoning": str,
    "risk_approved": bool,
    "position_size": float,
    "timestamp": datetime,
    "execution_status": Optional[str]
}
```

### Position Model
```python
{
    "market_id": str,
    "market_question": str,
    "outcome": str,  # "YES" or "NO"
    "initial_stake": float,
    "initial_probability": float,
    "current_probability": float,
    "shares": float,
    "current_value": float,
    "unrealized_pnl": float,
    "opened_at": datetime,
    "last_updated": datetime
}
```

## SDK Examples

### Python SDK

```python
from manifold_swarms_bot import ManifoldSwarmsBot

# Initialize bot
bot = ManifoldSwarmsBot()

# Get portfolio status
portfolio = bot.get_portfolio()
print(f"Total P&L: {portfolio['total_pnl']}")

# Execute market analysis
result = bot.analyze_market("market_id", workflow_type="hierarchical")
print(f"Analysis result: {result['status']}")

# Get agent performance
performance = bot.get_agent_performance()
for agent in performance['agents']:
    print(f"{agent['agent_name']}: {agent['accuracy_rate']:.2%}")
```

### JavaScript SDK

```javascript
import { ManifoldSwarmsBot } from 'manifold-swarms-bot-sdk';

// Initialize client
const bot = new ManifoldSwarmsBot({
    apiKey: 'your-api-key',
    baseUrl: 'http://localhost:8000'
});

// Get portfolio
const portfolio = await bot.getPortfolio();
console.log('Portfolio:', portfolio);

// Subscribe to real-time updates
bot.subscribe('portfolio', (update) => {
    console.log('Portfolio update:', update);
});

// Execute workflow
const result = await bot.executeWorkflow({
    marketId: 'market_123',
    workflowType: 'hierarchical'
});
```

### CLI Tool

```bash
# Get status
manifold-swarms status

# Get portfolio
manifold-swarms portfolio

# Analyze market
manifold-swarms analyze --market-id market_123 --workflow hierarchical

# Get agent performance
manifold-swarms agents --performance

# Start dashboard
manifold-swarms dashboard

# Export logs
manifold-swarms logs --export --format csv
```

## Testing

### Unit Testing
```python
# Test agent communication
def test_agent_messaging():
    coordinator = AgentCoordinator()
    agent = TestAgent()
    
    message = AgentMessage(
        sender_agent="test",
        receiver_agent="target",
        message_type="REQUEST",
        content={"test": "data"}
    )
    
    response = await agent.process_message(message)
    assert response.message_type == "RESPONSE"
```

### Integration Testing
```python
# Test full workflow
async def test_hierarchical_workflow():
    bot = ManifoldSwarmsBot()
    market = create_test_market()
    
    result = await bot.workflow_manager.execute_market_analysis(
        market, workflow_type="hierarchical"
    )
    
    assert result['status'] == 'completed'
    assert 'consensus' in result
```

### Load Testing
```python
# Test API performance
def test_api_load():
    client = TestClient(app)
    
    # Simulate concurrent requests
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(client.get, '/api/v1/status')
            for _ in range(100)
        ]
        
        responses = [f.result() for f in futures]
        assert all(r.status_code == 200 for r in responses)
```

This API documentation provides comprehensive information for integrating with and extending the Manifold Swarms Trading Bot.