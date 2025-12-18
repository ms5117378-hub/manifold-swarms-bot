# Architecture Documentation

## System Overview

The Manifold Swarms Trading Bot is a sophisticated multi-agent system that uses the Swarms.ai framework to coordinate specialized AI agents for prediction market trading. The system is designed to trade exclusively on markets created by user "MikhailTal" on Manifold Markets.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Manifold Swarms Bot                    │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Swarms Framework Layer               │   │
│  │  ┌─────────────┐  ┌─────────────┐            │   │
│  │  │ Sequential   │  │ AgentRearrange│            │   │
│  │  │ Workflow    │  │   Pattern    │            │   │
│  │  └─────────────┘  └─────────────┘            │   │
│  │  ┌─────────────┐  ┌─────────────┐            │   │
│  │  │Mixture of   │  │Hierarchical │            │   │
│  │  │  Agents     │  │ Workflow    │            │   │
│  │  └─────────────┘  └─────────────┘            │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Agent Coordination                │   │
│  │  ┌─────────────┐  ┌─────────────┐            │   │
│  │  │ Message     │  │ Consensus   │            │   │
│  │  │  Broker     │  │  Builder    │            │   │
│  │  └─────────────┘  └─────────────┘            │   │
│  │  ┌─────────────┐  ┌─────────────┐            │   │
│  │  │ Workflow    │  │ Performance │            │   │
│  │  │ Manager     │  │  Tracking   │            │   │
│  │  └─────────────┘  └─────────────┘            │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                 Agent Swarm                    │   │
│  │  ┌─────────────────────────────────────────────┐ │   │
│  │  │        Analysis Department               │ │   │
│  │  │ ┌─────────┐ ┌─────────┐ ┌─────────┐ │ │   │
│  │  │ │Fundamental│ │Technical│ │Sentiment│ │ │   │
│  │  │ Analyst  │ │ Analyst  │ │ Analyst  │ │ │   │
│  │  │ └─────────┘ └─────────┘ └─────────┘ │ │   │
│  │  └─────────────────────────────────────────────┘ │   │
│  │  ┌─────────────────────────────────────────────┐ │   │
│  │  │        Strategy Department                │ │   │
│  │  │ ┌─────────┐ ┌─────────┐ ┌─────────┐ │ │   │
│  │  │ │  Value  │ │Momentum│ │Mean Rev.│ │ │   │
│  │  │ │Investor │ │ Trader  │ │ Trader  │ │ │   │
│  │  │ └─────────┘ └─────────┘ └─────────┘ │ │   │
│  │  │ ┌─────────┐                         │ │   │
│  │  │ │Arbitrage│                         │ │   │
│  │  │ │ Finder  │                         │ │   │
│  │  │ └─────────┘                         │ │   │
│  │  └─────────────────────────────────────────────┘ │   │
│  │  ┌─────────────────────────────────────────────┐ │   │
│  │  │       Operations Department               │ │   │
│  │  │ ┌─────────┐ ┌─────────┐ ┌─────────┐ │ │   │
│  │  │ │Orchestra│ │  Risk   │ │ Trade   │ │ │   │
│  │  │ │   tor   │ │ Manager │ │Executor │ │ │   │
│  │  │ └─────────┘ └─────────┘ └─────────┘ │ │   │
│  │  │ ┌─────────┐                         │ │   │
│  │  │ │Portfolio│                         │ │   │
│  │  │ │ Manager │                         │ │   │
│  │  │ └─────────┘                         │ │   │
│  │  └─────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              External Interfaces                │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  │   │
│  │  │Manifold │  │Dashboard│  │  Logs   │  │   │
│  │  │   API   │  │Streamlit│  │ System  │  │   │
│  │  └─────────┘  └─────────┘  └─────────┘  │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Swarms Framework Layer

The foundation of the system, providing multi-agent orchestration capabilities.

#### Sequential Workflow
- Linear execution through predefined agent sequence
- Ensures ordered processing: Analysis → Strategy → Risk → Execution
- Used for simple, straightforward trading scenarios

#### AgentRearrange Pattern
- Parallel execution with coordinated flow
- Allows multiple agents to work simultaneously
- Defines complex agent interaction patterns

#### Mixture of Agents
- Consensus building with multiple refinement rounds
- Weights agent inputs by historical performance
- Handles conflicting signals intelligently

#### Hierarchical Workflow
- Department-based organization
- Bottom-up analysis, top-down decisions
- Mirrors real trading desk structure

### 2. Agent Coordination Layer

Manages communication and collaboration between agents.

#### Message Broker
- Asynchronous message passing
- Topic-based subscriptions
- Priority message handling
- Message history and persistence

#### Consensus Builder
- Multi-agent decision synthesis
- Performance-weighted voting
- Conflict resolution mechanisms
- Confidence threshold enforcement

#### Workflow Manager
- Workflow lifecycle management
- Error handling and recovery
- Performance tracking
- Dynamic workflow selection

### 3. Agent Swarm

Eleven specialized agents organized into three departments.

#### Analysis Department
**Fundamental Analyst**
- Market quality assessment
- Question clarity evaluation
- Creator reputation analysis
- Resolution criteria validation

**Technical Analyst**
- Price pattern analysis
- Volume and liquidity assessment
- Momentum calculation
- Support/resistance identification

**Sentiment Analyst**
- Web search integration
- Social media analysis
- News sentiment processing
- Market correlation analysis

#### Strategy Department
**Value Investor**
- Fundamental mispricing identification
- Kelly criterion position sizing
- Margin of safety calculation
- Long-term value focus

**Momentum Trader**
- Trend following strategies
- Momentum signal generation
- Volume confirmation
- Risk-managed trend trading

**Mean Reversion Trader**
- Statistical arbitrage
- Overreaction identification
- Z-score analysis
- Contrarian positioning

**Arbitrage Finder**
- Cross-market arbitrage
- Logical arbitrage detection
- Probability inconsistency analysis
- Risk-free opportunity identification

#### Operations Department
**Trading Orchestrator**
- Agent coordination
- Final decision making
- Workflow management
- System monitoring

**Risk Manager**
- Risk limit enforcement
- Position sizing validation
- Portfolio risk monitoring
- Stop-loss/take-profit execution

**Trade Executor**
- API trade execution
- Retry logic implementation
- Error handling
- Transaction confirmation

**Portfolio Manager**
- Performance tracking
- P&L calculation
- Rebalancing recommendations
- Reporting generation

### 4. External Interfaces

Connections to external systems and user interfaces.

#### Manifold API
- Market data retrieval
- Trade execution
- Position monitoring
- Account information

#### Dashboard
- Real-time monitoring
- Performance visualization
- Agent status display
- System control

#### Logging System
- Structured logging
- Performance metrics
- Error tracking
- Audit trail

## Data Flow

### Market Analysis Flow

```
1. Market Discovery
   ├── Fetch MikhailTal markets
   ├── Filter tradable markets
   └── Update market cache

2. Analysis Department (Parallel)
   ├── Fundamental Analyst: Quality score
   ├── Technical Analyst: Technical indicators
   └── Sentiment Analyst: Sentiment score

3. Strategy Department (Parallel)
   ├── Value Investor: Mispricing signals
   ├── Momentum Trader: Trend signals
   ├── Mean Reversion: Contrarian signals
   └── Arbitrage Finder: Arbitrage opportunities

4. Consensus Building
   ├── Collect all strategy signals
   ├── Weight by agent performance
   ├── Apply consensus rules
   └── Generate final decision

5. Risk Validation
   ├── Check portfolio limits
   ├── Validate position size
   ├── Apply risk rules
   └── Approve/reject decision

6. Trade Execution
   ├── Execute approved trades
   ├── Handle retries
   ├── Confirm execution
   └── Update positions

7. Portfolio Management
   ├── Track performance
   ├── Monitor risk
   ├── Generate reports
   └── Suggest rebalancing
```

### Agent Communication Flow

```
Agent Message Flow:
┌─────────────┐    Message     ┌─────────────┐
│   Agent A   │ ──────────────► │   Agent B   │
│             │               │             │
│ Message     │ ◄────────────── │ Response    │
│ Queue       │    Response    │ Queue       │
└─────────────┘               └─────────────┘
       │                             │
       ▼                             ▼
┌─────────────┐               ┌─────────────┐
│   Message   │               │   Message   │
│   Broker    │               │   Broker    │
└─────────────┘               └─────────────┘
```

## Configuration Architecture

### Configuration Hierarchy

```
Configuration Sources (Priority Order):
1. Environment Variables
2. .env file
3. config/*.yaml files
4. Default values
```

### Configuration Files

#### config/config.yaml
- Main bot configuration
- Trading parameters
- Workflow settings
- Monitoring options

#### config/agents.yaml
- Agent system prompts
- Agent-specific settings
- Role definitions
- Performance weights

#### config/workflows.yaml
- Workflow patterns
- Agent sequences
- Consensus rules
- Timeout settings

#### config/risk_limits.yaml
- Risk management rules
- Position limits
- Stop-loss settings
- Portfolio constraints

## Security Architecture

### API Key Management
- Environment variable storage
- Encrypted configuration files
- Runtime key validation
- Key rotation support

### Risk Controls
- Pre-trade validation
- Real-time monitoring
- Emergency stop mechanisms
- Position limit enforcement

### Data Protection
- Local data storage
- Encrypted sensitive data
- Access logging
- Audit trails

## Performance Architecture

### Scalability Features
- Asynchronous agent execution
- Parallel market analysis
- Configurable concurrency limits
- Resource usage monitoring

### Optimization Strategies
- Agent performance weighting
- Workflow caching
- Connection pooling
- Batch processing

### Monitoring Capabilities
- Real-time performance metrics
- Agent health monitoring
- Workflow execution tracking
- System resource usage

## Deployment Architecture

### Development Environment
- Local execution
- Debug logging
- Mock data support
- Hot reloading

### Production Environment
- Containerized deployment
- Process supervision
- Log aggregation
- Health checks

### High Availability
- Graceful error handling
- Automatic recovery
- State persistence
- Backup mechanisms

This architecture provides a robust, scalable, and maintainable foundation for sophisticated multi-agent trading on prediction markets.