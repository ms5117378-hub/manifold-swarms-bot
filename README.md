# Manifold Swarms Trading Bot

A sophisticated multi-agent trading system using the **Swarms.ai** framework to participate exclusively in markets created by user "MikhailTal" on Manifold Markets.

## 🚀 Overview

This project implements an intelligent trading bot that uses coordinated AI agents for market analysis, strategy development, risk management, and trade execution. The system leverages the power of multiple specialized agents working together to make informed trading decisions.

## 🏗️ Architecture

### Multi-Agent System

The bot consists of 11 specialized agents organized into departments:

#### 📊 Analysis Department
- **Fundamental Analyst**: Evaluates question clarity, resolution criteria, creator reputation
- **Technical Analyst**: Analyzes price patterns, volume, liquidity, momentum
- **Sentiment Analyst**: Performs web search, analyzes social sentiment, market correlations

#### 💡 Strategy Department  
- **Value Investor**: Identifies mispriced markets using fundamental analysis
- **Momentum Trader**: Follows trends and momentum signals
- **Mean Reversion Trader**: Takes contrarian positions on overreactions
- **Arbitrage Finder**: Discovers cross-market arbitrage opportunities

#### ⚙️ Operations Department
- **Trading Orchestrator**: Coordinates all agents, builds consensus, makes final decisions
- **Risk Manager**: Enforces risk limits, validates decisions, manages position sizing
- **Trade Executor**: Executes trades through Manifold API with retry logic
- **Portfolio Manager**: Monitors performance, generates reports, suggests rebalancing

### Workflow Patterns

The system implements multiple Swarms.ai workflow patterns:

1. **Sequential Workflow**: Linear execution through all stages
2. **AgentRearrange**: Parallel analysis with coordinated flow
3. **Mixture of Agents**: Consensus building with multiple rounds of refinement
4. **Hierarchical Workflow**: Department-based organization with clear reporting lines

## 🛠️ Installation

### Prerequisites

- Python 3.9+
- Manifold Markets API key
- OpenAI API key (or other LLM provider)

### Quick Setup

```bash
# Clone the repository
git clone <repository-url>
cd manifold-swarms-bot

# Run setup script
./setup.sh

# Activate virtual environment
source venv/bin/activate

# Configure your API keys
cp .env.example .env
# Edit .env with your keys

# Start the bot
python main.py
```

### Manual Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p logs agent_states data db

# Run setup
python scripts/setup_swarms.py
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file with the following:

```bash
# Required API Keys
MANIFOLD_API_KEY=your_manifold_api_key
OPENAI_API_KEY=your_openai_api_key

# Optional: Alternative LLM Providers
ANTHROPIC_API_KEY=your_anthropic_api_key
COHERE_API_KEY=your_cohere_api_key

# Trading Configuration
TARGET_USER_ID=MikhailTal
MAX_POSITION_SIZE=0.10
MAX_ACTIVE_POSITIONS=5
MIN_CONSENSUS_CONFIDENCE=0.65

# Risk Management
MAX_TOTAL_EXPOSURE=0.80
STOP_LOSS_THRESHOLD=-0.15
TAKE_PROFIT_THRESHOLD=0.25
```

### Configuration Files

- `config/config.yaml`: Main bot configuration
- `config/agents.yaml`: Agent system prompts and settings
- `config/workflows.yaml`: Workflow patterns and parameters
- `config/risk_limits.yaml`: Risk management rules and limits

## 🎯 Usage

### Starting the Bot

```bash
# Run the main trading bot
python main.py
```

### Monitoring Dashboard

```bash
# Start the Streamlit dashboard
streamlit run dashboard/app.py
```

The dashboard provides:
- Real-time portfolio performance
- Agent status and performance metrics
- Market analysis and sentiment
- Workflow execution statistics
- System logs and alerts

### Command Line Options

```bash
# Custom configuration
python main.py --config custom_config.yaml

# Dry run mode (no actual trades)
python main.py --dry-run

# Specific workflow
python main.py --workflow hierarchical

# Debug mode
python main.py --debug
```

## 📊 Agent Details

### Fundamental Analyst

**Role**: Market quality assessment
**Inputs**: Question text, description, creator history
**Outputs**: Quality score (0-1), confidence, reasoning
**Specialties**: 
- Question clarity analysis
- Resolution criteria evaluation
- Creator reputation scoring
- Researchability assessment

### Technical Analyst

**Role**: Price and volume analysis
**Inputs**: Historical price data, volume, order book
**Outputs**: Technical score (0-1), momentum indicators, support/resistance levels
**Specialties**:
- Trend analysis
- Momentum calculation
- Volume pattern recognition
- Liquidity assessment

### Sentiment Analyst

**Role**: Market sentiment and information analysis
**Inputs**: Market question, web search results, social media
**Outputs**: Sentiment score (0-1), direction, key drivers
**Specialties**:
- Web search integration
- Social media sentiment
- News analysis
- Contrarian opportunity identification

### Value Investor

**Role**: Mispricing identification
**Inputs**: Analysis scores, market fundamentals
**Outputs**: Trading signal, estimated true probability, position size
**Specialties**:
- Fundamental valuation
- Kelly criterion position sizing
- Margin of safety calculation
- Long-term value focus

### Momentum Trader

**Role**: Trend following
**Inputs**: Technical indicators, price history
**Outputs**: Momentum signals, trend strength, entry/exit timing
**Specialties**:
- Momentum scoring
- Trend identification
- Volume confirmation
- Risk-managed trend following

### Mean Reversion Trader

**Role**: Contrarian trading
**Inputs**: Price deviations, statistical measures
**Outputs**: Reversion signals, Z-scores, expected timing
**Specialties**:
- Statistical arbitrage
- Mean reversion timing
- Overreaction identification
- Risk-adjusted contrarian plays

### Arbitrage Finder

**Role**: Risk-free opportunity discovery
**Inputs**: Related markets, probability inconsistencies
**Outputs**: Arbitrage opportunities, hedge requirements, profit potential
**Specialties**:
- Cross-market arbitrage
- Logical arbitrage
- Probability inconsistency detection
- Risk-free profit calculation

## 🔧 Risk Management

### Portfolio-Level Controls

- **Maximum Total Exposure**: 80% of balance
- **Maximum Active Positions**: 5 positions
- **Minimum Balance Reserve**: 10% of balance
- **Maximum Single Position**: 10% of balance

### Position-Level Controls

- **Stop Loss**: -15% automatically close
- **Take Profit**: +25% partial closing
- **Time-based Exits**: Reduce exposure near resolution
- **Correlation Limits**: Diversification requirements

### Agent-Level Validation

- **Minimum Consensus**: 65% confidence required
- **Agent Agreement**: Minimum 2 agents must agree
- **Performance Weighting**: Better agents have more influence
- **Conflict Resolution**: Handle contradictory signals

## 📈 Performance Metrics

### Agent Performance

- **Signal Accuracy**: % of profitable signals
- **P&L Contribution**: Profit/loss attributed to agent
- **Consensus Participation**: How often agent aligns with consensus
- **Confidence Calibration**: Signal confidence vs actual outcomes

### System Performance

- **Total Return**: Overall portfolio performance
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: % of profitable trades
- **Average Trade Duration**: Time positions are held

### Workflow Efficiency

- **Success Rate**: % of workflows completed successfully
- **Execution Time**: Average time per workflow
- **Consensus Quality**: Agreement level among agents
- **Parallel Efficiency**: Speedup from concurrent execution

## 🔄 Workflow Examples

### Hierarchical Workflow (Default)

```
1. Analysis Department (Parallel)
   ├── Fundamental Analyst: Market quality assessment
   ├── Technical Analyst: Price/volume analysis  
   └── Sentiment Analyst: Sentiment analysis

2. Strategy Department (Parallel)
   ├── Value Investor: Fundamental mispricing
   ├── Momentum Trader: Trend following
   ├── Mean Reversion Trader: Contrarian signals
   └── Arbitrage Finder: Risk-free opportunities

3. Orchestrator Decision
   ├── Collect all agent signals
   ├── Weight by performance
   ├── Build consensus
   └── Generate final decision

4. Operations Department
   ├── Risk Manager: Validate decision
   ├── Trade Executor: Execute if approved
   └── Portfolio Manager: Update positions
```

### Consensus Building

```
Round 1: Strategy agents generate signals
- Value Investor: BUY @ 0.65 confidence
- Momentum Trader: BUY @ 0.72 confidence  
- Mean Reversion: SELL @ 0.58 confidence
- Arbitrage Finder: HOLD @ 0.45 confidence

Round 2: Orchestrator synthesizes
- Weight by historical performance
- Require 2+ agents agreeing
- Minimum 65% consensus confidence
- Final: BUY @ 0.68 confidence
```

## 🚨 Safety Features

### Error Handling

- **Retry Logic**: Up to 3 retries with exponential backoff
- **Circuit Breakers**: Stop trading on consecutive failures
- **Graceful Degradation**: Continue with reduced functionality
- **Comprehensive Logging**: All actions and errors logged

### Risk Controls

- **Pre-Trade Validation**: All decisions checked before execution
- **Real-Time Monitoring**: Continuous position and risk monitoring
- **Emergency Stops**: Manual and automatic shutdown capabilities
- **Position Limits**: Hard limits on exposure and position size

### Data Protection

- **Local Storage**: All data stored locally
- **Encrypted Configuration**: API keys encrypted at rest
- **Audit Trail**: Complete trading history maintained
- **Backup States**: Agent states regularly backed up

## 🐛 Troubleshooting

### Common Issues

**API Connection Errors**
```bash
# Check API keys
cat .env | grep API_KEY

# Test connection
python -c "import manifoldbot; print('API OK')"
```

**Agent Import Errors**
```bash
# Check Swarms installation
pip show swarms

# Reinstall if needed
pip install --upgrade swarms swarm-models
```

**Memory Issues**
```bash
# Monitor resource usage
htop

# Reduce concurrent workflows
# Edit config/workflows.yaml
```

### Debug Mode

```bash
# Enable debug logging
python main.py --debug

# Check logs
tail -f logs/manifold_swarms.log
```

### Performance Issues

```bash
# Monitor agent performance
streamlit run dashboard/app.py

# Check workflow statistics
python scripts/analyze_performance.py
```

## 🤝 Contributing

We welcome contributions! Key areas:

1. **New Agent Types**: Additional analysis or strategy agents
2. **Workflow Patterns**: New coordination mechanisms  
3. **Risk Management**: Enhanced risk controls
4. **Performance**: Optimization and speed improvements
5. **Documentation**: Improvements and examples

### Development Setup

```bash
# Clone development branch
git clone -b develop <repository-url>

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Code formatting
black src/
ruff check src/
```

### Pull Request Process

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit pull request with description

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Swarms.ai**: Multi-agent orchestration framework
- **Manifold Markets**: Prediction market platform
- **Manifoldbot**: Python API client
- **OpenAI**: LLM services
- **Streamlit**: Dashboard framework

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/ms5117378-hub/manifold-swarms-bot/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ms5117378-hub/manifold-swarms-bot/discussions)
- **Contacts**:  [Freelancer](https://www.freelancer.com/solutionslanguag)

## Warning

Currently this project build for [Freelancer.com](https://www.freelancer.com/contest/Best-OpenSource-Judgmental-Prediction-Python-Repository-2658923)

---

**Happy Trading! 🚀**
