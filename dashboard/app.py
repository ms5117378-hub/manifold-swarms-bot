"""
Streamlit dashboard for monitoring the Manifold Swarms Trading Bot
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import asyncio
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.swarms_core.communication import AgentCoordinator
from src.swarms_core.workflows import TradingWorkflowManager
from src.manifold.market_fetcher import MarketDiscovery, MarketMonitor
from src.utils.config import config
from src.utils.logger import get_logger

log = get_logger(__name__)

# Initialize session state
if 'coordinator' not in st.session_state:
    st.session_state.coordinator = AgentCoordinator()
    st.session_state.workflow_manager = TradingWorkflowManager(st.session_state.coordinator)
    st.session_state.market_discovery = MarketDiscovery()
    st.session_state.market_monitor = MarketMonitor(st.session_state.market_discovery)

# Page configuration
st.set_page_config(
    page_title="Manifold Swarms Trading Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
    margin: 0.5rem 0;
}
.agent-status {
    padding: 0.5rem;
    border-radius: 0.25rem;
    margin: 0.25rem 0;
}
.status-active { background-color: #d4edda; color: #155724; }
.status-inactive { background-color: #f8d7da; color: #721c24; }
.status-idle { background-color: #fff3cd; color: #856404; }
</style>
""", unsafe_allow_html=True)

# Header
st.title("🤖 Manifold Swarms Trading Bot")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Control Panel")
    
    # System status
    st.subheader("System Status")
    system_status = st.selectbox(
        "System Status",
        ["Running", "Paused", "Stopped"],
        index=0
    )
    
    # Refresh controls
    st.subheader("Data Refresh")
    auto_refresh = st.checkbox("Auto Refresh", value=True)
    refresh_interval = st.selectbox(
        "Refresh Interval",
        [5, 10, 30, 60],
        index=1
    )
    
    if st.button("Refresh Data"):
        st.rerun()
    
    # Configuration
    st.subheader("Configuration")
    target_user = st.text_input(
        "Target User",
        value=config.get('trading.target_user', 'MikhailTal')
    )
    
    max_positions = st.number_input(
        "Max Positions",
        min_value=1,
        max_value=10,
        value=config.get('trading.max_active_positions', 5)
    )

# Main content
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h3>Active Positions</h3>
        <h2>3</h2>
        <small>Max: 5</small>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h3>Total P&L</h3>
        <h2>+M$45.30</h2>
        <small>+4.5%</small>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h3>Win Rate</h3>
        <h2>68%</h2>
        <small>17/25 trades</small>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <h3>Sharpe Ratio</h3>
        <h2>1.42</h2>
        <small>Risk-adjusted</small>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard", "🤖 Agents", "📈 Markets", "⚙️ Workflows", "📋 Logs"
])

with tab1:
    st.header("Portfolio Dashboard")
    
    # Portfolio performance chart
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Portfolio Performance")
        
        # Mock data for demonstration
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        portfolio_value = [1000 + i * 2.5 + (i % 7) * 10 for i in range(len(dates))]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=portfolio_value,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#1f77b4', width=2)
        ))
        
        fig.update_layout(
            title="Portfolio Value Over Time",
            xaxis_title="Date",
            yaxis_title="Portfolio Value (M$)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Asset Allocation")
        
        # Mock allocation data
        allocation_data = pd.DataFrame({
            'Asset': ['Stocks', 'Crypto', 'Sports', 'Politics', 'Cash'],
            'Allocation': [35, 25, 20, 15, 5]
        })
        
        fig = px.pie(
            allocation_data,
            values='Allocation',
            names='Asset',
            title='Portfolio Allocation'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent trades
    st.subheader("Recent Trades")
    
    # Mock trade data
    trade_data = pd.DataFrame({
        'Date': ['2024-01-15 14:30', '2024-01-15 12:15', '2024-01-14 16:45'],
        'Market': ['BTC > $50k by EOY', 'Election Winner', 'Sports Championship'],
        'Action': ['BUY', 'SELL', 'BUY'],
        'Amount': ['M$50', 'M$30', 'M$25'],
        'P&L': ['+M$5.20', '+M$3.10', '-M$1.50'],
        'Status': ['Active', 'Closed', 'Active']
    })
    
    st.dataframe(trade_data, use_container_width=True)

with tab2:
    st.header("Agent Status")
    
    # Agent overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Agent Performance")
        
        # Mock agent performance data
        agent_data = pd.DataFrame({
            'Agent': ['Fundamental Analyst', 'Technical Analyst', 'Sentiment Analyst',
                     'Value Investor', 'Momentum Trader', 'Mean Reversion', 'Arbitrage Finder'],
            'Accuracy': ['72%', '68%', '75%', '70%', '65%', '78%', '62%'],
            'Signals': [45, 42, 38, 35, 40, 33, 28],
            'Profit Contribution': ['+M$12.5', '+M$8.3', '+M$15.2', '+M$18.7', '+M$6.1', '+M$14.9', '+M$3.8']
        })
        
        st.dataframe(agent_data, use_container_width=True)
    
    with col2:
        st.subheader("Agent Communication")
        
        # Mock communication metrics
        comm_data = pd.DataFrame({
            'Metric': ['Messages Sent', 'Messages Received', 'Response Time', 'Consensus Rate'],
            'Value': ['1,247', '1,198', '2.3s', '73%']
        })
        
        for _, row in comm_data.iterrows():
            st.metric(row['Metric'], row['Value'])
    
    # Agent status details
    st.subheader("Detailed Agent Status")
    
    agent_status_data = pd.DataFrame({
        'Agent': ['Orchestrator', 'Risk Manager', 'Trade Executor', 'Portfolio Manager'],
        'Status': ['Active', 'Active', 'Active', 'Active'],
        'Last Activity': ['2 min ago', '1 min ago', '5 min ago', '3 min ago'],
        'CPU Usage': ['12%', '8%', '15%', '5%'],
        'Memory Usage': ['45MB', '32MB', '67MB', '28MB']
    })
    
    st.dataframe(agent_status_data, use_container_width=True)

with tab3:
    st.header("Market Analysis")
    
    # Market discovery
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Discovered Markets")
        
        # Mock market data
        market_data = pd.DataFrame({
            'Question': [
                'Will BTC exceed $100k by end of 2024?',
                'Who will win the 2024 US election?',
                'Will Lakers win NBA championship?',
                'S&P 500 > 5000 by EOY?'
            ],
            'Probability': [0.65, 0.52, 0.38, 0.71],
            'Volume': ['M$1,250', 'M$3,420', 'M$680', 'M$2,150'],
            'Days to Close': [350, 280, 45, 350],
            'Score': [0.82, 0.75, 0.68, 0.79]
        })
        
        st.dataframe(market_data, use_container_width=True)
    
    with col2:
        st.subheader("Market Sentiment")
        
        # Sentiment analysis
        sentiment_data = pd.DataFrame({
            'Market': ['BTC > $100k', 'Election', 'Lakers', 'S&P 500'],
            'Sentiment': ['Bullish', 'Neutral', 'Bearish', 'Bullish'],
            'Confidence': [0.78, 0.45, 0.62, 0.71],
            'News Impact': ['High', 'Medium', 'Low', 'High']
        })
        
        # Color code sentiment
        def color_sentiment(val):
            if val == 'Bullish':
                return 'background-color: #d4edda'
            elif val == 'Bearish':
                return 'background-color: #f8d7da'
            else:
                return 'background-color: #fff3cd'
        
        styled_sentiment = sentiment_data.style.applymap(color_sentiment, subset=['Sentiment'])
        st.dataframe(styled_sentiment, use_container_width=True)
    
    # Market correlation heatmap
    st.subheader("Market Correlations")
    
    # Mock correlation data
    correlation_matrix = pd.DataFrame({
        'BTC': [1.0, 0.3, -0.1, 0.6],
        'Election': [0.3, 1.0, 0.2, 0.4],
        'Sports': [-0.1, 0.2, 1.0, -0.2],
        'S&P': [0.6, 0.4, -0.2, 1.0]
    }, index=['BTC', 'Election', 'Sports', 'S&P'])
    
    fig = px.imshow(
        correlation_matrix,
        text_auto=True,
        aspect="auto",
        title="Market Correlation Matrix"
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Workflow Management")
    
    # Workflow statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Workflow Performance")
        
        workflow_stats = pd.DataFrame({
            'Workflow': ['Sequential', 'Parallel', 'Consensus', 'Hierarchical'],
            'Runs': [45, 38, 52, 67],
            'Success Rate': ['89%', '92%', '85%', '94%'],
            'Avg Duration': ['45s', '28s', '35s', '62s'],
            'Preferred': ['No', 'Yes', 'No', 'Yes']
        })
        
        st.dataframe(workflow_stats, use_container_width=True)
    
    with col2:
        st.subheader("Active Workflows")
        
        active_workflows = pd.DataFrame({
            'Workflow ID': ['run_abc123', 'run_def456'],
            'Market': ['BTC > $100k', 'Election Winner'],
            'Type': ['Hierarchical', 'Parallel'],
            'Status': ['Running', 'Completed'],
            'Progress': ['65%', '100%'],
            'Started': ['14:32:15', '14:28:42']
        })
        
        st.dataframe(active_workflows, use_container_width=True)
    
    # Workflow configuration
    st.subheader("Workflow Configuration")
    
    workflow_config = {
        'Primary Workflow': config.get('workflows.main_workflow', 'hierarchical'),
        'Analysis Pattern': config.get('workflows.analysis_pattern', 'agent_rearrange'),
        'Consensus Pattern': config.get('workflows.consensus_pattern', 'mixture_of_agents'),
        'Max Loops': config.get('workflows.max_loops', 1)
    }
    
    for key, value in workflow_config.items():
        st.metric(key, value)

with tab5:
    st.header("System Logs")
    
    # Log level filter
    log_level = st.selectbox(
        "Log Level",
        ["ALL", "ERROR", "WARNING", "INFO", "DEBUG"],
        index=0
    )
    
    # Mock log data
    log_data = pd.DataFrame({
        'Timestamp': [
            '2024-01-15 14:35:22',
            '2024-01-15 14:34:18',
            '2024-01-15 14:33:45',
            '2024-01-15 14:32:12',
            '2024-01-15 14:31:05'
        ],
        'Level': ['INFO', 'WARNING', 'INFO', 'ERROR', 'INFO'],
        'Agent': ['Trade Executor', 'Risk Manager', 'Portfolio Manager', 'Market Discovery', 'Orchestrator'],
        'Message': [
            'Trade executed successfully: BUY M$50 on BTC > $100k',
            'Position size reduced due to risk constraints',
            'Portfolio updated: 3 active positions, +4.5% P&L',
            'Failed to fetch market data: Connection timeout',
            'Consensus reached: BUY with 73% confidence'
        ]
    })
    
    # Color code log levels
    def color_log_level(val):
        if val == 'ERROR':
            return 'background-color: #f8d7da; color: #721c24'
        elif val == 'WARNING':
            return 'background-color: #fff3cd; color: #856404'
        elif val == 'INFO':
            return 'background-color: #d1ecf1; color: #0c5460'
        else:
            return 'background-color: #e2e3e5'
    
    styled_logs = log_data.style.applymap(color_log_level, subset=['Level'])
    st.dataframe(styled_logs, use_container_width=True)
    
    # Export logs
    if st.button("Export Logs"):
        csv = log_data.to_csv(index=False)
        st.download_button(
            label="Download logs as CSV",
            data=csv,
            file_name=f"trading_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# Auto refresh
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Manifold Swarms Trading Bot v1.0 | Last updated: {}</p>
    <p>Powered by Swarms.ai Framework</p>
</div>
""".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True)