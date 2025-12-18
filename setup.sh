#!/bin/bash

# Manifold Swarms Trading Bot Setup Script

echo "🤖 Setting up Manifold Swarms Trading Bot..."

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs
mkdir -p agent_states
mkdir -p data
mkdir -p db

# Setup environment file
echo "⚙️ Setting up environment..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "📝 Please edit .env file with your API keys and configuration"
fi

# Setup database
echo "🗄️ Setting up database..."
python -c "
from src.utils.config import config
from pathlib import Path
Path('db').mkdir(exist_ok=True)
print('Database directory created')
"

# Run initial setup
echo "🔧 Running initial setup..."
python scripts/setup_swarms.py

echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Edit .env file with your API keys"
echo "2. Run 'source venv/bin/activate' to activate virtual environment"
echo "3. Run 'python main.py' to start the bot"
echo "4. Run 'streamlit run dashboard/app.py' to start the dashboard"
echo ""
echo "🚀 Happy trading!"