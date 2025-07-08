#!/bin/bash

# Medical AI Assistant Startup Script
# This script sets up the environment and starts the application

echo "🏥 Medical AI Assistant - Startup Script"
echo "========================================"

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed. Please install Python 3.8+ and try again."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📈 Upgrading pip..."
python3 -m pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p uploads
mkdir -p data/vector_store
mkdir -p data/documents
mkdir -p static

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found. Creating template..."
    cp .env.example .env
    echo "📝 Please edit .env file and add your GEMINI_API_KEY"
    echo "   You can get an API key from: https://aistudio.google.com/app/apikey"
    read -p "Press Enter to continue once you've set up your .env file..."
fi

# Verify environment variables
source .env

if [ -z "$GEMINI_API_KEY" ]; then
    echo "❌ Error: GEMINI_API_KEY is not set in .env file"
    echo "Please add your Google Gemini API key to the .env file:"
    echo "GEMINI_API_KEY=your_api_key_here"
    exit 1
fi

echo "✅ Environment setup complete!"
echo ""
echo "🚀 Starting Medical AI Assistant..."
echo "📍 Server will be available at: http://localhost:8000"
echo "📖 API Documentation: http://localhost:8000/docs"
echo "📋 Alternative Docs: http://localhost:8000/redoc"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the application
python3 main.py 