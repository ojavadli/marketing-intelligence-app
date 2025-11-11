# Marketing Intelligence Web App - Railway Deployment

## 🚀 Deployed Application

This is a Flask web application for marketing intelligence analysis using Reddit data.

## 📦 Files for Deployment

- `app.py` - Main Flask application
- `requirements.txt` - Python dependencies
- `Procfile` - Process configuration
- `railway.toml` - Railway-specific configuration
- `.gitignore` - Git ignore rules

## 🔧 Environment Variables Required

Set these in Railway dashboard:

```
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

## 🌐 Deployment Steps

### Option 1: Deploy via Railway Dashboard (Recommended)

1. Go to https://railway.app
2. Click "New Project" → "Deploy from GitHub repo"
3. Select this repository
4. Railway will auto-detect the configuration
5. Add environment variables in Settings
6. Deploy!

### Option 2: Deploy via Railway CLI

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Initialize project
railway init

# Add environment variables
railway variables set OPENAI_API_KEY=your_key
railway variables set TAVILY_API_KEY=your_key

# Deploy
railway up
```

## 🎯 Features

- Minimalistic Apple-style UI
- Real-time pipeline updates
- 7-step marketing intelligence analysis
- Reddit trend scraping
- AI-powered insights generation

## 📊 Pipeline Steps

1. Profile Analyzer - Research company profile
2. Keyword Generator - Generate Reddit search keywords  
3. Trend Scraper - Scrape Reddit posts (30s)
4. Ranking Agent - Extract insights from posts
5. Report Generator - Create comprehensive report
6. Summarizer - Save markdown file
7. Evaluator - LLM judge evaluation

## 🔗 Links

- **Deployed App**: Will be available at Railway URL
- **GitHub**: https://github.com/ojavadli/mymcp2

