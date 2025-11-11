# 🚀 Marketing Intelligence Web App

A beautiful, minimalistic web application that analyzes Reddit discussions to generate comprehensive marketing intelligence reports for any business.

## ✨ Features

- 🎨 **Apple-style minimalistic UI** - Clean, professional design
- ⚡ **Real-time updates** - Watch the analysis pipeline execute live
- 📊 **7-step intelligent analysis** - From profile research to LLM evaluation
- 🔍 **Reddit trend scraping** - Analyzes discussions from the past week
- 🤖 **AI-powered insights** - Uses GPT-4 for intelligent analysis
- ✅ **Auto-deploy** - Push to GitHub → Automatically updates on Railway

## 🎯 Live Demo

Deploy your own instance to Railway in 2 minutes!

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new)

## 🏗️ Architecture

### Pipeline Steps:

1. **Profile Analyzer** - Researches company using Tavily + GPT-4
2. **Keyword Generator** - Generates 50 Reddit search keywords
3. **Trend Scraper** - Scrapes Reddit posts for 30 seconds
4. **Ranking Agent** - Analyzes posts and extracts insights
5. **Report Generator** - Creates comprehensive marketing report
6. **Summarizer** - Saves report as markdown file
7. **Evaluator** - Runs LLM judge evaluation on 5 metrics

### Tech Stack:

- **Backend:** Flask + Gunicorn
- **AI:** OpenAI GPT-4, LangChain
- **Research:** Tavily API
- **Data:** Reddit public JSON API (no auth needed)
- **Frontend:** Vanilla HTML/CSS/JS with Apple-style design
- **Deployment:** Railway

## 🚀 Quick Start

### Deploy to Railway (Recommended)

1. Fork this repository
2. Go to [Railway](https://railway.app)
3. Click "New Project" → "Deploy from GitHub repo"
4. Select this repository
5. Add environment variables:
   - `OPENAI_API_KEY`
   - `TAVILY_API_KEY`
6. Deploy! ✅

### Run Locally

```bash
# Clone repository
git clone https://github.com/ojavadli/marketing-intelligence-app.git
cd marketing-intelligence-app

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your_key"
export TAVILY_API_KEY="your_key"

# Run app
python app.py

# Open http://localhost:5000
```

## 🔑 Environment Variables

Required:
- `OPENAI_API_KEY` - Your OpenAI API key
- `TAVILY_API_KEY` - Your Tavily API key

## 📖 Usage

1. Enter any business name (e.g., "Spotify", "Netflix", "Tesla")
2. Click "Run All"
3. Watch the 7-step pipeline execute in real-time
4. Get comprehensive marketing intelligence report in 2-3 minutes

## 🎨 Screenshots

### Input Interface
Clean input field with gradient purple button

### Real-time Pipeline
Each step shows:
- Purple border + gradient while running
- Green border + checkmark ✓ when complete
- Live output for each step

### Results
- Detailed business profile
- Reddit trends and insights
- Pain points analysis
- LLM evaluation scores

## 📊 Example Output

**For "Duolingo":**
- ✅ Industry: Language Learning
- ✅ 235 Reddit posts analyzed
- ✅ 15+ subreddits discovered
- ✅ 9 pain points identified
- ✅ 9 trends extracted
- ✅ Comprehensive 4000+ character report
- ✅ LLM evaluation: 0.85/1.0 average score

## 🔄 Auto-Deploy Workflow

```bash
# Make changes
git add -A
git commit -m "Update analysis"
git push

# Railway automatically redeploys! ✅
```

## 📁 Project Structure

```
marketing-intelligence-app/
├── app.py                  # Main Flask application
├── requirements.txt        # Python dependencies
├── Procfile               # Gunicorn configuration
├── railway.toml           # Railway deployment config
├── README.md              # This file
├── .gitignore             # Git ignore rules
└── marketing_intel_lean2.ipynb  # Original notebook
```

## 🎓 Academic Context

This project was developed for **CS329T: Trustworthy Machine Learning** at Stanford University.

**Key learnings:**
- LLM-as-judge evaluation (TruLens)
- Grounded report generation with citations
- Reddit data collection without API keys
- Production Flask deployment
- Real-time UI updates

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

MIT License - feel free to use for your projects!

## 🙏 Acknowledgments

- OpenAI for GPT-4 API
- Tavily for research API
- Reddit for public JSON API
- Railway for deployment platform

## 📧 Contact

**Orkhan Javadli** - [@ojavadli](https://github.com/ojavadli)

**Project Link:** https://github.com/ojavadli/marketing-intelligence-app

---

**⭐ Star this repo if you find it useful!**
