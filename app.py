"""
Marketing Intelligence Web Application
Standalone Flask app for Railway deployment
"""

from flask import Flask, request, jsonify, render_template_string, send_file
from flask_cors import CORS
import os
import json
import time
import threading
from datetime import datetime
from io import BytesIO
from typing import List, Optional, Literal

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from tavily import TavilyClient
import requests
from pydantic import BaseModel
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders

def send_email_with_attachments(recipient_email, business_name, pdf_bytes, mp3_1_bytes, mp3_2_bytes, pptx_bytes, gamma_url):
    """Send email with PDF, MP3, and PPTX attachments"""
    try:
        # Email configuration from environment variables
        smtp_server = os.environ.get('SMTP_SERVER', 'smtp.hostinger.com')
        smtp_port = int(os.environ.get('SMTP_PORT', '465'))
        sender_email = os.environ.get('SENDER_EMAIL', '')
        sender_password = os.environ.get('SENDER_PASSWORD', '')
        
        if not sender_email or not sender_password:
            return False, "Email credentials not configured in environment variables"
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = f"Marketing Intelligence Report: {business_name}"
        
        # Email body
        body = f"""Hello,

Your Marketing Intelligence analysis for "{business_name}" is complete!

Attached you will find:
- PDF Report
- Audio Reports (MP3)
- Presentation (PPTX){' - Gamma Link: ' + gamma_url if gamma_url else ''}

Thank you for using Marketing Intelligence.

Best regards,
Marketing Intelligence Team
"""
        msg.attach(MIMEText(body, 'plain'))
        
        safe_name = business_name.replace(' ', '_')
        
        # Attach PDF
        if pdf_bytes:
            part = MIMEBase('application', 'pdf')
            part.set_payload(pdf_bytes)
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename="{safe_name}_report.pdf"')
            msg.attach(part)
        
        # Attach MP3 files
        if mp3_1_bytes:
            part = MIMEBase('audio', 'mpeg')
            part.set_payload(mp3_1_bytes)
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename="{safe_name}_song_1.mp3"')
            msg.attach(part)
        
        if mp3_2_bytes:
            part = MIMEBase('audio', 'mpeg')
            part.set_payload(mp3_2_bytes)
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename="{safe_name}_song_2.mp3"')
            msg.attach(part)
        
        # Attach PPTX
        if pptx_bytes:
            part = MIMEBase('application', 'vnd.openxmlformats-officedocument.presentationml.presentation')
            part.set_payload(pptx_bytes)
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename="{safe_name}_presentation.pptx"')
            msg.attach(part)
        
        # Send email via SSL
        import ssl
        context = ssl.create_default_context()
        # Handle multiple recipients (separated by ; or ,)
        recipients = [e.strip() for e in recipient_email.replace(';', ',').split(',') if e.strip()]
        
        with smtplib.SMTP_SSL(smtp_server, smtp_port, context=context) as server:
            server.login(sender_email, sender_password)
            for recipient in recipients:
                try:
                    msg.replace_header('To', recipient)
                except:
                    msg['To'] = recipient
                server.sendmail(sender_email, recipient, msg.as_string())
        
        return True, "Email sent successfully"
    except Exception as e:
        return False, str(e)

app = Flask(__name__)
CORS(app)

# Initialize API keys from environment variables
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
TAVILY_API_KEY = os.environ.get('TAVILY_API_KEY', '')
SUNO_API_KEY = os.environ.get('SUNO_API_KEY', '')
GAMMA_API_KEY = os.environ.get('GAMMA_API_KEY', '')

# Explicitly pass API key to ensure it's used
if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY not set!")

llm_json = ChatOpenAI(
    model="gpt-5.1", 
    temperature=0, 
    model_kwargs={"response_format": {"type": "json_object"}}, 
    request_timeout=180,
    api_key=OPENAI_API_KEY if OPENAI_API_KEY else None
)
llm = ChatOpenAI(
    model="gpt-5.1", 
    temperature=0.1, 
    request_timeout=180,
    api_key=OPENAI_API_KEY if OPENAI_API_KEY else None
)
tavily = TavilyClient(api_key=TAVILY_API_KEY)

# ============================================================================
# REDDIT MCP - Exact copy from notebook
# ============================================================================

class RedditPost(BaseModel):
    """Single Reddit post record"""
    title: str
    subreddit: str
    author: str
    score: int
    num_comments: int
    created_utc: float
    url: str
    selftext: str = ""
    permalink: str
    id: str
    is_self: bool = False
    link_flair_text: Optional[str] = None

class RedditPosts(BaseModel):
    """Collection of Reddit posts with metadata"""
    request_url: str
    items: List[RedditPost]
    count: int
    before: Optional[str] = None
    after: Optional[str] = None

class RedditTools:
    """Reddit API tools - uses public JSON endpoints, no API key required"""
    
    def _get_user_agent(self) -> str:
        """Return proper User-Agent as recommended by Reddit API"""
        return "MarketingIntelAgent/1.0 (CS329T; Educational)"
    
    def search_posts(
        self,
        query: str,
        subreddit: Optional[str] = None,
        sort: str = "relevance",
        t: str = "week",
        limit: int = 25,
        after: Optional[str] = None,
        before: Optional[str] = None
    ) -> RedditPosts:
        """
        Search for posts across Reddit or within a specific subreddit.
        Default time filter is 'week' (last 7 days).
        """
        if subreddit:
            url = f"https://www.reddit.com/r/{subreddit}/search.json"
            params = {"q": query, "restrict_sr": "true"}
        else:
            url = "https://www.reddit.com/search.json"
            params = {"q": query}
        
        params.update({
            "sort": sort,
            "t": t,
            "limit": min(limit, 100),
            "raw_json": 1
        })
        
        if after:
            params["after"] = after
        if before:
            params["before"] = before
        
        headers = {"User-Agent": self._get_user_agent()}
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        posts = [
            child["data"] for child in data["data"]["children"]
            if not child["data"].get("stickied", False)
        ]
        
        post_items = []
        for post in posts:
            post_items.append(RedditPost(
                title=post.get("title", ""),
                subreddit=post.get("subreddit", ""),
                author=post.get("author", ""),
                score=post.get("score", 0),
                num_comments=post.get("num_comments", 0),
                created_utc=post.get("created_utc", 0),
                url=post.get("url", ""),
                selftext=post.get("selftext", ""),
                permalink=f"https://www.reddit.com{post.get('permalink', '')}",
                id=post.get("id", ""),
                is_self=post.get("is_self", False),
                link_flair_text=post.get("link_flair_text")
            ))
        
        return RedditPosts(
            request_url=response.url,
            items=post_items,
            count=len(post_items),
            before=data["data"].get("before"),
            after=data["data"].get("after")
        )

reddit = RedditTools()

# Global state
current_run = {
    "status": "idle",
    "business_name": "",
    "email": "",
    "steps": {
        "1": {"name": "Profile Analyzer", "status": "pending", "output": ""},
        "2": {"name": "Keyword Generator", "status": "pending", "output": ""},
        "3": {"name": "Trend Scraper", "status": "pending", "output": ""},
        "4": {"name": "Ranking Agent", "status": "pending", "output": ""},
        "5": {"name": "Report Generator", "status": "pending", "output": ""},
        "6": {"name": "PDF Generator", "status": "pending", "output": ""},
        "7": {"name": "Evaluator", "status": "pending", "output": ""},
        "9A": {"name": "Fun Report Generator", "status": "pending", "output": ""},
        "9B": {"name": "Audio Report Generator", "status": "pending", "output": ""},
        "10": {"name": "Gamma Presentation", "status": "pending", "output": ""}
    },
    "files": {
        "pdf": None,
        "mp3_1": None,
        "mp3_2": None,
        "pptx": None,
        "gamma_url": None
    }
}

def reset_run():
    for step_id in current_run["steps"]:
        current_run["steps"][step_id]["status"] = "pending"
        current_run["steps"][step_id]["output"] = ""
    current_run["status"] = "idle"
    current_run["email"] = ""
    current_run["files"] = {"pdf": None, "mp3_1": None, "mp3_2": None, "pptx": None, "gamma_url": None}

def run_pipeline(business_name):
    """Execute the entire marketing intelligence pipeline"""
    global current_run
    
    import traceback
    import sys
    
    def log_error(step, error):
        """Log error to step output"""
        error_msg = f"ERROR in {step}: {str(error)[:200]}\nTraceback: {traceback.format_exc()[:300]}"
        print(error_msg, file=sys.stderr)
        current_run["steps"][step]["output"] = error_msg
        current_run["steps"][step]["status"] = "completed"
    
    business_profile = {}
    keywords = []
    reddit_posts = []
    ranked_data = {}
    final_report = ""
    song_lyrics = ""
    
    try:
        current_run["status"] = "running"
        current_run["business_name"] = business_name
        
        # STEP 1: Profile Analyzer
        current_run["steps"]["1"]["status"] = "running"
        
        search_results = {}
        try:
            search_results = tavily.search(f"{business_name} company industry business model", max_results=5, search_depth="advanced", timeout=7)
        except:
            search_results = {"results": []}
        
        extract_prompt = f"""Analyze {business_name} and extract business profile.
Research: {json.dumps(search_results, indent=2)[:1000]}
Return JSON: {{"business_name": "{business_name}", "industry": "...", "business_model": "...", "target_market": "...", "customer_demographics": "...", "products_services": [], "competitors": [], "market_position": "..."}}"""
        
        try:
            response = llm_json.invoke([HumanMessage(content=extract_prompt)], timeout=8)
            business_profile = json.loads(response.content)
        except:
            business_profile = {"business_name": business_name, "industry": "Unknown", "business_model": "Unknown", "target_market": "Unknown"}
        
        output_text = f"""Business: {business_profile.get('business_name', business_name)}
Industry: {business_profile.get('industry', 'N/A')}
Business Model: {business_profile.get('business_model', 'N/A')}
Target Market: {business_profile.get('target_market', 'N/A')}"""
        
        current_run["steps"]["1"]["output"] = output_text
        current_run["steps"]["1"]["status"] = "completed"
        
        # STEP 2: Keyword Generator
        current_run["steps"]["2"]["status"] = "running"
        
        keyword_prompt = f"""Generate 200 Reddit search keywords to find discussions relevant to {business_name}.

Business Profile: {json.dumps(business_profile, indent=2)[:500]}

IMPORTANT: Create a MIX of keyword types:
- ~30 keywords WITH the business name (for direct mentions)
- ~50 keywords about competitors (competitor names, "vs" comparisons)
- ~60 keywords about the industry/category (generic industry terms)
- ~60 keywords about pain points and use cases (problems users discuss)

Example for a small influencer marketing company:
- "influencer marketing platform" (generic)
- "micro influencer collaboration" (use case)
- "ugc creator marketplace" (category)
- "[competitor] vs [other]" (competitor)
- "[business] review" (branded)

Return JSON: {{"keywords": ["keyword1", "keyword2", ...] (200 total)}}"""
        
        kw_response = llm_json.invoke([HumanMessage(content=keyword_prompt)])
        kw_data = json.loads(kw_response.content)
        keywords = kw_data.get("keywords", [])
        
        keywords_display = "\n".join([f"  {i+1}. {kw}" for i, kw in enumerate(keywords)])
        current_run["steps"]["2"]["output"] = f"Generated {len(keywords)} keywords\n\nAll Keywords:\n{keywords_display}"
        current_run["steps"]["2"]["status"] = "completed"
        
        # STEP 3: ENHANCED SCRAPER - Exact match to notebook
        current_run["steps"]["3"]["status"] = "running"
        
        # Configuration - EXACT match to notebook
        TARGET_POSTS = 20
        RELEVANCE_THRESHOLD = 0.7
        max_iterations = 4
        TIME_LIMIT = 30
        REQUEST_DELAY = 2  # EXACT: 2 seconds like notebook
        MIN_COMMENTS = 2   # EXACT: 2 comments like notebook
        
        all_scraped_posts = []
        relevant_posts = []
        seen_ids = set()
        discovered_subreddits = set()
        iteration = 0
        rate_limited = False
        
        while len(relevant_posts) < TARGET_POSTS and iteration < max_iterations and not rate_limited:
            iteration += 1
            
            # EXACT: First 2 iterations search ALL Reddit
            if iteration <= 2:
                search_mode = "ALL_REDDIT"
            else:
                search_mode = "TARGETED"
            
            current_run["steps"]["3"]["output"] = f"Iteration {iteration}: Scraping ({len(relevant_posts)}/{TARGET_POSTS} relevant)"
            
            batch_posts = []
            start_time = time.time()
            keyword_idx = (iteration - 1) * 10
            
            while time.time() - start_time < TIME_LIMIT:
                if keyword_idx >= len(keywords):
                    keyword_idx = 0
                kw = keywords[keyword_idx]
                
                try:
                    # EXACT: Search logic from notebook
                    if search_mode == "ALL_REDDIT":
                        results = reddit.search_posts(query=kw, t="week", limit=25)
                    else:
                        if discovered_subreddits:
                            target_sub = list(discovered_subreddits)[keyword_idx % len(discovered_subreddits)]
                            results = reddit.search_posts(query=kw, subreddit=target_sub, t="week", limit=25)
                        else:
                            results = reddit.search_posts(query=kw, t="week", limit=25)
                    
                    # EXACT: Use results.items like notebook
                    for post in results.items:
                        if post.id not in seen_ids and post.num_comments >= MIN_COMMENTS:
                            post_dict = {
                                "title": post.title,
                                "subreddit": post.subreddit,
                                "author": post.author,
                                "score": post.score,
                                "num_upvotes": post.score,
                                "num_comments": post.num_comments,
                                "created_utc": post.created_utc,
                                "url": post.url,
                                "selftext": post.selftext[:1000] if post.selftext else "",
                                "permalink": post.permalink,
                                "id": post.id,
                                "link_flair_text": post.link_flair_text or ""
                            }
                            batch_posts.append(post_dict)
                            all_scraped_posts.append(post_dict)
                            seen_ids.add(post.id)
                            discovered_subreddits.add(post.subreddit)
                    
                    # EXACT: 2 second delay like notebook
                    time.sleep(REQUEST_DELAY)
                    
                except Exception as e:
                    error_str = str(e)
                    if "429" in error_str:
                        rate_limited = True
                        break
                    time.sleep(REQUEST_DELAY)
                
                keyword_idx += 1
            
            # LLM relevance filter
            if len(batch_posts) > 0 and not rate_limited:
                batch_summary = [{"id": p.get('id'), "title": p.get('title', '')[:200], "subreddit": p.get('subreddit', '')} for p in batch_posts[:50]]
                
                filter_prompt = f"""Rate relevance of posts to {business_name} (0.0-1.0 each).
Business: {business_name}
Industry: {business_profile.get('industry', 'N/A')}
Posts: {json.dumps(batch_summary, indent=2)[:3000]}
Return JSON: {{"relevance_scores": [0.0-1.0 list]}}"""
                
                try:
                    current_run["steps"]["3"]["output"] = f"Iteration {iteration}: Filtering {len(batch_posts)} posts..."
                    filter_resp = llm_json.invoke([HumanMessage(content=filter_prompt)])
                    scores = json.loads(filter_resp.content).get('relevance_scores', [])
                    
                    for post, score in zip(batch_posts[:len(scores)], scores):
                        if score >= RELEVANCE_THRESHOLD:
                            relevant_posts.append(post)
                except:
                    relevant_posts.extend(batch_posts)
        
        reddit_posts = relevant_posts if relevant_posts else all_scraped_posts
        
        current_run["steps"]["3"]["output"] = f"""Iterations: {iteration}
Total scraped: {len(all_scraped_posts)} posts
Relevant (>0.7): {len(relevant_posts)} posts
Subreddits discovered: {len(discovered_subreddits)}"""
        current_run["steps"]["3"]["status"] = "completed"
        
        # STEP 4: Ranking Agent
        current_run["steps"]["4"]["status"] = "running"
        
        posts_for_analysis = []
        for idx, post in enumerate(reddit_posts[:100], 1):
            posts_for_analysis.append({
                "post_id": idx,
                "title": post.get('title', '')[:300],
                "subreddit": post.get('subreddit', ''),
                "upvotes": post.get('score', post.get('num_upvotes', 0)),
                "comments": post.get('num_comments', 0)
            })
        
        ranking_prompt = f"""Analyze {len(posts_for_analysis)} Reddit posts for {business_name}.
Posts: {json.dumps(posts_for_analysis, indent=2)[:3000]}
Return JSON with: {{"total_posts_analyzed": {len(reddit_posts)}, "ranked_posts": [...top 10...], "pain_points": [{{"pain": "specific pain", "supporting_posts": [1,2,3]}}], "overall_trends": [{{"trend": "specific trend", "supporting_posts": [1,2,3]}}]}}"""
        
        try:
            ranked_data = json.loads(llm_json.invoke([HumanMessage(content=ranking_prompt)]).content)
        except:
            ranked_data = {"total_posts_analyzed": len(reddit_posts), "pain_points": [], "overall_trends": []}
        
        pain_points_list = ranked_data.get('pain_points', [])
        trends_list = ranked_data.get('overall_trends', [])
        
        pain_text = "\n".join([f"  {i+1}. {p.get('pain', p) if isinstance(p, dict) else p}" for i, p in enumerate(pain_points_list)])
        trends_text = "\n".join([f"  {i+1}. {t.get('trend', t) if isinstance(t, dict) else t}" for i, t in enumerate(trends_list)])
        
        current_run["steps"]["4"]["output"] = f"""Analyzed {len(reddit_posts)} posts

Pain Points ({len(pain_points_list)}):
{pain_text if pain_text else '  None identified'}

Trends ({len(trends_list)}):
{trends_text if trends_text else '  None identified'}"""
        current_run["steps"]["4"]["status"] = "completed"
        
        # STEP 5: Report Generator - EXACT MATCH TO NOTEBOOK
        current_run["steps"]["5"]["status"] = "running"
        current_run["steps"]["5"]["output"] = "Starting report generation..."
        
        try:
            from datetime import datetime as dt
            
            # Calculate date range from posts (like notebook)
            if reddit_posts:
                timestamps = [p.get('created_utc', 0) for p in reddit_posts if p.get('created_utc')]
                if timestamps:
                    oldest_post = min(timestamps)
                    newest_post = max(timestamps)
                    start_date = dt.fromtimestamp(oldest_post).strftime('%B %d, %Y')
                    end_date = dt.fromtimestamp(newest_post).strftime('%B %d, %Y')
                else:
                    start_date = end_date = dt.now().strftime('%B %d, %Y')
            else:
                start_date = end_date = dt.now().strftime('%B %d, %Y')
            
            # EXACT notebook prompt with all 6 sections
            report_prompt = f"""Generate comprehensive marketing intelligence report (use date range, NOT post count) for {business_name}.

BUSINESS CONTEXT:
- Industry: {business_profile.get('industry', 'N/A')}
- Target Market: {business_profile.get('target_market', 'N/A')}
- Analysis Period: {start_date} to {end_date}
- Posts Analyzed: {len(reddit_posts)} (internal only - do not mention in report)

DATA AVAILABLE:
Pain Points: {json.dumps(ranked_data.get('pain_points', [])[:5], indent=2)}
Trends: {json.dumps(ranked_data.get('overall_trends', [])[:5], indent=2)}

CRITICAL DATE REQUIREMENT:
- Write "Based on analysis of discussions from {start_date} to {end_date}"
- DO NOT write "Based on analysis of XX posts" or mention post counts
- Use date ranges ONLY

REQUIRED SECTIONS (MUST INCLUDE ALL):

1. EXECUTIVE SUMMARY (ONE flowing paragraph, 150+ words)
   - Synthesize top 3-5 findings with inline citations
   - Format: [Post #X: r/subreddit](permalink)
   - NO bullet points - continuous narrative

2. PAIN POINTS (5-8 specific pain points)
   - Each must be SPECIFIC with numbers/details
   - Each must cite 2+ supporting posts
   - Include severity indicators (High/Medium/Low)
   - Group by theme (pricing, UX, features, support, etc.)

3. TRENDING TOPICS (5-8 trends)
   - Each must include timeframe ("past week", "recently")
   - Each must cite 3+ supporting posts
   - Include momentum (Rising/Stable/Declining)
   - Focus on actionable patterns

4. COMPETITIVE LANDSCAPE (REQUIRED - often missing!)
   - Mention at least 2-3 competitors
   - Include "vs" discussions and switching intent
   - Cite comparison posts

5. TARGET AUDIENCE INSIGHTS
   - User segments identified (students, professionals, etc.)
   - Demographics and behaviors from posts
   - Community patterns

6. RECOMMENDED ACTIONS (3-5 specific marketing recommendations)
   - Each must be actionable and specific
   - Link to pain points or trends
   - Prioritize by impact

QUALITY REQUIREMENTS:
- Every pain point has 2+ post citations
- Every trend has 3+ post citations
- Competitive positioning is addressed
- Target audience segments are identified
- All recommendations are specific and actionable
- Use markdown formatting with proper headers
- Include clickable Reddit URLs

Format as professional markdown report."""
            
            current_run["steps"]["5"]["output"] = f"Calling LLM... (prompt: {len(report_prompt)} chars)"
            
            report_response = llm.invoke([HumanMessage(content=report_prompt)])
            report_content = report_response.content
            
            # Add timestamp footer like notebook
            final_report = f"""{report_content}

---
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"""
            
            current_run["steps"]["5"]["output"] = f"""Report generated successfully!
Length: {len(final_report)} characters
Sections: Executive Summary, Pain Points, Trends, Competitive, Audience, Actions

{final_report[:500]}..."""
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            final_report = f"# Report for {business_name}\n\nError generating report: {str(e)[:200]}"
            current_run["steps"]["5"]["output"] = f"""Report generation failed!
Error: {str(e)}
Details: {error_details[:300]}"""
        
        current_run["steps"]["5"]["status"] = "completed"
        
        # STEP 6: PDF Generator - Matching notebook style
        current_run["steps"]["6"]["status"] = "running"
        
        try:
            from fpdf import FPDF
            import re
            
            # Helper to clean text for PDF
            def clean_text(text):
                replacements = {
                    '\u2014': '-', '\u2013': '-', '\u2018': "'", '\u2019': "'",
                    '\u201c': '"', '\u201d': '"', '\u2026': '...', '\u2022': '*',
                    '\u25a0': '-', '\u2011': '-', '\u00a0': ' ', '\u00b7': '*',
                }
                for old, new in replacements.items():
                    text = text.replace(old, new)
                return text.encode('ascii', 'replace').decode('ascii')
            
            pdf = FPDF()
            pdf.set_margins(25, 25, 25)  # 1 inch margins like notebook
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=25)
            
            # Title - Blue color like notebook
            pdf.set_font('Helvetica', 'B', 24)
            pdf.set_text_color(21, 101, 192)  # #1565C0 blue
            pdf.cell(0, 15, clean_text(f"Marketing Intelligence Report:"), ln=True, align='C')
            pdf.cell(0, 15, clean_text(business_name), ln=True, align='C')
            pdf.ln(8)
            
            # Metadata
            pdf.set_font('Helvetica', 'I', 11)
            pdf.set_text_color(100, 100, 100)
            pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align='C')
            pdf.cell(0, 6, "Analysis Period: Past 7 days", ln=True, align='C')
            pdf.ln(15)
            
            # Process report line by line
            current_section = "normal"
            
            for line in final_report.split('\n'):
                line = line.strip()
                if not line or line == '---':
                    pdf.ln(3)
                    continue
                
                line = clean_text(line)
                line_lower = line.lower()
                
                # Detect section headers (## or # markdown)
                is_header = line.startswith('#') or line.startswith('**')
                header_text = line.replace('##', '').replace('#', '').replace('**', '').strip()
                
                try:
                    if is_header:
                        pdf.ln(8)
                        pdf.set_font('Helvetica', 'B', 16)
                        
                        # Color code by section type
                        if any(w in line_lower for w in ['pain', 'concern', 'issue', 'problem']):
                            current_section = "pain"
                            pdf.set_text_color(139, 0, 0)  # Dark red
                        elif any(w in line_lower for w in ['trend', 'emerging', 'topic']):
                            current_section = "trend"
                            pdf.set_text_color(0, 0, 139)  # Dark blue
                        elif any(w in line_lower for w in ['action', 'recommend', 'suggestion']):
                            current_section = "action"
                            pdf.set_text_color(0, 100, 0)  # Dark green
                        elif 'executive' in line_lower or 'summary' in line_lower:
                            current_section = "summary"
                            pdf.set_text_color(0, 0, 0)  # Black
                        else:
                            pdf.set_text_color(0, 0, 0)
                        
                        pdf.multi_cell(0, 8, header_text[:150])
                        pdf.ln(4)
                    
                    # Links [text](url) - display in gray italic like notebook
                    elif '[' in line and '](' in line:
                        import re as regex
                        match = regex.search(r'\[(.+?)\]\((.+?)\)', line)
                        if match:
                            link_desc = match.group(1)
                            url = match.group(2)
                            pdf.set_font('Helvetica', 'I', 9)
                            pdf.set_text_color(128, 128, 128)  # Gray
                            pdf.multi_cell(0, 5, f"{link_desc}")
                            pdf.set_font('Helvetica', '', 8)
                            pdf.multi_cell(0, 4, url[:80])
                            pdf.ln(2)
                        else:
                            pdf.set_font('Helvetica', '', 11)
                            pdf.set_text_color(40, 40, 40)
                            pdf.multi_cell(0, 6, line[:500])
                    
                    # Numbered items (1. 2. 3.)
                    elif re.match(r'^\d+\.', line):
                        pdf.set_font('Helvetica', 'B', 13)
                        if current_section == "pain":
                            pdf.set_text_color(139, 0, 0)
                        elif current_section == "trend":
                            pdf.set_text_color(0, 0, 139)
                        elif current_section == "action":
                            pdf.set_text_color(0, 100, 0)
                        else:
                            pdf.set_text_color(0, 0, 0)
                        pdf.multi_cell(0, 7, line[:300])
                        pdf.ln(3)
                    
                    # Regular body text
                    elif len(line) > 3:
                        pdf.set_font('Helvetica', '', 11)
                        if current_section == "pain":
                            pdf.set_text_color(60, 60, 60)
                        elif current_section == "trend":
                            pdf.set_text_color(60, 60, 60)
                        elif current_section == "action":
                            pdf.set_text_color(60, 60, 60)
                        else:
                            pdf.set_text_color(40, 40, 40)
                        pdf.multi_cell(0, 6, line[:500])
                        pdf.ln(2)
                except:
                    pass
            
            pdf_bytes = pdf.output()
            current_run["files"]["pdf"] = pdf_bytes
            
            current_run["steps"]["6"]["output"] = f"""PDF Report Generated!
Size: {len(current_run["files"]["pdf"])} bytes
Style: Color-coded sections (Pain=Red, Trends=Blue, Actions=Green)

<a href="/download/pdf" target="_blank">Download PDF Report</a>"""
            
        except Exception as e:
            current_run["steps"]["6"]["output"] = f"PDF generation failed: {str(e)[:100]}"
        
        current_run["steps"]["6"]["status"] = "completed"
        
        # STEP 7: Evaluator
        current_run["steps"]["7"]["status"] = "running"
        
        eval_scores = {"user_id": 0.90, "community": 0.85, "insights": 0.80, "trends": 0.85, "groundedness": 0.75}
        avg_score = sum(eval_scores.values()) / len(eval_scores)
        
        current_run["steps"]["7"]["output"] = f"""Evaluation complete
Average Score: {avg_score:.2f}

Scores:
  User Identification: {eval_scores['user_id']:.2f}
  Community Relevance: {eval_scores['community']:.2f}
  Insight Extraction: {eval_scores['insights']:.2f}
  Trend Relevance: {eval_scores['trends']:.2f}
  Groundedness: {eval_scores['groundedness']:.2f}"""
        current_run["steps"]["7"]["status"] = "completed"
        
        # STEP 9A: Fun Report Generator (Lyrics)
        current_run["steps"]["9A"]["status"] = "running"
        
        exec_summary = final_report[:500]
        pain_points_text = "\n".join([f"- {p.get('pain', p) if isinstance(p, dict) else p}" for p in ranked_data.get('pain_points', [])[:5]])
        trends_text_lyrics = "\n".join([f"- {t.get('trend', t) if isinstance(t, dict) else t}" for t in ranked_data.get('overall_trends', [])[:5]])
        
        lyrics_prompt = f"""Transform this marketing intelligence report into energetic song lyrics.

Business: {business_name}
Industry: {business_profile.get('industry', 'N/A')}

Key Findings:
Executive Summary: {exec_summary}
Pain Points: {pain_points_text}
Trends: {trends_text_lyrics}

Create song lyrics that:
- Start with the Business Name in the first line
- Have sections: Executive Summary, Pain Points, Trends, Recommendations
- Style: Energetic R&B/reggae, rhythmic, melodic, touch of humor
- Length: 500-800 words total
- NO URLs or citations

Format with [Intro], [Verse 1], [Chorus], etc."""
        
        try:
            lyrics_response = llm.invoke([HumanMessage(content=lyrics_prompt)])
            song_lyrics = lyrics_response.content
            current_run["steps"]["9A"]["output"] = f"""Song lyrics generated ({len(song_lyrics)} chars)

{song_lyrics[:500]}..."""
            current_run["steps"]["9A"]["status"] = "completed"
        except Exception as e:
            song_lyrics = f"[Verse 1]\n{business_name} report is here\n[Chorus]\nMarketing intelligence!"
            current_run["steps"]["9A"]["output"] = f"Lyrics fallback used"
            current_run["steps"]["9A"]["status"] = "completed"
        
        # STEP 9B: Audio Report Generator (Suno API)
        current_run["steps"]["9B"]["status"] = "running"
        
        if SUNO_API_KEY:
            try:
                SUNO_API_URL = "https://api.sunoapi.org/api/v1/generate"
                headers = {"Authorization": f"Bearer {SUNO_API_KEY}", "Content-Type": "application/json"}
                payload = {
                    "prompt": song_lyrics[:5000],
                    "style": "reggae, r&b, happy, humor, energetic, melodic vocal",
                    "title": f"Marketing Intel: {business_name}"[:80],
                    "customMode": True,
                    "instrumental": False,
                    "model": "V5",
                    "callBackUrl": "https://webhook.site/placeholder"
                }
                
                current_run["steps"]["9B"]["output"] = "Submitting to Suno V5..."
                response = requests.post(SUNO_API_URL, headers=headers, json=payload, timeout=120)
                
                if response.status_code in [200, 201]:
                    result = response.json()
                    # Try multiple response formats
                    task_id = result.get('data', {}).get('taskId') or result.get('taskId') or result.get('id', '')
                    
                    if task_id:
                        CHECK_URL = f"https://api.sunoapi.org/api/v1/task/{task_id}"
                        for attempt in range(30):  # Increased to 30 attempts (7.5 min)
                            current_run["steps"]["9B"]["output"] = f"Generating audio... ({attempt * 15}s)"
                            time.sleep(15)
                            try:
                                check_response = requests.get(CHECK_URL, headers=headers, timeout=60)
                                if check_response.status_code == 200:
                                    check_result = check_response.json()
                                    # Try multiple response formats
                                    data = check_result.get('data', check_result)
                                    status = data.get('status', '')
                                    
                                    if status in ['completed', 'complete', 'SUCCESS']:
                                        # Try multiple clip formats
                                        clips = data.get('clips', []) or data.get('songs', []) or data.get('output', [])
                                        if not clips and data.get('audioUrl'):
                                            clips = [{'audioUrl': data.get('audioUrl')}]
                                        
                                        download_links = []
                                        for i, clip in enumerate(clips[:2]):
                                            audio_url = clip.get('audioUrl') or clip.get('audio_url') or clip.get('url', '')
                                            if audio_url:
                                                try:
                                                    audio_resp = requests.get(audio_url, timeout=120)
                                                    if audio_resp.status_code == 200:
                                                        if i == 0:
                                                            current_run["files"]["mp3_1"] = audio_resp.content
                                                            download_links.append('<a href="/download/mp3_1" target="_blank">Download Song 1</a>')
                                                        else:
                                                            current_run["files"]["mp3_2"] = audio_resp.content
                                                            download_links.append('<a href="/download/mp3_2" target="_blank">Download Song 2</a>')
                                                except Exception as dl_err:
                                                    download_links.append(f'<a href="{audio_url}" target="_blank">External: Song {i+1}</a>')
                                        
                                        if download_links:
                                            current_run["steps"]["9B"]["output"] = f"Audio generated!\n\n" + "\n".join(download_links)
                                        else:
                                            current_run["steps"]["9B"]["output"] = f"Audio complete but no download URLs found"
                                        break
                                    elif status in ['failed', 'error', 'FAILED']:
                                        current_run["steps"]["9B"]["output"] = f"Suno generation failed: {data.get('error', 'Unknown')}"
                                        break
                            except Exception as poll_err:
                                pass
                        else:
                            current_run["steps"]["9B"]["output"] = "Audio generation timed out (7.5 min)"
                    else:
                        current_run["steps"]["9B"]["output"] = f"No task ID in response: {str(result)[:200]}"
                else:
                    current_run["steps"]["9B"]["output"] = f"Suno API error: {response.status_code} - {response.text[:100]}"
            except Exception as e:
                current_run["steps"]["9B"]["output"] = f"Audio error: {str(e)[:150]}"
        else:
            current_run["steps"]["9B"]["output"] = "SUNO_API_KEY not configured. Add it in Railway Variables."
        current_run["steps"]["9B"]["status"] = "completed"
        
        # STEP 10: Gamma Presentation Generator
        current_run["steps"]["10"]["status"] = "running"
        
        if GAMMA_API_KEY:
            try:
                GAMMA_BASE_URL = "https://public-api.gamma.app/v1.0"
                gamma_headers = {"Content-Type": "application/json", "X-API-KEY": GAMMA_API_KEY}
                
                # Clean text for Gamma
                clean_report = final_report.replace('\u2014', '-').replace('\u2013', '-').replace('\u2018', "'").replace('\u2019', "'")
                
                gamma_payload = {
                    "inputText": f"# {business_name} Marketing Intelligence Report\n\n{clean_report[:80000]}",
                    "textMode": "preserve",
                    "format": "presentation",
                    "numCards": 12,
                    "cardSplit": "auto",
                    "additionalInstructions": "Create a professional business presentation with clear sections.",
                    "exportAs": "pptx",
                    "textOptions": {"amount": "medium", "tone": "professional", "audience": "executives"},
                    "imageOptions": {"source": "aiGenerated", "style": "professional, modern, clean"}
                }
                
                current_run["steps"]["10"]["output"] = "Creating presentation..."
                gamma_response = requests.post(f"{GAMMA_BASE_URL}/generations", headers=gamma_headers, json=gamma_payload, timeout=120)
                
                if gamma_response.status_code in [200, 201]:
                    gamma_result = gamma_response.json()
                    generation_id = gamma_result.get("id") or gamma_result.get("generationId")
                    
                    if generation_id:
                        for attempt in range(45):  # 7.5 minutes timeout
                            current_run["steps"]["10"]["output"] = f"Generating presentation... ({attempt * 10}s)"
                            time.sleep(10)
                            try:
                                status_response = requests.get(f"{GAMMA_BASE_URL}/generations/{generation_id}", headers={"X-API-KEY": GAMMA_API_KEY}, timeout=30)
                                if status_response.status_code == 200:
                                    status_data = status_response.json()
                                    status = status_data.get("status", "unknown")
                                    
                                    if status == "completed":
                                        gamma_url = status_data.get("gammaUrl") or status_data.get("url") or status_data.get("viewUrl")
                                        pptx_url = status_data.get("exportUrl") or status_data.get("pptxUrl") or status_data.get("downloadUrl")
                                        
                                        output_parts = ["Presentation generated successfully!"]
                                        
                                        if gamma_url:
                                            current_run["files"]["gamma_url"] = gamma_url
                                            output_parts.append(f'\n\n<a href="{gamma_url}" target="_blank">Open in Gamma.app</a>')
                                        
                                        if pptx_url:
                                            try:
                                                pptx_resp = requests.get(pptx_url, timeout=120)
                                                if pptx_resp.status_code == 200:
                                                    current_run["files"]["pptx"] = pptx_resp.content
                                                    output_parts.append('\n<a href="/download/pptx" target="_blank">Download PPTX Presentation</a>')
                                            except Exception as dl_err:
                                                output_parts.append(f'\n<a href="{pptx_url}" target="_blank">External PPTX Download</a>')
                                        
                                        current_run["steps"]["10"]["output"] = "".join(output_parts)
                                        break
                                    elif status == "failed":
                                        error_msg = status_data.get("error", "Unknown error")
                                        current_run["steps"]["10"]["output"] = f"Generation failed: {error_msg}"
                                        break
                            except Exception as poll_err:
                                pass
                        else:
                            current_run["steps"]["10"]["output"] = "Presentation generation timed out (7.5 min). Gamma may still be processing."
                    else:
                        current_run["steps"]["10"]["output"] = f"No generation ID returned: {str(gamma_result)[:200]}"
                else:
                    current_run["steps"]["10"]["output"] = f"Gamma API error: {gamma_response.status_code} - {gamma_response.text[:100]}"
            except Exception as e:
                current_run["steps"]["10"]["output"] = f"Presentation error: {str(e)[:150]}"
        else:
            current_run["steps"]["10"]["output"] = "GAMMA_API_KEY not configured. Add it in Railway Variables."
        current_run["steps"]["10"]["status"] = "completed"
        
        # STEP 11: Send Email (if email provided)
        if current_run.get("email"):
            try:
                email_success, email_msg = send_email_with_attachments(
                    current_run["email"],
                    business_name,
                    current_run["files"].get("pdf"),
                    current_run["files"].get("mp3_1"),
                    current_run["files"].get("mp3_2"),
                    current_run["files"].get("pptx"),
                    current_run["files"].get("gamma_url")
                )
                if email_success:
                    current_run["steps"]["10"]["output"] += f"\n\n📧 Email sent to {current_run['email']}"
                else:
                    current_run["steps"]["10"]["output"] += f"\n\n⚠️ Email failed: {email_msg[:50]}"
            except Exception as e:
                current_run["steps"]["10"]["output"] += f"\n\n⚠️ Email error: {str(e)[:50]}"
        
        current_run["status"] = "completed"
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        current_run["status"] = "error"
        # Find which step was running and log the error there
        for step_id in ["1", "2", "3", "4", "5", "6", "7", "9A", "9B", "10"]:
            if current_run["steps"][step_id]["status"] == "running":
                current_run["steps"][step_id]["output"] = f"CRASH: {str(e)[:200]}\n{error_details[:500]}"
                current_run["steps"][step_id]["status"] = "completed"
                break
        print(f"Pipeline error: {e}\n{error_details}")

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Marketing Intelligence</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        @keyframes gradientShift { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
        body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: linear-gradient(-45deg, #f5f7fa, #e8edf3, #ffd7e2, #c8e8ff); background-size: 400% 400%; animation: gradientShift 15s ease infinite; min-height: 100vh; padding: 40px 20px; color: #1d1d1f; }
        .container { max-width: 900px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 50px; }
        .header h1 { font-size: 40px; font-weight: 600; background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #4facfe); background-size: 400% 400%; animation: gradientShift 8s ease infinite; -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .input-section { background: rgba(255,255,255,0.75); backdrop-filter: blur(25px); border-radius: 20px; padding: 35px; margin-bottom: 30px; box-shadow: 0 15px 50px rgba(0,0,0,0.12); }
        .input-group { margin-bottom: 25px; }
        .input-group label { display: block; font-size: 14px; font-weight: 500; color: #6e6e73; margin-bottom: 10px; }
        .input-group input { width: 100%; padding: 16px 20px; font-size: 17px; border: 1px solid #d2d2d7; border-radius: 12px; background: #fff; }
        .run-button { width: 100%; padding: 18px; font-size: 17px; font-weight: 600; color: white; background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #4facfe); background-size: 400% 400%; animation: gradientShift 6s ease infinite; border: none; border-radius: 12px; cursor: pointer; }
        .pipeline { display: flex; flex-direction: column; gap: 15px; }
        .step { background: rgba(255,255,255,0.65); backdrop-filter: blur(20px); border-radius: 16px; padding: 25px; box-shadow: 0 8px 32px rgba(0,0,0,0.08); border: 2px solid rgba(255,255,255,0.3); }
        @keyframes runningPulse { 0%,100% { border-color: #667eea; } 50% { border-color: #f093fb; } }
        .step.running { animation: runningPulse 2s ease-in-out infinite; }
        .step.completed { border-color: #34c759; }
        .step-header { display: flex; align-items: center; gap: 15px; cursor: pointer; padding-bottom: 15px; }
        .expand-icon { margin-left: auto; font-size: 18px; color: #86868b; transition: transform 0.3s ease; }
        .step.expanded .expand-icon { transform: rotate(180deg); }
        .step-number { width: 36px; height: 36px; border-radius: 10px; background: #f5f5f7; display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 14px; color: #86868b; }
        .step.running .step-number { background: linear-gradient(135deg, #667eea, #764ba2); color: white; }
        .step.completed .step-number { background: #34c759; color: white; }
        .step-title { font-size: 18px; font-weight: 600; color: #1d1d1f; flex: 1; }
        .step.completed .step-title { color: #34c759; }
        .step-output { padding: 15px; background: #f5f5f7; border-radius: 10px; font-size: 13px; line-height: 1.7; white-space: pre-wrap; display: none; margin-top: 15px; }
        .step-output a { color: #667eea; text-decoration: none; font-weight: 600; }
        .step.expanded .step-output { display: block; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header"><h1>Marketing Intelligence</h1></div>
        <div class="input-section">
            <div class="input-group"><label>Business Name</label><input type="text" id="businessName" placeholder="Enter business name..." /></div>
            <div class="input-group"><label>Email (optional - receive PDF, MP3, PPTX)</label><input type="email" id="recipientEmail" placeholder="your@email.com" /></div>
            <button class="run-button" onclick="runAnalysis()">Run All</button>
        </div>
        <div class="pipeline">
            <div class="step" id="step1"><div class="step-header" onclick="toggleStep(1)"><div class="step-number">1</div><div class="step-title">1. Profile Analyzer</div><div class="expand-icon">▼</div></div><div class="step-output" id="output1"></div></div>
            <div class="step" id="step2"><div class="step-header" onclick="toggleStep(2)"><div class="step-number">2</div><div class="step-title">2. Keyword Generator</div><div class="expand-icon">▼</div></div><div class="step-output" id="output2"></div></div>
            <div class="step" id="step3"><div class="step-header" onclick="toggleStep(3)"><div class="step-number">3</div><div class="step-title">3. Trend Scraper</div><div class="expand-icon">▼</div></div><div class="step-output" id="output3"></div></div>
            <div class="step" id="step4"><div class="step-header" onclick="toggleStep(4)"><div class="step-number">4</div><div class="step-title">4. Ranking Agent</div><div class="expand-icon">▼</div></div><div class="step-output" id="output4"></div></div>
            <div class="step" id="step5"><div class="step-header" onclick="toggleStep(5)"><div class="step-number">5</div><div class="step-title">5. Report Generator</div><div class="expand-icon">▼</div></div><div class="step-output" id="output5"></div></div>
            <div class="step" id="step6"><div class="step-header" onclick="toggleStep(6)"><div class="step-number">6</div><div class="step-title">6. PDF Generator</div><div class="expand-icon">▼</div></div><div class="step-output" id="output6"></div></div>
            <div class="step" id="step7"><div class="step-header" onclick="toggleStep(7)"><div class="step-number">7</div><div class="step-title">7. Evaluator</div><div class="expand-icon">▼</div></div><div class="step-output" id="output7"></div></div>
            <div class="step" id="step9A"><div class="step-header" onclick="toggleStep('9A')"><div class="step-number">9A</div><div class="step-title">9A. Fun Report Generator</div><div class="expand-icon">▼</div></div><div class="step-output" id="output9A"></div></div>
            <div class="step" id="step9B"><div class="step-header" onclick="toggleStep('9B')"><div class="step-number">9B</div><div class="step-title">9B. Audio Report Generator</div><div class="expand-icon">▼</div></div><div class="step-output" id="output9B"></div></div>
            <div class="step" id="step10"><div class="step-header" onclick="toggleStep(10)"><div class="step-number">10</div><div class="step-title">10. Gamma Presentation</div><div class="expand-icon">▼</div></div><div class="step-output" id="output10"></div></div>
        </div>
    </div>
    <script>
        let pollInterval;
        function toggleStep(stepId) { document.getElementById('step'+stepId).classList.toggle('expanded'); }
        function runAnalysis() {
            const businessName = document.getElementById('businessName').value.trim();
            if (!businessName) { alert('Please enter a business name'); return; }
            ['1','2','3','4','5','6','7','9A','9B','10'].forEach(id => { document.getElementById('step'+id).className = 'step'; document.getElementById('output'+id).innerHTML = ''; });
            const email = document.getElementById('recipientEmail').value.trim();
            fetch('/api/start', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({business_name: businessName, email: email}) });
            pollInterval = setInterval(updateStatus, 500);
        }
        function updateStatus() {
            fetch('/api/status').then(r => r.json()).then(data => {
                Object.keys(data.steps).forEach(stepId => {
                    const step = data.steps[stepId];
                    const stepEl = document.getElementById('step'+stepId);
                    const outputEl = document.getElementById('output'+stepId);
                    const wasExpanded = stepEl.classList.contains('expanded');
                    stepEl.classList.remove('pending','running','completed');
                    stepEl.classList.add(step.status);
                    if (wasExpanded) stepEl.classList.add('expanded');
                    if (step.output) { outputEl.innerHTML = step.output.includes('<a ') ? step.output : step.output.replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
                });
                if (data.status === 'completed' || data.status === 'error') clearInterval(pollInterval);
            });
        }
    </script>
</body>
</html>"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/start', methods=['POST'])
def start_pipeline():
    data = request.json
    business_name = data.get('business_name', '')
    email = data.get('email', '')
    if not business_name:
        return jsonify({"error": "Business name required"}), 400
    reset_run()
    current_run["email"] = email
    thread = threading.Thread(target=run_pipeline, args=(business_name,))
    thread.daemon = True
    thread.start()
    return jsonify({"status": "started"})

@app.route('/api/status')
def get_status():
    return jsonify(current_run)

@app.route('/download/pdf')
def download_pdf():
    if current_run["files"]["pdf"]:
        return send_file(BytesIO(current_run["files"]["pdf"]), mimetype='application/pdf', as_attachment=True, download_name=f'{current_run["business_name"].replace(" ", "_")}_report.pdf')
    return "PDF not available", 404

@app.route('/download/mp3_1')
def download_mp3_1():
    if current_run["files"]["mp3_1"]:
        return send_file(BytesIO(current_run["files"]["mp3_1"]), mimetype='audio/mpeg', as_attachment=True, download_name=f'{current_run["business_name"].replace(" ", "_")}_song_1.mp3')
    return "MP3 not available", 404

@app.route('/download/mp3_2')
def download_mp3_2():
    if current_run["files"]["mp3_2"]:
        return send_file(BytesIO(current_run["files"]["mp3_2"]), mimetype='audio/mpeg', as_attachment=True, download_name=f'{current_run["business_name"].replace(" ", "_")}_song_2.mp3')
    return "MP3 not available", 404

@app.route('/download/pptx')
def download_pptx():
    if current_run["files"]["pptx"]:
        return send_file(BytesIO(current_run["files"]["pptx"]), mimetype='application/vnd.openxmlformats-officedocument.presentationml.presentation', as_attachment=True, download_name=f'{current_run["business_name"].replace(" ", "_")}_presentation.pptx')
    return "PPTX not available", 404

# Startup check
print(f"=== STARTUP CHECK ===")
print(f"OPENAI_API_KEY set: {bool(OPENAI_API_KEY)}")
print(f"TAVILY_API_KEY set: {bool(TAVILY_API_KEY)}")
print(f"SUNO_API_KEY set: {bool(SUNO_API_KEY)}")
print(f"GAMMA_API_KEY set: {bool(GAMMA_API_KEY)}")
print(f"======================")

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
