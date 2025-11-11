# 🚀 GitHub + Railway Auto-Deploy Setup

## ✅ What I've Done:
- ✅ Created complete Flask web app (`app.py`)
- ✅ Added all deployment files (Procfile, railway.toml, requirements.txt)
- ✅ Included your notebook (`marketing_intel_lean2.ipynb`)
- ✅ Committed everything to Git locally
- ✅ Opened GitHub new repo page for you

---

## 📋 STEP 1: Create GitHub Repository (30 seconds)

**I just opened GitHub in your browser** (https://github.com/new)

Fill in:
- **Repository name:** `marketing-intelligence-app`
- **Description:** Marketing Intelligence Web App with Reddit Analysis
- **Visibility:** ✅ Public (or Private if you prefer)
- ❌ **DO NOT** check "Add README" (we already have files)
- ❌ **DO NOT** check "Add .gitignore"
- ❌ **DO NOT** check "Choose a license"

Click **"Create repository"**

---

## 📋 STEP 2: Push Code to GitHub (10 seconds)

After creating the repo, GitHub shows commands. **IGNORE THOSE!**

Instead, run this in your Terminal (I'll do it for you):

```bash
# The repo is already set up, just need to push!
git push -u origin main
```

You'll be prompted for:
- **Username:** ojavadli
- **Password:** Use a Personal Access Token (NOT your password)

**Get token here:** https://github.com/settings/tokens/new
- Note: "Railway deployment"
- Expiration: 90 days
- Scope: ✅ Check "repo" (full control)
- Click "Generate token"
- Copy the token and paste as password

---

## 📋 STEP 3: Connect Railway to GitHub (1 minute)

**In your Railway dashboard** (railway.com/dashboard):

1. Click **"+ New"** → **"Deploy from GitHub repo"**

2. Click **"Configure GitHub App"**

3. Select **"ojavadli"** account

4. Choose: **"Only select repositories"**

5. Select: **"marketing-intelligence-app"**

6. Click **"Install & Authorize"**

7. Back in Railway, select **"marketing-intelligence-app"** from the list

8. Railway auto-detects configuration ✅

9. Click **"Variables"** (left sidebar) and add:
   ```
   OPENAI_API_KEY = [your OpenAI API key from notebook Cell 2]
   
   TAVILY_API_KEY = [your Tavily API key from notebook Cell 2]
   ```

10. Railway deploys automatically! ✅

11. Go to **"Settings"** → **"Networking"** → **"Generate Domain"**

12. Get your live URL! 🎉

---

## 🔄 AUTO-DEPLOY MAGIC

Now **every time you push to GitHub**, Railway automatically redeploys!

```bash
# Make changes in notebook or app.py
git add -A
git commit -m "Updated analysis"
git push

# Railway automatically redeploys! ✅
```

---

## ✅ BENEFITS OF THIS APPROACH:

✅ **Auto-deploy on every push** - No manual deploys!
✅ **Version control** - All changes tracked in Git
✅ **Easy rollback** - Revert to any previous version
✅ **Team collaboration** - Others can contribute
✅ **Professional workflow** - Industry standard

---

## 🎯 QUICK SUMMARY:

1. **Create GitHub repo** (I opened the page) ← DO THIS NOW
2. **Push code** (I'll help) ← I'LL DO THIS
3. **Connect Railway** (in your Railway tab) ← THEN DO THIS

**Total time: 2 minutes** ⚡

---

Let me know when you've created the GitHub repo and I'll push the code immediately!

