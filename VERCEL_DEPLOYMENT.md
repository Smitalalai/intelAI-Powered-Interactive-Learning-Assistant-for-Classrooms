# EduAI Pro - Vercel Deployment Guide

## 🚀 Deploying to Vercel

### Prerequisites
1. [Vercel Account](https://vercel.com)
2. [Git Repository](https://github.com) (optional but recommended)
3. [PostgreSQL Database](https://neon.tech) (free tier available)

### Step 1: Prepare Your Database

**Option A: Neon (Recommended - Free Tier)**
1. Go to [Neon.tech](https://neon.tech)
2. Create a free account
3. Create a new database
4. Copy the connection string (starts with `postgresql://`)

**Option B: Railway, Supabase, or other PostgreSQL providers**

### Step 2: Deploy to Vercel

**Method 1: Direct from folder**
1. Install Vercel CLI: `npm i -g vercel`
2. In your project folder: `vercel`
3. Follow the prompts

**Method 2: From Git Repository**
1. Push your code to GitHub
2. Connect your GitHub to Vercel
3. Import your repository
4. Configure environment variables

### Step 3: Environment Variables

Set these in Vercel Dashboard → Project → Settings → Environment Variables:

```
DATABASE_URL=postgresql://username:password@host:port/database
SECRET_KEY=your-secret-key-here
FLASK_ENV=production
AI_DEMO_MODE=true
OPENAI_API_KEY=your-openai-key (optional)
```

### Step 4: File Structure for Vercel

```
quiz_master_23f1002833/
├── vercel.json              # Vercel configuration
├── app_vercel.py           # Vercel-optimized app
├── requirements.txt        # Lightweight dependencies
├── config_vercel.py       # Production config
├── models.py              # Database models
├── ai_service.py          # AI service (demo mode)
├── templates/             # HTML templates
└── static/               # CSS, JS files
```

### Step 5: Important Notes

⚠️ **Limitations on Vercel:**
- **No OpenVINO**: Heavy ML libraries don't work on serverless
- **No File Storage**: Use cloud storage for uploads
- **No SQLite**: Must use PostgreSQL or similar
- **Function Timeout**: 30 seconds max execution time
- **Cold Starts**: First request may be slower

✅ **What Works:**
- Core Flask application
- Basic AI features (demo mode)
- User authentication
- Quiz functionality
- Database operations
- Responsive UI

### Step 6: Alternative Deployment Options

If you need full OpenVINO features, consider:

1. **Railway** - Supports heavier dependencies
2. **Render** - Good for Python apps
3. **Google Cloud Run** - Containerized deployment
4. **AWS ECS/Lambda** - With custom runtime
5. **DigitalOcean App Platform** - Simple deployment

### Step 7: Testing Deployment

After deployment, test these endpoints:
- `/` - Landing page
- `/login` - Authentication
- `/dashboard` - Main dashboard
- `/health` - Health check
- `/api/create_flashcards` - API functionality

### Troubleshooting

**Common Issues:**
1. **Database Connection**: Ensure DATABASE_URL is correct
2. **Import Errors**: Check requirements.txt
3. **Function Timeout**: Optimize heavy operations
4. **Static Files**: Ensure static files are served correctly

**Logs:**
- Check Vercel Function logs in dashboard
- Use `print()` statements for debugging

### Sample Environment Variables

```bash
# Required
DATABASE_URL=postgresql://user:pass@host:5432/eduaipro
SECRET_KEY=super-secret-key-change-this
FLASK_ENV=production

# Optional
AI_DEMO_MODE=true
OPENAI_API_KEY=sk-...
```

### Post-Deployment

1. **Test all features**
2. **Set up monitoring**
3. **Configure custom domain** (optional)
4. **Set up database backups**
5. **Monitor function usage and costs**

---

## 🎯 Quick Deploy Commands

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy from project folder
cd quiz_master_23f1002833
vercel

# Set environment variables
vercel env add DATABASE_URL
vercel env add SECRET_KEY
vercel env add FLASK_ENV

# Redeploy with new settings
vercel --prod
```

Your EduAI Pro application will be live at `https://your-project.vercel.app`!
