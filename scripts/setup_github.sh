#!/bin/bash
# GitHub Setup Script for CI/CD Demo

set -e

echo "=========================================="
echo "GitHub CI/CD Setup"
echo "=========================================="
echo ""

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check if GitHub CLI is installed
if ! command -v gh &> /dev/null; then
    echo -e "${RED}Error: GitHub CLI (gh) is not installed${NC}"
    echo "Install it from: https://cli.github.com/"
    echo ""
    echo "Or use Homebrew: brew install gh"
    exit 1
fi

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo -e "${BLUE}Initializing Git repository...${NC}"
    git init
    echo -e "${GREEN}✓ Git initialized${NC}"
else
    echo -e "${GREEN}✓ Git already initialized${NC}"
fi

# Create .gitignore if not exists
if [ ! -f ".gitignore" ]; then
    echo -e "${BLUE}Creating .gitignore...${NC}"
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg-info/
dist/
build/

# Virtual environments
.venv/
venv/
ENV/

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
logs/*.log
logs/drift_reports/*.json

# MLflow artifacts
mlruns/
mlartifacts/

# Test and coverage
.pytest_cache/
.coverage
htmlcov/

# Data (too large for Git)
data/raw/data.csv
data/processed/*.csv

# Models (use MLflow for versioning)
models/saved_models/*.pkl
models/artifacts/*.pkl

# Comparison results
comparison_results.json
EOF
    echo -e "${GREEN}✓ .gitignore created${NC}"
fi

# Check Git authentication
echo ""
echo -e "${BLUE}Checking GitHub authentication...${NC}"
if gh auth status &> /dev/null; then
    echo -e "${GREEN}✓ Already authenticated with GitHub${NC}"
else
    echo -e "${YELLOW}Not authenticated. Starting GitHub login...${NC}"
    gh auth login
fi

# Get repository details
echo ""
echo -e "${BLUE}Repository Setup${NC}"
read -p "Enter repository name (default: cancer-prediction): " REPO_NAME
REPO_NAME=${REPO_NAME:-cancer-prediction}

read -p "Make repository public? (y/N): " IS_PUBLIC
if [[ "$IS_PUBLIC" =~ ^[Yy]$ ]]; then
    VISIBILITY="--public"
else
    VISIBILITY="--private"
fi

# Check if remote already exists
if git remote get-url origin &> /dev/null; then
    echo -e "${YELLOW}⚠ Remote 'origin' already exists${NC}"
    read -p "Remove existing remote and continue? (y/N): " REMOVE_REMOTE
    if [[ "$REMOVE_REMOTE" =~ ^[Yy]$ ]]; then
        git remote remove origin
        echo -e "${GREEN}✓ Removed existing remote${NC}"
    else
        echo "Keeping existing remote. Skipping repository creation."
        SKIP_CREATE=true
    fi
fi

# Create GitHub repository
if [ "$SKIP_CREATE" != "true" ]; then
    echo ""
    echo -e "${BLUE}Creating GitHub repository...${NC}"
    
    if gh repo create "$REPO_NAME" $VISIBILITY --source=. --remote=origin; then
        echo -e "${GREEN}✓ Repository created: $REPO_NAME${NC}"
    else
        echo -e "${RED}✗ Failed to create repository${NC}"
        echo "You may need to create it manually at https://github.com/new"
        exit 1
    fi
fi

# Configure Git user if not set
if [ -z "$(git config user.name)" ]; then
    echo ""
    read -p "Enter your Git name: " GIT_NAME
    git config user.name "$GIT_NAME"
fi

if [ -z "$(git config user.email)" ]; then
    read -p "Enter your Git email: " GIT_EMAIL
    git config user.email "$GIT_EMAIL"
fi

# Stage all files
echo ""
echo -e "${BLUE}Staging files...${NC}"
git add .
echo -e "${GREEN}✓ Files staged${NC}"

# Commit
echo ""
echo -e "${BLUE}Creating initial commit...${NC}"
if git diff --cached --quiet; then
    echo -e "${YELLOW}⚠ No changes to commit${NC}"
else
    git commit -m "Initial commit: Complete MLOps implementation

Features:
- Data pipeline with 7 validation steps
- 7 ML models with MLflow tracking
- FastAPI service with Prometheus metrics
- Canary deployment (70/30 split)
- Drift detection system
- Continuous training pipeline
- Docker Compose with 6 services
- GitHub Actions CI/CD
- Comprehensive documentation
"
    echo -e "${GREEN}✓ Initial commit created${NC}"
fi

# Push to GitHub
echo ""
echo -e "${BLUE}Pushing to GitHub...${NC}"
if git push -u origin main 2>/dev/null || git push -u origin master 2>/dev/null; then
    echo -e "${GREEN}✓ Pushed to GitHub${NC}"
else
    # Try to create main branch and push
    git branch -M main
    git push -u origin main
    echo -e "${GREEN}✓ Pushed to GitHub (main branch)${NC}"
fi

# Get repository URL
REPO_URL=$(gh repo view --json url -q .url)

echo ""
echo -e "${GREEN}=========================================="
echo "GitHub Setup Complete! 🎉"
echo "==========================================${NC}"
echo ""
echo "Repository URL: $REPO_URL"
echo ""
echo -e "${YELLOW}Important: Configure GitHub Secrets for CI/CD${NC}"
echo ""
echo "Required secrets (Settings → Secrets → Actions):"
echo "  1. DOCKER_USERNAME - Your Docker Hub username"
echo "  2. DOCKER_PASSWORD - Your Docker Hub access token"
echo ""
echo "Optional secrets:"
echo "  3. MLFLOW_TRACKING_URI - Remote MLflow server URL"
echo "  4. SLACK_WEBHOOK - For CI/CD notifications"
echo ""
echo "To set secrets via CLI:"
echo "  gh secret set DOCKER_USERNAME"
echo "  gh secret set DOCKER_PASSWORD"
echo ""
echo "Next steps:"
echo "  1. Set up secrets: $REPO_URL/settings/secrets/actions"
echo "  2. Make a code change"
echo "  3. Push to trigger CI/CD: git push"
echo "  4. Watch Actions: $REPO_URL/actions"
echo ""
