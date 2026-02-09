#!/bin/bash
# Local Setup Script for Cancer Prediction MLOps Demo

set -e  # Exit on error

echo "=========================================="
echo "Cancer Prediction MLOps - Local Setup"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Verify Python environment
echo -e "${BLUE}Step 1: Verifying Python environment...${NC}"
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Virtual environment not found. Creating one...${NC}"
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip setuptools wheel
    pip install -r requirements.txt
else
    echo -e "${GREEN}✓ Virtual environment found${NC}"
    source .venv/bin/activate
fi

# Step 2: Process data
echo ""
echo -e "${BLUE}Step 2: Processing data...${NC}"
if [ ! -f "data/processed/train_processed.csv" ]; then
    echo "Running data pipeline..."
    python src/data/pipeline.py
    echo -e "${GREEN}✓ Data processed${NC}"
else
    echo -e "${GREEN}✓ Processed data already exists${NC}"
fi

# Step 3: Train models
echo ""
echo -e "${BLUE}Step 3: Training models...${NC}"
if [ ! -f "models/saved_models/best_model.pkl" ]; then
    echo "Training models (this may take a few minutes)..."
    python src/models/train_models.py
    echo -e "${GREEN}✓ Models trained${NC}"
else
    echo -e "${GREEN}✓ Trained models already exist${NC}"
fi

# Step 4: Build Docker images
echo ""
echo -e "${BLUE}Step 4: Building Docker images...${NC}"
cd docker
echo "Building api-v1..."
docker build -f Dockerfile.api -t cancer-prediction-api:v1 ..
echo "Building model-v2..."
docker build -f Dockerfile.model-v2 -t cancer-prediction-model:v2 ..
echo "Building canary-router..."
docker build -f Dockerfile.canary-router -t canary-router:latest ..
echo -e "${GREEN}✓ Docker images built${NC}"

# Step 5: Start services
echo ""
echo -e "${BLUE}Step 5: Starting Docker services...${NC}"
docker-compose up -d
echo -e "${GREEN}✓ Services started${NC}"

# Step 6: Wait for services to be healthy
echo ""
echo -e "${BLUE}Step 6: Waiting for services to be healthy...${NC}"
sleep 10

# Check services
echo "Checking service health..."
services=("api-v1:8000" "model-v2:8080" "canary-router:8888" "mlflow:5000" "prometheus:9090" "grafana:3000")
for service in "${services[@]}"; do
    IFS=':' read -r name port <<< "$service"
    if curl -s http://localhost:$port/health > /dev/null 2>&1 || \
       curl -s http://localhost:$port > /dev/null 2>&1; then
        echo -e "${GREEN}✓ $name is running on port $port${NC}"
    else
        echo -e "${YELLOW}⚠ $name may still be starting on port $port${NC}"
    fi
done

cd ..

echo ""
echo -e "${GREEN}=========================================="
echo "Setup Complete! 🎉"
echo "==========================================${NC}"
echo ""
echo "Available Services:"
echo "  - Canary Router: http://localhost:8888"
echo "  - API v1:        http://localhost:8000"
echo "  - Model v2:      http://localhost:8080"
echo "  - MLflow:        http://localhost:5000"
echo "  - Prometheus:    http://localhost:9090"
echo "  - Grafana:       http://localhost:3000"
echo ""
echo "Quick Test Commands:"
echo "  # Test prediction via canary router"
echo "  python scripts/test_local_deployment.py"
echo ""
echo "  # Verify canary traffic split"
echo "  python scripts/verify_canary.py"
echo ""
echo "  # View logs"
echo "  docker-compose -f docker/docker-compose.yml logs -f"
echo ""
echo "  # Stop services"
echo "  docker-compose -f docker/docker-compose.yml down"
echo ""
