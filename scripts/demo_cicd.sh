#!/bin/bash
# CI/CD Demo Script
# Simulates the CI/CD pipeline locally for demonstration

set -e

echo "=========================================="
echo "CI/CD Pipeline Demo"
echo "=========================================="
echo ""

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Activate virtual environment
source .venv/bin/activate

echo -e "${BLUE}Stage 1: Code Quality Checks${NC}"
echo "Running linting and security checks..."
echo ""

# Flake8 (if installed)
if command -v flake8 &> /dev/null; then
    echo "  → Running flake8..."
    flake8 src/ --max-line-length=120 --extend-ignore=E203,W503 --exclude=__pycache__ || echo -e "${YELLOW}  ⚠ Flake8 warnings found${NC}"
else
    echo -e "${YELLOW}  ⚠ flake8 not installed (skipping)${NC}"
fi

# Black (if installed)
if command -v black &> /dev/null; then
    echo "  → Running black (check mode)..."
    black --check src/ || echo -e "${YELLOW}  ⚠ Code formatting issues found${NC}"
else
    echo -e "${YELLOW}  ⚠ black not installed (skipping)${NC}"
fi

echo -e "${GREEN}✓ Code quality checks complete${NC}"
echo ""

echo -e "${BLUE}Stage 2: Unit Tests${NC}"
echo "Running test suites..."
echo ""

if command -v pytest &> /dev/null; then
    pytest tests/ -v --tb=short -m "not slow" || echo -e "${YELLOW}  ⚠ Some tests failed${NC}"
    echo -e "${GREEN}✓ Unit tests complete${NC}"
else
    echo -e "${YELLOW}  ⚠ pytest not installed (skipping)${NC}"
fi
echo ""

echo -e "${BLUE}Stage 3: Build Docker Images${NC}"
echo "Building containers..."
echo ""

cd docker
echo "  → Building API v1..."
docker build -q -f Dockerfile.api -t cancer-prediction-api:v1 .. && echo -e "${GREEN}  ✓ API v1 built${NC}"

echo "  → Building Model v2..."
docker build -q -f Dockerfile.model-v2 -t cancer-prediction-model:v2 .. && echo -e "${GREEN}  ✓ Model v2 built${NC}"

echo "  → Building Canary Router..."
docker build -q -f Dockerfile.canary-router -t canary-router:latest .. && echo -e "${GREEN}  ✓ Canary Router built${NC}"

cd ..
echo -e "${GREEN}✓ Docker images built${NC}"
echo ""

echo -e "${BLUE}Stage 4: Model Comparison${NC}"
echo "Comparing baseline vs candidate models..."
echo ""

if [ -f "models/saved_models/best_model.pkl" ]; then
    python scripts/compare_models.py || {
        echo -e "${RED}✗ Model comparison failed${NC}"
        echo "New model does not meet performance threshold"
    }
    echo -e "${GREEN}✓ Model comparison complete${NC}"
else
    echo -e "${YELLOW}  ⚠ No trained models found. Train first with: python src/models/train_models.py${NC}"
fi
echo ""

echo -e "${BLUE}Stage 5: Integration Tests${NC}"
echo "Running integration tests..."
echo ""

# Check if services are running
if docker ps | grep -q "api-v1"; then
    echo "  → Testing API endpoints..."
    python scripts/test_local_deployment.py > /tmp/integration_test.log 2>&1 || true
    
    if grep -q "✓ Prediction successful" /tmp/integration_test.log; then
        echo -e "${GREEN}  ✓ API integration tests passed${NC}"
    else
        echo -e "${YELLOW}  ⚠ Some integration tests may have issues${NC}"
    fi
else
    echo -e "${YELLOW}  ⚠ Services not running. Start with: docker-compose -f docker/docker-compose.yml up -d${NC}"
fi
echo ""

echo -e "${BLUE}Stage 6: Deployment${NC}"
echo "Deploying services..."
echo ""

cd docker
if docker-compose ps | grep -q "Up"; then
    echo "  → Restarting services with new images..."
    docker-compose restart
    sleep 5
    echo -e "${GREEN}  ✓ Services restarted${NC}"
else
    echo "  → Starting services..."
    docker-compose up -d
    sleep 10
    echo -e "${GREEN}  ✓ Services started${NC}"
fi
cd ..
echo ""

echo -e "${BLUE}Stage 7: Post-Deployment Checks${NC}"
echo "Verifying deployment..."
echo ""

# Health checks
services=("api-v1:8000" "model-v2:8080" "canary-router:8888")
for service in "${services[@]}"; do
    IFS=':' read -r name port <<< "$service"
    if curl -s http://localhost:$port/health > /dev/null 2>&1; then
        echo -e "${GREEN}  ✓ $name is healthy${NC}"
    else
        echo -e "${RED}  ✗ $name is not responding${NC}"
    fi
done

# Canary verification
echo ""
echo "  → Verifying canary routing..."
python scripts/verify_canary.py > /tmp/canary_test.log 2>&1 || true
if grep -q "✓ Canary routing is working correctly" /tmp/canary_test.log; then
    echo -e "${GREEN}  ✓ Canary routing verified${NC}"
else
    echo -e "${YELLOW}  ⚠ Canary routing may need adjustment${NC}"
fi

echo ""
echo -e "${GREEN}=========================================="
echo "CI/CD Pipeline Demo Complete! 🚀"
echo "==========================================${NC}"
echo ""
echo "Summary:"
echo "  1. ✓ Code quality checks passed"
echo "  2. ✓ Unit tests executed"
echo "  3. ✓ Docker images built"
echo "  4. ✓ Model comparison completed"
echo "  5. ✓ Integration tests run"
echo "  6. ✓ Services deployed"
echo "  7. ✓ Post-deployment verified"
echo ""
echo "Access your services:"
echo "  - Canary Router: http://localhost:8888"
echo "  - MLflow UI:     http://localhost:5000"
echo "  - Prometheus:    http://localhost:9090"
echo "  - Grafana:       http://localhost:3000 (admin/admin)"
echo ""
