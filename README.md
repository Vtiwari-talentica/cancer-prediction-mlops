# Cancer Prediction MLOps System 🏥

Complete end-to-end MLOps implementation for cancer survival prediction with automated CI/CD, canary deployment, drift detection, and continuous training.

## 🚀 Quick Start (5 Minutes)

```bash
# Automated setup - everything in one command
chmod +x scripts/setup_local.sh
./scripts/setup_local.sh
```

**What you get:**
- ✅ Data pipeline processing 167K patients  
- ✅ 7 trained ML models with MLflow tracking
- ✅ 6 running services (API, Canary Router, MLflow, Prometheus, Grafana)
- ✅ Complete monitoring and metrics
- ✅ Ready for predictions!

## 📊 System Overview

The system implements the complete MLOps workflow:
- GitHub → CI/CD → Docker (6 services) → Canary Deployment (70/30) → Monitoring → Drift Detection → Continuous Training Loop

## 🎯 Key Features

- **Data Pipeline**: 167K patients, 83 features, 7 validation steps
- **ML Models**: 7 algorithms with MLflow tracking, auto-selection by AUC
- **API Service**: FastAPI with Prometheus metrics, single & batch predictions
- **Canary Deployment**: 70/30 traffic split with dynamic routing
- **Monitoring**: Prometheus + Grafana dashboards with custom alerts
- **Drift Detection**: Automated feature & performance drift monitoring
- **Continuous Training**: Drift-triggered retraining with MLflow integration
- **CI/CD**: GitHub Actions with 7 stages (quality, tests, deploy, verify)

## 🌐 Service URLs

| Service | URL | Purpose |
|---------|-----|---------|
| Canary Router | http://localhost:8888 | Main prediction endpoint |
| API v1 | http://localhost:8000 | Stable production model |
| Model v2 | http://localhost:8080 | Canary/experimental model |
| MLflow | http://localhost:5000 | Experiments & model registry |
| Prometheus | http://localhost:9090 | Metrics & queries |
| Grafana | http://localhost:3000 | Dashboards (admin/admin) |

## 🧪 Test the System

```bash
# Quick health check
curl http://localhost:8888/health

# Test prediction
curl -X POST http://localhost:8888/predict -H "Content-Type: application/json" -d '{"Age": 55, "Gender": 1, "Smoking": 8, ...}'

# Comprehensive tests
/Users/vikast/cancer-prediction/.venv/bin/python scripts/test_local_deployment.py

# Demo CI/CD pipeline
./scripts/demo_cicd.sh
```

## 📚 Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Get running in 5 minutes
- **[DEMO_GUIDE.md](DEMO_GUIDE.md)** - Complete demo scenarios (30 min presentation)
- **[MLOPS_GUIDE.md](MLOPS_GUIDE.md)** - Full MLOps documentation (12,000+ words)
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture & design
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Command cheatsheet

## 🚢 Deploy to GitHub (CI/CD)

```bash
# Automated setup
chmod +x scripts/setup_github.sh
./scripts/setup_github.sh

# Configure secrets at: Settings → Secrets → Actions
# Required: DOCKER_USERNAME, DOCKER_PASSWORD

# Trigger pipeline
git push
```

## 📦 What's Included

✅ **50+ Files | 7,000+ lines of code | 20,000+ words of documentation**

- 20+ Python modules (data, models, API, monitoring)
- 11 test files (unit, integration, E2E)
- 3 Dockerfiles + docker-compose.yml
- CI/CD workflow (GitHub Actions)
- 6 helper scripts
- 7 configuration files
- 5 comprehensive documentation files

**Complete MLOps Stack**: Data versioning • Experiment tracking • Model registry • API service • Monitoring • Alerting • CI/CD • Containerization • Testing

---

**Built with:** Python 3.13 • FastAPI • MLflow • Docker • Prometheus • Grafana • GitHub Actions

**Status:** ✅ Production-ready MLOps system | See [QUICKSTART.md](QUICKSTART.md) to get started
