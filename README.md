# 🏥 Cancer Prediction MLOps Project

A complete end-to-end MLOps pipeline for cancer survival prediction using machine learning.

[![CI/CD](https://github.com/yourusername/cancer-prediction/workflows/CI/CD%20Pipeline/badge.svg)](https://github.com/yourusername/cancer-prediction/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Project Overview

This project implements a production-ready machine learning system to predict cancer patient survival based on various clinical, demographic, and lifestyle factors. The system follows MLOps best practices including:

- ✅ Data versioning with DVC
- ✅ Experiment tracking with MLflow
- ✅ Model deployment with FastAPI
- ✅ Containerization with Docker
- ✅ CI/CD with GitHub Actions
- ✅ Monitoring with Prometheus & Grafana

## 🏗️ Project Structure

```
cancer-prediction/
├── .github/
│   └── workflows/          # CI/CD pipelines
├── configs/                # Configuration files
│   └── config.yaml        # Main configuration
├── data/
│   ├── raw/               # Raw data files (not tracked)
│   ├── processed/         # Processed data (not tracked)
│   └── external/          # External datasets
├── docker/                # Docker configuration
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── docker-compose.dev.yml
├── docs/                  # Documentation
├── logs/                  # Application logs (not tracked)
├── models/
│   ├── saved_models/      # Trained models (not tracked)
│   └── artifacts/         # Model artifacts (not tracked)
├── monitoring/            # Monitoring configurations
│   └── prometheus.yml
├── notebooks/             # Jupyter notebooks
├── src/
│   ├── api/              # FastAPI application
│   ├── data/             # Data processing
│   ├── features/         # Feature engineering
│   ├── models/           # Model training & evaluation
│   └── utils/            # Utility functions
├── tests/
│   ├── unit/             # Unit tests
│   └── integration/      # Integration tests
├── .env.example          # Environment variables template
├── .gitignore
├── .pre-commit-config.yaml
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- Docker and Docker Compose (optional)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/cancer-prediction.git
   cd cancer-prediction
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Install development dependencies** (optional)
   ```bash
   pip install -r requirements-dev.txt
   ```

5. **Set up pre-commit hooks** (optional)
   ```bash
   pre-commit install
   ```

6. **Copy environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

7. **Place your data**
   ```bash
   # Move your data.csv to data/raw/
   mv DATA/data.csv data/raw/
   ```

## 📊 Data

The dataset contains cancer patient information with the following features:

- **Demographics**: Age, Gender, Country
- **Clinical**: Cancer Stage, Tumor Size, Treatment Type
- **Risk Factors**: Smoking, Alcohol, Family History, Genetic Mutations
- **Lifestyle**: BMI, Diet, Physical Activity
- **Healthcare**: Insurance Status, Healthcare Access, Costs
- **Target**: Survival Prediction (Yes/No)

**Dataset Size**: 167,497 patients × 28 features

## 🔧 Usage

### Data Exploration

```bash
python data_exploration.py
```

### Model Training

```bash
# Train models with MLflow tracking
python src/models/train.py

# View experiments
mlflow ui
# Open http://localhost:5000
```

### Running the API

**Local Development:**
```bash
uvicorn src.api.main:app --reload
```

**With Docker:**
```bash
# Build and run all services
docker-compose -f docker/docker-compose.yml up

# Development mode (with hot reload)
docker-compose -f docker/docker-compose.dev.yml up
```

**API Documentation:** http://localhost:8000/docs

### Making Predictions

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 65,
    "Gender": "M",
    "Cancer_Stage": "Localized",
    ...
  }'
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test types
pytest tests/unit -v
pytest tests/integration -v
```

## 📈 Monitoring

Access monitoring dashboards:

- **MLflow UI**: http://localhost:5000
- **API Docs**: http://localhost:8000/docs
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

## 🔄 MLOps Pipeline

### 1. Data Management
- Data versioning with DVC
- Automated data validation
- Feature store integration

### 2. Experimentation
- Experiment tracking with MLflow
- Hyperparameter tuning with Optuna
- Model comparison and selection

### 3. CI/CD
- Automated testing on push
- Code quality checks (black, flake8, isort)
- Docker image building
- Automated deployment

### 4. Model Serving
- RESTful API with FastAPI
- Model versioning
- A/B testing support

### 5. Monitoring
- Data drift detection
- Model performance tracking
- System metrics (Prometheus/Grafana)

## 📝 Configuration

Edit `configs/config.yaml` to customize:

- Data paths and preprocessing
- Model hyperparameters
- API settings
- Monitoring configuration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Quality Standards

- Follow PEP 8 style guide
- Write unit tests for new features
- Update documentation
- Run pre-commit hooks before committing

## 📚 Documentation

For detailed documentation, see the `/docs` directory:

- [API Documentation](docs/api.md)
- [Model Documentation](docs/models.md)
- [Deployment Guide](docs/deployment.md)

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| ML Framework | Scikit-learn, XGBoost, LightGBM |
| Experiment Tracking | MLflow |
| API Framework | FastAPI |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Monitoring | Prometheus, Grafana |
| Data Versioning | DVC |
| Testing | Pytest |
| Code Quality | Black, Flake8, isort |

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Your Name** - [GitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- Dataset source: [Add source]
- Inspired by MLOps best practices

## 📞 Contact

For questions or support, please open an issue or contact [your-email@example.com]

---

**⭐ Star this repository if you find it helpful!**
