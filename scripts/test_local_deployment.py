"""
Test Local Deployment
Sends test predictions to verify the system is working
"""
import requests
import json
import time
from typing import Dict, Any

# Test data - sample patient
SAMPLE_PATIENT = {
    "Age": 45,
    "Gender": 1,
    "Air Pollution": 3,
    "Alcohol use": 4,
    "Dust Allergy": 5,
    "OccuPational Hazards": 6,
    "Genetic Risk": 5,
    "chronic Lung Disease": 4,
    "Balanced Diet": 3,
    "Obesity": 4,
    "Smoking": 7,
    "Passive Smoker": 6,
    "Chest Pain": 4,
    "Coughing of Blood": 3,
    "Fatigue": 5,
    "Weight Loss": 4,
    "Shortness of Breath": 3,
    "Wheezing": 4,
    "Swallowing Difficulty": 3,
    "Clubbing of Finger Nails": 2,
    "Frequent Cold": 3,
    "Dry Cough": 4,
    "Snoring": 5,
    "Tumor_Size_mm": 25.5,
    "Healthcare_Access_Quality": 6,
    "Pollution_Exposure_Score": 4,
    "Treatment_Cost_USD": 15000
}

def test_health_endpoints():
    """Test health endpoints of all services"""
    print("=" * 60)
    print("Testing Health Endpoints")
    print("=" * 60)
    
    services = {
        "Canary Router": "http://localhost:8888/health",
        "API v1": "http://localhost:8000/health",
        "Model v2": "http://localhost:8080/health",
        "MLflow": "http://localhost:5000",
        "Prometheus": "http://localhost:9090/-/healthy",
    }
    
    for name, url in services.items():
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✓ {name:20} - OK")
            else:
                print(f"✗ {name:20} - Status {response.status_code}")
        except Exception as e:
            print(f"✗ {name:20} - Error: {str(e)}")
    print()

def test_single_prediction():
    """Test single prediction via canary router"""
    print("=" * 60)
    print("Testing Single Prediction (via Canary Router)")
    print("=" * 60)
    
    url = "http://localhost:8888/predict"
    
    try:
        start_time = time.time()
        response = requests.post(url, json=SAMPLE_PATIENT, timeout=10)
        latency = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            result = response.json()
            routed_to = response.headers.get('X-Routed-To', 'Unknown')
            
            print(f"✓ Prediction successful")
            print(f"  Routed to: {routed_to}")
            print(f"  Latency: {latency:.2f}ms")
            print(f"  Prediction: {result['prediction']}")
            print(f"  Probability: {result['probability']:.4f}")
            print(f"  Risk Level: {result['risk_level']}")
        else:
            print(f"✗ Prediction failed - Status {response.status_code}")
            print(f"  Response: {response.text}")
    except Exception as e:
        print(f"✗ Error: {str(e)}")
    print()

def test_batch_prediction():
    """Test batch prediction"""
    print("=" * 60)
    print("Testing Batch Prediction")
    print("=" * 60)
    
    url = "http://localhost:8888/predict/batch"
    
    # Create 3 variations of the sample patient
    patients = [
        {**SAMPLE_PATIENT, "Age": 35, "Smoking": 2},
        {**SAMPLE_PATIENT, "Age": 55, "Smoking": 8},
        {**SAMPLE_PATIENT, "Age": 65, "Smoking": 9, "Tumor_Size_mm": 45.0}
    ]
    
    try:
        start_time = time.time()
        response = requests.post(url, json={"patients": patients}, timeout=15)
        latency = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            results = response.json()
            routed_to = response.headers.get('X-Routed-To', 'Unknown')
            
            print(f"✓ Batch prediction successful")
            print(f"  Routed to: {routed_to}")
            print(f"  Latency: {latency:.2f}ms")
            print(f"  Predictions: {len(results['predictions'])}")
            for i, pred in enumerate(results['predictions'], 1):
                print(f"    Patient {i}: {pred['prediction']} (prob: {pred['probability']:.4f})")
        else:
            print(f"✗ Batch prediction failed - Status {response.status_code}")
            print(f"  Response: {response.text}")
    except Exception as e:
        print(f"✗ Error: {str(e)}")
    print()

def test_model_info():
    """Get model information"""
    print("=" * 60)
    print("Model Information")
    print("=" * 60)
    
    services = {
        "API v1": "http://localhost:8000/model/info",
        "Model v2": "http://localhost:8080/model/info"
    }
    
    for name, url in services.items():
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                info = response.json()
                print(f"\n{name}:")
                print(f"  Model Type: {info.get('model_type', 'N/A')}")
                print(f"  Version: {info.get('version', 'N/A')}")
                print(f"  Features: {info.get('n_features', 'N/A')}")
            else:
                print(f"\n{name}: Status {response.status_code}")
        except Exception as e:
            print(f"\n{name}: Error - {str(e)}")
    print()

def test_canary_distribution():
    """Test canary traffic distribution"""
    print("=" * 60)
    print("Testing Canary Distribution (20 requests)")
    print("=" * 60)
    
    url = "http://localhost:8888/predict"
    routes = {"api-v1": 0, "model-v2": 0}
    
    for i in range(20):
        try:
            response = requests.post(url, json=SAMPLE_PATIENT, timeout=10)
            if response.status_code == 200:
                routed_to = response.headers.get('X-Routed-To', 'Unknown')
                if routed_to in routes:
                    routes[routed_to] += 1
        except Exception as e:
            print(f"Request {i+1} failed: {str(e)}")
    
    total = sum(routes.values())
    if total > 0:
        print(f"✓ Completed {total} requests")
        print(f"  API v1:    {routes['api-v1']:2d} requests ({routes['api-v1']/total*100:.1f}%)")
        print(f"  Model v2:  {routes['model-v2']:2d} requests ({routes['model-v2']/total*100:.1f}%)")
        print(f"  Expected: ~70% / ~30%")
    else:
        print("✗ No successful requests")
    print()

def test_prometheus_metrics():
    """Check if Prometheus is scraping metrics"""
    print("=" * 60)
    print("Checking Prometheus Metrics")
    print("=" * 60)
    
    try:
        # Query for total predictions
        query = 'sum(predictions_total)'
        response = requests.get(
            f'http://localhost:9090/api/v1/query',
            params={'query': query},
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'success' and data['data']['result']:
                value = data['data']['result'][0]['value'][1]
                print(f"✓ Prometheus is collecting metrics")
                print(f"  Total predictions: {value}")
            else:
                print("⚠ Metrics not yet available (may need more time)")
        else:
            print(f"✗ Prometheus query failed - Status {response.status_code}")
    except Exception as e:
        print(f"✗ Error: {str(e)}")
    print()

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Cancer Prediction MLOps - Local Deployment Test")
    print("=" * 60 + "\n")
    
    # Run all tests
    test_health_endpoints()
    test_single_prediction()
    test_batch_prediction()
    test_model_info()
    test_canary_distribution()
    test_prometheus_metrics()
    
    print("=" * 60)
    print("Testing Complete!")
    print("=" * 60)
    print("\nNext Steps:")
    print("  - View MLflow experiments: http://localhost:5000")
    print("  - View Prometheus metrics: http://localhost:9090")
    print("  - View Grafana dashboards: http://localhost:3000")
    print("  - Check canary routing: python scripts/verify_canary.py")
    print()
