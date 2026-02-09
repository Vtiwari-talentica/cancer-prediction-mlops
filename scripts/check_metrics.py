"""Check Prometheus metrics after deployment"""

import requests
import argparse
from datetime import datetime, timedelta


def check_metrics(prometheus_url: str, duration: str = "5m"):
    """Query Prometheus for key metrics"""
    
    print(f"\n{'='*80}")
    print("Post-Deployment Metrics Check")
    print(f"{'='*80}")
    print(f"Prometheus: {prometheus_url}")
    print(f"Duration: {duration}")
    print(f"{'='*80}\n")
    
    queries = {
        'Error Rate': 'rate(route_errors_total[5m])',
        'Request Rate': 'rate(route_requests_total[5m])',
        'P95 Latency': 'histogram_quantile(0.95, rate(prediction_latency_seconds_bucket[5m]))',
        'Service Health': 'up'
    }
    
    results = {}
    all_healthy = True
    
    for metric_name, query in queries.items():
        try:
            response = requests.get(
                f"{prometheus_url}/api/v1/query",
                params={'query': query},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data['status'] == 'success' and data['data']['result']:
                    value = data['data']['result'][0]['value'][1]
                    results[metric_name] = float(value)
                    print(f"✅ {metric_name}: {value}")
                else:
                    results[metric_name] = None
                    print(f"⚠️  {metric_name}: No data")
            else:
                results[metric_name] = None
                all_healthy = False
                print(f"❌ {metric_name}: Query failed ({response.status_code})")
        
        except Exception as e:
            results[metric_name] = None
            all_healthy = False
            print(f"❌ {metric_name}: Error - {str(e)}")
    
    # Health checks
    print(f"\n{'='*80}")
    print("Health Assessment")
    print(f"{'='*80}")
    
    error_rate = results.get('Error Rate', 0)
    if error_rate and error_rate > 0.05:
        print(f"⚠️  High error rate: {error_rate:.4f} (threshold: 0.05)")
        all_healthy = False
    else:
        print(f"✅ Error rate acceptable: {error_rate:.4f if error_rate else 0}")
    
    latency = results.get('P95 Latency', 0)
    if latency and latency > 2.0:
        print(f"⚠️  High latency: {latency:.2f}s (threshold: 2s)")
        all_healthy = False
    else:
        print(f"✅ Latency acceptable: {latency:.2f if latency else 0}s")
    
    service_health = results.get('Service Health', 0)
    if service_health != 1:
        print(f"❌ Service unhealthy")
        all_healthy = False
    else:
        print(f"✅ All services up")
    
    print(f"{'='*80}\n")
    
    return all_healthy


def main():
    parser = argparse.ArgumentParser(description='Check Prometheus metrics')
    parser.add_argument('--prometheus', default='http://localhost:9090',
                       help='Prometheus URL')
    parser.add_argument('--duration', default='5m',
                       help='Query duration window')
    
    args = parser.parse_args()
    
    healthy = check_metrics(args.prometheus, args.duration)
    
    if healthy:
        print("✅ All metrics healthy")
        exit(0)
    else:
        print("❌ Some metrics unhealthy")
        exit(1)


if __name__ == "__main__":
    main()
