"""Verify canary traffic split ratio"""

import requests
import argparse
from collections import Counter
import time


def verify_canary_ratio(router_url: str = "http://localhost:8888", 
                       expected_ratio: str = "70:30",
                       num_requests: int = 100):
    """Send requests and verify traffic split"""
    
    expected_v1, expected_v2 = map(int, expected_ratio.split(':'))
    expected_v2_pct = expected_v2 / (expected_v1 + expected_v2)
    
    print(f"\n{'='*80}")
    print(f"Canary Traffic Split Verification")
    print(f"{'='*80}")
    print(f"Router: {router_url}")
    print(f"Expected Ratio: {expected_ratio} (API v1 / Model v2)")
    print(f"Test Requests: {num_requests}")
    print(f"{'='*80}\n")
    
    routes = []
    
    for i in range(num_requests):
        try:
            response = requests.get(f"{router_url}/health", timeout=5)
            routed_to = response.headers.get('X-Routed-To', 'unknown')
            routes.append(routed_to)
            
            if (i + 1) % 10 == 0:
                print(f"Progress: {i + 1}/{num_requests} requests sent", end='\r')
        
        except Exception as e:
            print(f"Error on request {i + 1}: {str(e)}")
            routes.append('error')
        
        time.sleep(0.05)  # Small delay to avoid overwhelming
    
    # Count routes
    route_counts = Counter(routes)
    
    api_v1_count = route_counts.get('api_v1', 0)
    model_v2_count = route_counts.get('model_v2', 0)
    
    total_successful = api_v1_count + model_v2_count
    
    if total_successful == 0:
        print("\n❌ All requests failed")
        return False
    
    actual_v2_pct = model_v2_count / total_successful
    
    # Results
    print(f"\n\n{'='*80}")
    print("Results")
    print(f"{'='*80}")
    print(f"API v1:    {api_v1_count:3d} requests ({api_v1_count/total_successful*100:.1f}%)")
    print(f"Model v2:  {model_v2_count:3d} requests ({model_v2_count/total_successful*100:.1f}%)")
    print(f"Errors:    {route_counts.get('error', 0):3d} requests")
    print(f"\nExpected v2 ratio: {expected_v2_pct*100:.0f}%")
    print(f"Actual v2 ratio:   {actual_v2_pct*100:.1f}%")
    
    # Tolerance of ±5%
    tolerance = 0.05
    within_tolerance = abs(actual_v2_pct - expected_v2_pct) <= tolerance
    
    if within_tolerance:
        print(f"\n✅ Traffic split verified - within {tolerance*100}% tolerance")
        print(f"{'='*80}\n")
        return True
    else:
        print(f"\n❌ Traffic split outside tolerance")
        print(f"{'='*80}\n")
        return False


def main():
    parser = argparse.ArgumentParser(description='Verify canary traffic split')
    parser.add_argument('--router-url', default='http://localhost:8888', 
                       help='Canary router URL')
    parser.add_argument('--expected-ratio', default='70:30',
                       help='Expected traffic ratio (v1:v2)')
    parser.add_argument('--num-requests', type=int, default=100,
                       help='Number of test requests')
    
    args = parser.parse_args()
    
    success = verify_canary_ratio(
        args.router_url,
        args.expected_ratio,
        args.num_requests
    )
    
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
