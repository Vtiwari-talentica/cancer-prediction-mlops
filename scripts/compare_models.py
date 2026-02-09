"""Compare baseline and candidate models"""

import json
import argparse
from pathlib import Path
import sys

def compare_models(baseline_path: str, candidate_path: str, threshold: float = 0.02) -> dict:
    """Compare model metrics and determine if candidate is better"""
    
    # Load metrics
    with open(baseline_path, 'r') as f:
        baseline = json.load(f)
    
    with open(candidate_path, 'r') as f:
        candidate = json.load(f)
    
    # Calculate deltas
    metrics = ['accuracy', 'auc', 'f1', 'precision', 'recall']
    comparison = {
        'baseline': {},
        'candidate': {},
        'delta': {},
        'approved': True,
        'reason': []
    }
    
    for metric in metrics:
        base_val = baseline.get(metric, 0)
        cand_val = candidate.get(metric, 0)
        delta = cand_val - base_val
        
        comparison['baseline'][metric] = round(base_val, 4)
        comparison['candidate'][metric] = round(cand_val, 4)
        comparison['delta'][metric] = round(delta, 4)
        
        # Check if candidate is worse than threshold
        if delta < -threshold:
            comparison['approved'] = False
            comparison['reason'].append(f"{metric} degraded by {abs(delta):.4f}")
    
    # Overall decision
    if comparison['approved']:
        comparison['reason'] = ['Candidate model meets performance criteria']
    
    return comparison


def main():
    parser = argparse.ArgumentParser(description='Compare baseline and candidate models')
    parser.add_argument('--baseline', required=True, help='Path to baseline metrics JSON')
    parser.add_argument('--candidate', required=True, help='Path to candidate metrics JSON')
    parser.add_argument('--threshold', type=float, default=0.02, 
                       help='Maximum allowed performance degradation')
    
    args = parser.parse_args()
    
    comparison = compare_models(args.baseline, args.candidate, args.threshold)
    
    # Save results
    with open('comparison_results.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    
    # Print summary
    print("\nModel Comparison Results")
    print("="*80)
    print(f"Status: {'✅ APPROVED' if comparison['approved'] else '❌ REJECTED'}")
    print(f"\nBaseline Model: {Path(args.baseline).stem}")
    print(f"Candidate Model: {Path(args.candidate).stem}")
    
    print("\n Metrics Comparison:")
    print(f"{'Metric':<12} {'Baseline':<10} {'Candidate':<10} {'Delta':<10}")
    print("-"*80)
    
    for metric in comparison['baseline'].keys():
        base = comparison['baseline'][metric]
        cand = comparison['candidate'][metric]
        delta = comparison['delta'][metric]
        delta_str = f"{delta:+.4f}"
        print(f"{metric:<12} {base:<10.4f} {cand:<10.4f} {delta_str:<10}")
    
    print("\nReason:", ', '.join(comparison['reason']))
    print("="*80)
    
    # Exit code for CI/CD
    sys.exit(0 if comparison['approved'] else 1)


if __name__ == "__main__":
    main()
