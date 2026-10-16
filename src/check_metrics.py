"""Próg jakości: kończy się błędem, gdy model jest słabszy niż wymagany (używane w Jenkinsie)."""
import argparse
import json
import sys


def check(metrics_path: str, metric: str, minimum: float) -> bool:
    with open(metrics_path) as f:
        value = json.load(f)[metric]
    passed = value >= minimum
    print(f"{metric} = {value:.4f} (próg {minimum:.4f}) -> {'OK' if passed else 'ZA SŁABY MODEL'}")
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="reports/metrics.json")
    parser.add_argument("--metric", default="f1_weighted")
    parser.add_argument("--min", type=float, required=True)
    args = parser.parse_args()
    sys.exit(0 if check(args.metrics, args.metric, args.min) else 1)
