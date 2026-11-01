"""Postprocess locally generated evaluation.pt without retraining."""
import argparse
import json
from pathlib import Path
import torch
from paper_protocol import degree_statistical_parity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Only load trusted local checkpoints produced by this project.
    data = torch.load(args.evaluation, map_location='cpu')
    result = degree_statistical_parity(data['predictions'], data['labels'],
                                      data['degree'], data['masks']['test'])
    result['source_evaluation'] = str(args.evaluation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
