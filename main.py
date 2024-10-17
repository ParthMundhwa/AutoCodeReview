import json
import os
from common_functions.common_utils import load_data, create_model, train_model, evaluate_model, save_model
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Code Review Project")
    parser.add_argument('--config', type=str, default='common_configs/common_configs.json', help='Path to config file')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)

    if __name__ == '__main__':
        main()