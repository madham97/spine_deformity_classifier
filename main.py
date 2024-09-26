
import argparse
from train import main as train_main
from inference import main as inference_main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Spinal Disease Classification')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--infer', action='store_true', help='Run inference and create submission')

    args = parser.parse_args()

    if args.train:
        train_main()
    elif args.infer:
        inference_main()
    else:
        print("Please specify --train or --infer")
