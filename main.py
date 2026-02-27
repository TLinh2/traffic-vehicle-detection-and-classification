# main.py
from src.core.trainer import Trainer
import argparse

# python main.py --config configs/exp_v1_yolo11m.yaml --predict_val --analyze_errors

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--predict_val", action="store_true")
    parser.add_argument("--analyze_errors", action="store_true")
    args = parser.parse_args()

    trainer = Trainer(args.config)
    trainer.train()

    if args.predict_val:
        trainer.predict_val()

    if args.analyze_errors:
        trainer.run_error_analysis()

    
