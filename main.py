import torch
import argparse
from trainer import Trainer, Evaluater

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="model/mengzi-t5-base")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", default="name")
    parser.add_argument("--train_bz", type=int, default=32)
    parser.add_argument("--eval_bz", type=int, default=256)
    parser.add_argument("--accu", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--eval_step", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--pred_path", type=str, default="")
    args = parser.parse_args()

    trainer = Trainer(args)

    trainer.train()
    
    preds = {}
    preds["test"] = trainer.test()
    if args.task == "component":
        preds["train"] = Evaluater(args, trainer.model, trainer.tokenizer, "train").generate()
        preds["valid"] = Evaluater(args, trainer.model, trainer.tokenizer, "valid").generate()
        torch.save(preds, f"{trainer.out_dir}/preds.pt")

if __name__ == "__main__":
    main()