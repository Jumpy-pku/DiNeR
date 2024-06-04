import torch
import numpy as np
import pandas as pd
import random
import re
import os
import json
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from dataset import create_dataset, create_dataloader
from utils import to_device, NameParser


class Evaluater:
    def __init__(self, args, model, tokenizer, split):
        self.args = args
        self.split = split
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device("cuda:0")
        self.data = create_dataset(args, split, tokenizer, self.args.pred_path)
        if self.split == "train":
            self.loader = create_dataloader(args, self.data, "test")
        else:
            self.loader = create_dataloader(args, self.data, split)
        self.glossary = {}
        for glo in json.load(open("data/glossary.json")).values():
            for k, v in glo.items():
                self.glossary[k] = v

        self.clusters = {}
        for k, v in self.glossary.items():
            if v in self.clusters:
                self.clusters[v].append(k)
            else:
                self.clusters[v] = [k]

        self.parser = NameParser()

    def generate(self):
        print("generating")
        preds = []
        self.model.eval()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            with torch.no_grad():
                for inputs in tqdm(self.loader):
                    inputs = to_device(inputs, self.device)
                    pred = self.model.generate(
                        input_ids=inputs["input_ids"], 
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=16,
                        early_stopping=True,
                        num_beams=1
                    ).detach().cpu().numpy()
                    preds += [p.tolist() for p in pred]

        preds = [re.sub(r"<.*?>", "", "".join(p.split())) for p in self.tokenizer.batch_decode(preds, skip_special_tokens=True)]
        return preds
       
    def evaluate(self):
        print("evaluating...")

        preds, all_loss = [], []
        self.model.eval()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            with torch.no_grad():
                for inputs in tqdm(self.loader):
                    inputs = to_device(inputs, self.device)
                    loss = self.model(**inputs).loss
                    all_loss.append(loss.item())
                    pred = self.model.generate(
                        input_ids=inputs["input_ids"], 
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=16,
                        early_stopping=True,
                        num_beams=1
                    ).detach().cpu().numpy()
                    preds += [p.tolist() for p in pred]
        
        preds = [re.sub(r"<.*?>", "", "".join(p.split())) for p in self.tokenizer.batch_decode(preds, skip_special_tokens=True)]
        labels = self.data.labels

        # metrics
        f1s, precs, recs, ems = [], [], [], []

        if self.args.task == "component":
            for i in range(len(preds)//3):
                pred_id = preds[3*i].split(",") + preds[3*i+1].split(",") + preds[3*i+2].split(",")
                pred_id = {self.glossary[item] if item in self.glossary else item for item in pred_id}
                pred_id.discard("")
                label_id = labels[3*i].split(",") + labels[3*i+1].split(",") + labels[3*i+2].split(",")
                label_id = {self.glossary[item] if item in self.glossary else item for item in label_id}
                label_id.discard("")
                tp = len(pred_id.intersection(label_id))
                if tp == 0:
                    prec, rec, f1 = 0, 0, 0
                else:
                    prec, rec = tp / len(pred_id), tp / len(label_id)
                    f1 = 2 * prec * rec / (prec + rec)
                f1s.append(f1)
                precs.append(prec)
                recs.append(rec)
                ems.append((preds[3*i]==labels[3*i]) and (preds[3*i+1]==labels[3*i+1]) and ((preds[3*i+2]==labels[3*i+2])))
        else:
            for pred, label in zip(preds, labels):
                
                label_id = set(self.parser(label)[2])
                pred_id = set(self.parser(pred)[2])

                if pred_id == []:
                    tp = 0
                    for id in label_id:
                        tp += any([item in pred for item in self.clusters[id]])
                    if tp == 0:
                        prec, rec, f1 = 0, 0, 0
                    else:
                        rec = tp / len(label_id)
                        prec = rec * len(label) / len(pred)
                        f1 = 2 * prec * rec / (prec + rec)
                else:
                    tp = len(pred_id.intersection(label_id))
                    if tp == 0:
                        prec, rec, f1 = 0, 0, 0
                    else:
                        prec, rec = tp / len(pred_id), tp / len(label_id)
                        f1 = 2 * prec * rec / (prec + rec)
                precs.append(prec)
                recs.append(rec)
                f1s.append(f1)
                ems.append(pred == label)

        metrics = {"f1": np.mean(f1s), 
                "prec": np.mean(precs), "rec": np.mean(recs), "EM": np.mean(ems),
                "acc": np.mean([f1 == 1 for f1 in f1s])}
        if self.split == "valid":
            metrics["loss"] = np.mean(all_loss)
        metrics = {f"{self.split}_{k}": v for k, v in metrics.items()}

        return preds, metrics

class Trainer:
    def __init__(self, args):
        self.args = args

        # random seed
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)

        torch.set_float32_matmul_precision('medium')

        # gpu
        self.device = torch.device("cuda:0")

        # model
        print("building model...")
        self.model = T5ForConditionalGeneration.from_pretrained(self.args.model_path)
        self.tokenizer = T5Tokenizer.from_pretrained(self.args.model_path)
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["<title>"]})
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.model.to(self.device)

        run_name = f"{self.args.task}_{self.args.seed}_{self.args.model_path.split('/')[-1]}"

        self.out_dir = f"{self.args.output_dir}/{run_name}"
        if not os.path.exists(self.args.output_dir):
            os.mkdir(self.args.output_dir)
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)
        
        # training data
        print("loading data...")
        self.data = create_dataset(self.args, "train", self.tokenizer, self.args.pred_path)
        self.loader = create_dataloader(self.args, self.data, "train")


    def configure_optimizer(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.lr)
        total_steps = len(self.loader) * self.args.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps // (2*self.args.epochs),
            num_training_steps=total_steps
        )
        return optimizer, scheduler
    

    def train(self):

        scaler = torch.cuda.amp.GradScaler()

        optimizer, scheduler = self.configure_optimizer()

        max_step = self.args.epochs * len(self.loader)
        valid_evaluater = Evaluater(self.args, self.model, self.tokenizer, "valid")

        early_stopping_count = 0
        loss_hist = []
        out_path = os.path.join(self.out_dir, "model.pt")

        metric_key = "valid_loss"
        if metric_key == "valid_loss":
            best_score = 1e5
        else:
            best_score = 0

        pbar = tqdm(range(1, max_step+1))
        for step in pbar:
            self.model.train()
            if step % len(self.loader) == 1:
                iterator = iter(self.loader)
            
            inputs = next(iterator)
            inputs = to_device(inputs, self.device)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                loss = self.model(**inputs).loss
                pbar.set_postfix_str(f"loss={loss.item()}")
                loss = loss / self.args.accu
            
            loss_hist.append(loss.item())
            scaler.scale(loss).backward()

            if step % self.args.accu == 0 or step == max_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            if step % self.args.eval_step == 0 or step == max_step:
                preds, metrics = valid_evaluater.evaluate()
                metrics["train_loss"] = np.mean(loss_hist)
                loss_hist = []
                print(metrics)

                score = metrics[metric_key]
                if (metric_key != "valid_loss" and score > best_score) or \
                    (metric_key == "valid_loss" and score < best_score):
                    best_score = score
                    torch.save(self.model.state_dict(), out_path)
                    early_stopping_count = 0
                else:
                    early_stopping_count += 1
                    if early_stopping_count >= self.args.patience:
                        print("out of patience, stop training")
                        print(f"best score: {best_score}")
                        break

        print("load best model...")
        self.model.load_state_dict(torch.load(out_path))

    def test(self):
        print(f"Testing...")

        test_evaluater = Evaluater(self.args, self.model, self.tokenizer, "test")
        preds, metrics = test_evaluater.evaluate()
        print(metrics)
        test_df = pd.DataFrame({"instructions": test_evaluater.data.inputs,
                                "preds": preds,
                                "labels": test_evaluater.data.labels})
        test_df.to_csv(os.path.join(self.out_dir, f"test_result.csv"), escapechar="\\")

        return preds
        