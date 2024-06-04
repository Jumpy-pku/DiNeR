import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers.trainer_pt_utils import LengthGroupedSampler
import random
from utils import NameParser


name_prompt = "这个菜谱的菜名是什么？"

comp_prompts = [
    "这个菜谱的主要操作是什么？",
    "这个菜谱的主要风味是什么？",
    "这个菜谱的主要食材是什么？"
]
cot_prompt = "这些元素组合出这个菜谱的菜名是什么？"

parser = NameParser()

def gen_component_label(name):
    word_list, type_list, _ = parser(name)
    labels = []
    for i in range(len(comp_prompts)):
        comps = [w for w, t in zip(word_list, type_list) if t == i]
        random.shuffle(comps)
        labels.append(",".join(comps))

    return labels


def create_dataset(args, split, tokenizer, pred_path=""):
    data = json.load(open("data/tmcd_data.json"))
    split_data = data[split]
    sorted_data = sorted(zip(split_data["instructions"], split_data["names"]), key=lambda x: len(tokenizer.encode(x)), reverse=True)
    split_data["instructions"] = [x[0] for x in sorted_data]
    split_data["names"] = [x[1] for x in sorted_data]
    

    inputs, labels = [], []
    if args.task == "name":
        inputs = [f"“{inst}” {name_prompt}" for inst in split_data["instructions"]]
        labels = split_data["names"]
    elif args.task == "component":
        for instruction, name in zip(split_data["instructions"], split_data["names"]):
            label = gen_component_label(name)
            for i, l in enumerate(label):
                inputs.append(f"“{instruction}” {comp_prompts[i]}")
                labels.append(l)
    elif args.task == "name_cpft":
        preds = torch.load(pred_path)[split]
        prompts = ["" for inst in split_data["instructions"]]
        for i, pred in enumerate(preds):
            if len(pred) > 0:
                prompts[i//len(comp_prompts)] += f"{comp_prompts[i%len(comp_prompts)][:-3]}{pred},"
        inputs = [f"“{inst}” {prompt}{name_prompt}" for inst, prompt in zip(split_data["instructions"], prompts)]
        labels = split_data["names"]

    print(len(inputs))
    return NameDataset(tokenizer, inputs, labels)

def create_dataloader(args, dataset, split):
        
    if split == "train":
        sampler = LengthGroupedSampler(args.train_bz, dataset, lengths=dataset.encodings["length"])
        return DataLoader(dataset, batch_size=args.train_bz, sampler=sampler, collate_fn=dynamic_padding, num_workers=8, drop_last=True)

    return DataLoader(dataset, batch_size=args.eval_bz, collate_fn=dynamic_padding, num_workers=8, shuffle=False)

class NameDataset(Dataset):
    def __init__(self, tokenizer, inputs, labels):
        super().__init__()
        self.inputs = inputs
        max_length = 512
        self.labels = labels
        print("tokenize data...")
        self.encodings = tokenizer(self.inputs, max_length=max_length, padding=False, truncation=True, return_length=True)
        self.label_encodings = tokenizer(text_target=self.labels, max_length=max_length, padding=False, truncation=True, return_length=True)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        item = {k: v[index] for k, v in self.encodings.items()}
        item['label'] = self.label_encodings['input_ids'][index]
        item["label_length"] = self.label_encodings['length'][index]
        return item
    
def collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["label"] for b in batch])
    }

def dynamic_padding(batch):
        lengths = torch.LongTensor([b["length"] for b in batch])
        max_length = lengths.max()
        max_label_length = max([b["label_length"] for b in batch])
        if max_label_length > max_length:
            max_length = max_label_length

        input_ids = torch.zeros((len(batch), max_length), dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_length), dtype=torch.long)
        labels = torch.fill(torch.zeros((len(batch), max_label_length), dtype=torch.long), -100)

        for i, b in enumerate(batch):
            input_ids[i, :b["length"]] = torch.LongTensor(b["input_ids"])
            attention_mask[i, :b["length"]] = torch.LongTensor(b["attention_mask"])
            labels[i, :b["label_length"]] = torch.LongTensor(b["label"])

        ret = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        return ret