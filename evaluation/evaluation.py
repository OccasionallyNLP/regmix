# tokenize -> calculate just response part loss
import torch
from accelerate import Accelerator
from accelerate.utils import gather
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from datasets import load_dataset, Dataset
from tqdm import tqdm
import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_constant_schedule,
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling
)
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

def preprocess_function(examples, tokenizer):
    SEP=' '
    new_examples = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }
    for question, answer in zip(examples['question'], examples['answer']):
        tokenized = tokenizer(question + SEP + answer)
        question_tokenized = tokenizer.tokenize(question + SEP)
        new_examples["input_ids"].append(tokenized["input_ids"])
        new_examples["attention_mask"].append(tokenized["attention_mask"])
        _label = tokenized['input_ids'][:]
        len_question = len(question_tokenized)
        _label[:len_question]=[-100] * len_question
        new_examples["labels"].append(_label)
    return new_examples

def get_dataset(args, tokenizer):
    #dataset = load_dataset('json', data_files = args.data_name, split = args.split).select(range(10))
    dataset = load_dataset(args.data_name, split = args.split).select(range(10))
    dataset = dataset.map(
    lambda examples: preprocess_function(examples, tokenizer),
    batched=True,
    num_proc=args.num_proc,
    remove_columns=dataset.column_names)
    return dataset

def collate_fn(batch):
    input_ids = [torch.tensor(example['input_ids']) for example in batch]
    attention_mask = [torch.tensor(example['attention_mask']) for example in batch]

    # label_ids가 가변 길이 리스트라면 패딩 추가
    labels = [torch.tensor(example['labels']) for example in batch]
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)  # 필요 시 패딩 값 조정

    return {
        'input_ids': pad_sequence(input_ids, batch_first=True, padding_value=0),
        'attention_mask': pad_sequence(attention_mask, batch_first=True, padding_value=0),
        'labels': labels
    }

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default = 'Qwen/Qwen2.5-0.5B')
    parser.add_argument("--data_name", type=str, default = 'OccasionallyNLP/mmlu_test')
    parser.add_argument("--split", type=str, default = 'train')
    parser.add_argument("--output_dir", type=str, default = 'tmp')
    parser.add_argument("--name", type=str, default='1')
    parser.add_argument("--batch_size", type=int, default = 2)
    parser.add_argument("--num_proc", type=int, default = 1)
    # args = parser.parse_args()
    args = parser.parse_args()
    return args


def get_tokenizer_and_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    # for gpt 4 model
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "<|endoftext|>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
            args.model_path, device_map=accelerator.device)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    return tokenizer, model

if __name__ == "__main__":
    args = get_args()
    # sanity check
    os.makedirs(args.output_dir, exist_ok = True)
    accelerator = Accelerator()
    ##########################################################################
    # tokenizer, model
    tokenizer, model  = get_tokenizer_and_model(args)
    ##########################################################################

    ##########################################################################
    # data
    ##########################################################################
    dataset = get_dataset(args, tokenizer)
    dataloader = DataLoader(dataset, collate_fn = lambda example: collate_fn(example), batch_size = args.batch_size)
    ##########################################################################

    ##########################################################################
    # accelerate
    ##########################################################################
    model = accelerator.prepare(model)
    dataloader = accelerator.prepare(dataloader) 
    ##########################################################################
    
    model.eval()
    losses = []
    for data in tqdm(dataloader, disable=not accelerator.is_main_process):    
        with torch.no_grad():
            data = {i:j.to(accelerator.device) for i,j in data.items()}
            output = model.forward(
                input_ids=data["input_ids"],
                attention_mask=data["attention_mask"],
                labels=data["labels"],
            )
            loss = output.loss
            losses.append(accelerator.gather_for_metrics(loss.repeat(args.batch_size)))
    losses = torch.cat(losses)
    report_loss = losses.mean().item()

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, 'results.txt'),'w') as f:
            f.write(f'{args.name} -- {args.data_name} -- {str(report_loss)}')