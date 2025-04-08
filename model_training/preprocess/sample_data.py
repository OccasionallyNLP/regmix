import json
import glob
import os
from pathlib import Path
import sys
from typing import List
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Dataset
import argparse

def save_jsonl(data, path):
    with open(path, 'w',encoding='utf-8') as f:
        for example in tqdm(data,desc='save'):
            f.write(json.dumps(example, ensure_ascii=False)+'\n')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--size_of_data', type=int)
    parser.add_argument('--file_extention', type=str, default='jsonl')
    parser.add_argument('--sample_size', type=int, default=1100000000)
    parser.add_argument('--validation_ratio', type=float, default=0.1)
    parser.add_argument('--data_name', type=str)
    parser.add_argument('--output_path', type=str)
    args = parser.parse_args()
    return args

def get_samples(dataset, sample_size, validation_ratio, offset=0.75):
    cnt = 0
    output = []
    for i in dataset:
        output.append(i)
        try:
            token_length = i['metadata']['tokens_length']
        except:
            token_length = len(i['text'].split(' '))/offset
        cnt+=token_length
        if cnt>=sample_size:
            break
    validation_sample_size = int(len(output) * validation_ratio)
    train, valid = output[:-validation_sample_size], output[-validation_sample_size:]
    #train = Dataset.from_list(train)
    #valid = Dataset.from_list(valid)
    return train, valid

def main():
    args = get_args()
    data_paths = glob.glob(f"{args.data_path}/*{args.file_extention}")
    if args.size_of_data is not None:
        data_paths = data_paths[:args.size_of_data] 
    print(data_paths)
    data_paths = [os.path.join(args.data_path,i) for i in data_paths]
    for _, data_path in enumerate(data_paths):
        try:
            dataset = load_dataset('json', data_files=data_path, split='train', streaming=True)
            dataset = dataset.shuffle()
        except:
            print(data_path)
        train, valid = get_samples(dataset, args.sample_size, args.validation_ratio)
        save_name = os.path.join(args.output_path, 'train', f'{args.data_name}-{_}.jsonl')
        save_jsonl(train, save_name)
        #train.to_json(save_name)
        save_name = os.path.join(args.output_path, 'valid', f'{args.data_name}-{_}.jsonl')
        # valid.to_json(save_name)
        save_jsonl(valid, save_name)
    
if __name__=='__main__':
    main()