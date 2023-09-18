import os
import json
import torch

def formatting_input_func(example):
    output_texts = []
    for i in range(len(example['question'])):
        text = f"### Input: {example['question'][i]}\n ### Output:{example['program'][i]}"
        output_texts.append(text)
        return output_texts


def formatting_dataset(data_path, output_path):
    new_data = []
    with open(data_path, 'r') as f:
        data = json.load(f)
    for ind, entry in enumerate(data):
        text = f"### Input: {entry['question']}\n ### Output:{program2seq(entry['program'])}"
        new_data.append({"id": ind, "text": text})
    with open(output_path, 'w') as f:
        json.dump(new_data, f)

def eval_data_preprocess(eval_data):
    output = []
    for entry in eval_data:
        inputs = f"### Input: {entry['question']}\n ### Output:"
        #gt_str = program_serialize(entry['program'])
        gt_str = program2seq(entry['program'])
        ans = entry['answer']
        output.append([inputs, gt_str, ans])
    return output

def output_postprocess(string):
    #string = program_serialize(string)
    output_str = []
    parts = string.split(')')
    for part in parts:
        if len(part)==0:
            continue
        tmp = part.split('(')
        func_name = tmp[0]
        a = ''
        if len(tmp)>1:
            args = tmp[1]
            if len(args.strip()) > 0:
                aa = args.split(',')
                for arg in aa:
                    arg = arg.strip()
                    a += ' <arg> ' + arg
        output_str.append(func_name + a)
    output_str = ' <func> '.join(output_str)
    return output_str

def program2seq(program):
    seq = []
    for item in program:
        func = item['function']
        inputs = item['inputs']
        args = ''
        for input in inputs:
            args += ' [arg] ' + input
        seq.append(func + args)
    seq = ' [func] '.join(seq)
    return seq

def program_serialize( program):
    ret = ''
    for p in program:
        tmp = p['function']
        tmp += '(' + ', '.join(p['inputs']) + ')'
        ret += tmp
    return ret

if __name__=="__main__":
    data_path = '/home/ljx/new_cache_server32_0411/KQAPro/dataset/train_1w.json'
    output_path = '/home/ljx/new_cache_server32_0411/KQAPro/dataset/train_1w_formated_2.json'
    formatting_dataset(data_path, output_path)
    #string = 'FindAll()FilterNum(population, 7800, >)FilterConcept(county of Pennsylvania)FindAll()FilterNum(population, 40000000, <)FilterConcept'
    #print(output_postprocess(string))
