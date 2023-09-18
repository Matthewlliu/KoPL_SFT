import argparse
import os
import json

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, GenerationConfig
from util import eval_data_preprocess, output_postprocess, program2seq

from Bart_Program.data import DataLoader
from Bart_Program.executor_rule import RuleExecutor

def parse_args():
    parser = argparse.ArgumentParser()
    # model arguments
    parser.add_argument("--base_model_name", type=str, help="the base model name")
    parser.add_argument("--checkpoint_path", type=str, help="the checkpoint path")
    parser.add_argument("--eval_path", type=str, help="the checkpoint path")
    parser.add_argument("--cache_dir", type=str, default=None, help="The cache directory to save the model")
    parser.add_argument("--load_in_8bit", action='store_true', help="load the model in 8 bits precision")
    parser.add_argument("--load_in_4bit", action='store_true', help="load the model in 4 bits precision")
    parser.add_argument("--trust_remote_code", type=bool, default=True, help="Enable `trust_remote_code`")
    parser.add_argument("--use_auth_token", type=bool, default=True, help="Use HF auth token to access the model")
    # inference arguments
    parser.add_argument("--do_sample", type=bool, default=True, help="Enable sampling")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling softmax temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="The maximum number of tokens to generate")
    args = parser.parse_args()
    return args

class LoRA_Inference:
    def __init__(self, args):
        self.base_model_name = args.base_model_name
        self.checkpoint_path = args.checkpoint_path
        self.args = args

        # setup generation config
        self.generation_config = GenerationConfig(
            do_sample = args.do_sample,
            temperature = args.temperature,
            top_p = args.top_p,
            num_return_sequences = args.num_return_sequences, 
            max_new_tokens = args.max_new_tokens)
        
        self.load_model()

    def load_model(self):
        # load base model
        if self.args.load_in_8bit and self.args.load_in_4bit:
            raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
        elif self.args.load_in_8bit or self.args.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=self.args.load_in_8bit, load_in_4bit=self.args.load_in_4bit
            )
            # device_map = {"": 0} # fit the entire model on the GPU:0
            device_map = "auto"
            torch_dtype = torch.bfloat16
        else:
            device_map = None
            quantization_config = None
            torch_dtype = None

        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            trust_remote_code=self.args.trust_remote_code,
            torch_dtype=torch_dtype,
            use_auth_token=self.args.use_auth_token,
            cache_dir=self.args.cache_dir,
        )
        print(f"Base Model {self.base_model_name} loaded")

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, trust_remote_code=self.args.trust_remote_code)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.generation_config.eos_token_id = self.tokenizer.eos_token_id

        # load peft model
        self.model = PeftModel.from_pretrained(self.base_model, self.checkpoint_path)
        print(f"PEFT Model {self.checkpoint_path} loaded")

    # generate for a single input
    def generate(self, input_string):
        # call tokenizer
        inputs = self.tokenizer(input_string, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            generate_ids = self.model.generate(**inputs, generation_config=self.generation_config)
            # only output the generated tokens
            input_length = 1 if self.model.config.is_encoder_decoder else inputs.input_ids.shape[1]
            generate_ids = generate_ids[:, input_length:]
        return self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

if __name__ == "__main__":
    args = parse_args()
    lora_inference = LoRA_Inference(args)
    #input_str = "Read the Instruction below and provide an answer.\n### INSTRUCTION:\nRead the following table and answer a question and generate the explanation.\n\nCaption: Kelli Maroney Filmography\nTable:\n|| Year | Title | Role | Notes ||\n|| 1982 | Fast Times at Ridgemont High | Cindy | - ||\n|| 1983 | Slayground | Jolene | - ||\n|| 1984 | Night of the Comet | Samantha | - ||\n|| 1986 | Chopping Mall | Alison Parks | - ||\n|| 1986 | The Zero Boys | Jamie | - ||\n|| 1987 | Big Bad Mama II | Willie McClatchie | - ||\n|| 1988 | Not of This Earth | Nurse Oxford | - ||\n|| 1989 | Jaded | Jennifer | - ||\n|| 1989 | Transylvania Twist | Hannah | - ||\n|| 1990 | Hard to Die | Porno Wife | - ||\n|| 1991 | Servants of Twilight | Sherry Ordway | - ||\n|| 1991 | Scream Queen Hot Tub Party | Herself | Video ||\n|| 1993 | Midnight Witness | Devon | - ||\n|| 1999 | Sam and Mike | - | Short film ||\n|| 2004 | Audition | Brett | Short film ||\n|| 2011 | Lip Service | Janice | - ||\n|| 2012 | Dark Star Hollow | Sarah Rose Barteaux | Post-production ||\n\nQuestion: Aside from Jolene in Slayground, what were Kelli Maroney's first four other roles?\n\n### RESPONSE:"
    #gt_answer = "Kelli Maroney's film roles were Cindy Carr in Fast Times at Ridgemont High (1982), Samantha in Night of the Comet (1984), Jamie in The Zero Boys, and Allison in Chopping Mall (1986).\nTherefore the answer is Kelli Maroney's film roles were Cindy Carr in Fast Times at Ridgemont High (1982), Samantha in Night of the Comet (1984), Jamie in The Zero Boys, and Allison in Chopping Mall (1986).\n\n### End"
    
    #input_str = "Read the Instruction below and provide an answer.\n### INSTRUCTION:\nRead the following table and answer a question and generate the explanation.\n\nCaption: Roscoe Parrish Personal bests\nTable:\n|| Event | Time (seconds) | Venue | Date ||\n|| 55 meters | 6.38 | Gainesville, Florida | January 18, 2003 ||\n|| 60 meters | 6.89 | Syracuse, New York | February 16, 2002 ||\n|| 100 meters | 10.65 | Coral Gables, Florida | April 12, 2003 ||\n|| 200 meters | 21.13 | Storrs, Connecticut | May 4, 2003 ||\n\nQuestion: What is the personal best of Roscoe Parrish in 55 meters, and where and when did he make it?\n\n### RESPONSE:"
    #gt_answer = "Roscoe Parrish competed in the 55 meters, posting a personal best time of 6.38 seconds in Gainesville, Florida on January 18, 2003 \nTherefore the answer is Roscoe Parrish competed in the 55 meters, posting a personal best time of 6.38 seconds in Gainesville, Florida on January 18, 2003 \n\n### End"
    with open(args.eval_path, 'r') as f:
        eval_data = json.load(f)
    eval_data = eval_data_preprocess(eval_data)

    # kb executor
    data_root = '/home/ljx/new_cache_server32_0411/KQAPro/dataset/Codellama_processed'
    vocab_json = os.path.join(data_root, 'vocab.json')
    val_pt = os.path.join(data_root, 'val.pt')
    val_loader = DataLoader(vocab_json, val_pt, 256)
    vocab = val_loader.vocab #train_loader.vocab
    executor = RuleExecutor(vocab, os.path.join(data_root, 'kb.json'))

    pred_file = args.checkpoint_path.split('/')[0] + 'pred.json'
    pred_f = open(pred_file, 'w')

    em = 0
    correct = 0
    eval_num = 1000
    eval_data = eval_data[:eval_num]
    count = 0
    for input_str, gt_str, gt_ans in tqdm(eval_data):
        inputs = [input_str]
        output_str = lora_inference.generate(inputs)
        output_str = output_str[0]
        output_str = output_str.split('\n')[0]

        # exact match score
        if output_str == gt_str:
            em += 1
        print(output_str)
        print(gt_str)

        pred_f.write(str(count+1)+'.\n')
        pred_f.write(output_str+'\n')
        pred_f.write(gt_str+'\n')
        count += 1
        #print(em)

        # accuracy
        #output_str = output_postprocess(output_str)
        chunks = output_str.split('[func]')
        func_list = []
        inputs_list = []
        for chunk in chunks:
            chunk = chunk.strip()
            res = chunk.split('[arg]')
            res = [_.strip() for _ in res]
            if len(res) > 0:
                func = res[0]
                inputs = []
                if len(res) > 1:
                    for x in res[1:]:
                        inputs.append(x)
                else:
                    inputs = []
                func_list.append(func)
                inputs_list.append(inputs)
        ans = executor.forward(func_list, inputs_list, ignore_error = True)
        if ans == None:
            ans = 'no'
        if ans == gt_ans:
            correct += 1
        print(ans)
        print(gt_ans)
        #print(correct)
        input()
    print("Exact Match Score: {}".format(em/len(eval_data)))
    print("Accuracy: {}".format(correct/len(eval_data)))

    pred_f.close()