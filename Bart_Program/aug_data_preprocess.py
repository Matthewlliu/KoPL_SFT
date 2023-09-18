import json
import os

def process_seq(seq):
    seq = seq.replace('<pad>', '')
    seq = seq.split('</s>')[0]
    seq = seq.split('<unk>')[0]
    return seq.strip()

root = '/data/ljx/result/probeLLM/kopl/lf2nl_kopl_text-davinci-003_2023-07-16_demo3_30000_seed1'
original_src = '/home/ljx/new_cache_server32_0411/KQAPro/dataset'

#file_num = 19
file_num = [0,3,4,5]
file_name = 'generated_ques_%s.json'

train_data = []
#for i in range(file_num):
for i in file_num:
    file = os.path.join(root, file_name % i)
    with open(file, 'r') as f:
        tmp = json.load(f)
    train_data.extend(tmp)
print(len(train_data))

out_file = os.path.join(root, 'train.json')
#for entry in train_data:
#    entry['question'] = process_seq(entry['question'])

with open(out_file, 'w') as f:
    json.dump(train_data, f)

cmd = ' '.join(['cp' ,os.path.join(original_src, 'test.json'), root])
os.system(cmd)

cmd = ' '.join(['cp' ,os.path.join(original_src, 'val.json'), root])
os.system(cmd)

cmd = ' '.join(['cp' ,os.path.join(original_src, 'kb.json'), root])
os.system(cmd)