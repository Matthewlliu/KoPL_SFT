import json
import os

root = '/data/ljx/result/probeLLM/kopl/human_origin_2w'
original_src = '/home/ljx/new_cache_server32_0411/KQAPro/dataset'

if not os.path.exists(root):
    os.makedirs(root)

with open(os.path.join(original_src, 'train.json'), 'r') as f:
    data = json.load(f)

train_data = []

train_data.extend(data[:5000])
train_data.extend(data[15000:30000])

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