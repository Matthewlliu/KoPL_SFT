import os
gold_ans_file = '/data/ljx/data/KQAPro/dataset/train_ans.txt'
gold_ans = []
with open(gold_ans_file, 'r') as f:
    gold_ans = f.readlines()
gold_ans = [g.strip() for g in gold_ans]

predict_file = '/data/ljx/result/kqa_pro/bart-base_program_testaug_230301/preds-new_revised/predict.txt'
pred_ans = []
with open(predict_file, 'r') as f:
    pred_ans = f.readlines()
pred_ans = [g.strip() for g in pred_ans]

assert len(pred_ans)==len(gold_ans)
total = len(pred_ans)
count = 0
for pred, gold in zip(pred_ans, gold_ans):
    if pred == gold:
        count += 1
print("Acc: %s" % str(count/total))
