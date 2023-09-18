CUDA_VISIBLE_DEVICES=2 python lora_inference.py \
    --base_model_name /data1/ljx/cpt/Codellama-7b-hf \
    --checkpoint_path /data1/ljx/result/lora/Codellama-7b-hf_kopl_2023-09-16-00:43_epoch30_bs32_lr8e-05_1w/checkpoint-4680 \
    --load_in_8bit \
    --do_sample True\
    --temperature 1.0 \
    --top_p 0.92 \
    --eval_path /home/ljx/new_cache_server32_0411/KQAPro/dataset/val.json
