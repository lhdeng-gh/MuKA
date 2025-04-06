#!/bin/bash
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

model_path="./pretrained_models/VILA1.5-13b" # change accordingly
conv_mode="v1"
vision_tower="google/siglip-so400m-patch14-384"

data_path="./training_examples.json" # change accordingly
images_base="./images_base" # change accordingly

per_device_train_batch_size=8
gradient_accumulation_steps=8

output_dir="./outputs_vila" # dir to save the model
run_name="example_run_name" # name to specify a run 


torchrun "vila_train.py" \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed "./zero3.json" \
    --model_name_or_path "${model_path}" \
    --version "${conv_mode}" \
    --data_mixture "" \
    --data_path "${data_path}" \
    --image_folder "${images_base}" \
    --vision_tower "${vision_tower}" \
    --mm_vision_select_feature cls_patch \
    --mm_projector mlp_downsample \
    --tune_vision_tower False \
    --tune_mm_projector True \
    --tune_language_model True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio resize \
    --bf16 True \
    --output_dir "${output_dir}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size "${per_device_train_batch_size}" \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps "${gradient_accumulation_steps}" \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 100 \
    --save_total_limit 1 \
    --save_only_model True \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --vflan_no_system_prompt True \
    --report_to wandb \
    --run_name "${run_name}"
