#!/bin/bash
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

## the paths needs to be changed accordingly
processor_name="laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
checkpoint_path="./pretrained_models/PreFLMR_ViT-G"
dataset=Infoseek # "EVQA" for EVQA
image_root_dir="./Infoseek/val_images" # change accordingly
index_use_images="True"
title_key=title # "passage_id" for EVQA
doc_image_root_dir="./infoseek_passages_images" # change accordingly
doc_image_title2image="./infoseek_passages_title2image.json" # change accordingly

run_indexing_setting="--run_indexing"
index_root_path="./indexes"

split=test
Ks="1 5 10 20 50 100"
sample_examples_setting="" # "--sample_examples 100000" for the train split on Infoseek

index_name="example_index_name" # name to specify an index
experiment_name="example_experiment_name"
reports_dir="./reports" # dir to save the report


python preflmr_build_index.py \
  --use_gpu \
  ${run_indexing_setting} \
  --index_root_path "${index_root_path}" \
  --index_name "${index_name}" \
  --experiment_name "${experiment_name}" \
  --indexing_batch_size 64 \
  --image_root_dir "${image_root_dir}" \
  --dataset_hf_path "./multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR" \
  --dataset "${dataset}" \
  --use_split "${split}" \
  --nbits 8 \
  --num_gpus 8 \
  --Ks ${Ks} \
  --checkpoint_path "${checkpoint_path}" \
  --image_processor_name "${processor_name}" \
  --query_batch_size 32 \
  --compute_pseudo_recall \
  --save_report_path "${reports_dir}" \
  --doc_image_root_dir "${doc_image_root_dir}" \
  --doc_image_title2image "${doc_image_title2image}" \
  --index_use_images "${index_use_images}" \
  --title_key "${title_key}" \
  ${sample_examples_setting}