# MuKA

> WIP

MuKA is intended to leverage multimodal documents into the knowledge retrieval and answer generations processes to answer the visual information-seeking problems.

## Data Preparation

We leveraged the [M2KR](https://huggingface.co/datasets/BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR) suite in our experiments, which was raised along with the PreFLMR models at [LinWeizheDragon/FLMR](https://github.com/LinWeizheDragon/FLMR). More specifically, we used the training/testing examples and knowledge bases for E-VQA and InfoSeek subset in the M2KR suite. 

## Knowledge Retrieval

### Retriever Training

### Retriever Indexing and Testing

## Answer Generation

The answer generators are first trained on reading examples and then tested. The reading examples for training are derived from the same knowledge retrieval results, for a fair comparison, which is 

### LLaVA Training

The [LLaVA-1.5](https://github.com/haotian-liu/LLaVA) models are trained to leverage a single image as the visual context.

We trained LLaVA-1.5 models for answer generators using the official LoRA fine-tuning script [finetune_lora.sh](https://github.com/haotian-liu/LLaVA/blob/main/scripts/v1_5/finetune_lora.sh), with super-parameters stated in the paper.

### VILA Training

The [VILA-1.5](https://github.com/NVlabs/VILA/tree/vila1.5) models are trained to handle contexts with multiple images as the visual context.

Following the [VILA-1.5/Installation](https://github.com/NVlabs/VILA/tree/vila1.5?tab=readme-ov-file#installation) instructions to install VILA-1.5 codebase first. We recommend to use an individual virtual environment for VILA-1.5 codebase since it does modifications to the `transformers` package in the installation instructions.

Since the official VILA-1.5 codebase does not provide official support for LoRA fine-tuning, we implement ourselves. We also provide a eval script 


## Acknowledgements

We extend our sincere thanks for the data, models and codebases aforementioned, which made it possible for this project.