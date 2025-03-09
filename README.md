# MuKA

> WIP

MuKA is intended to leverage multimodal documents into the knowledge retrieval and answer generations processes to answer the visual information-seeking questions.

## Data Preparation

We leveraged the [M2KR](https://huggingface.co/datasets/BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR) suite in our experiments, which was raised along with the PreFLMR models at [LinWeizheDragon/FLMR](https://github.com/LinWeizheDragon/FLMR). More specifically, we used the examples and knowledge bases for the E-VQA and InfoSeek subset in the train/test splits of M2KR suite.

The train split of the InfoSeek released in M2KR provides up to \~676k examples, we sub-sampled 100k examples using the seed 42 to meet the M2KR statistics table.

## Knowledge Retrieval

The knowledge retrieval process is intended to provide retrieval results for building reading examples to train/test answer generators.

The MuKA retriever is initialized from the [PreFLMR-G](https://huggingface.co/LinWeizheDragon/PreFLMR_ViT-G) model.

### Retriever Training

### Retriever Indexing and Testing

## Answer Generation

The answer generators are first trained on the reading examples and then tested. A reading example provides the question, retrieved documents and a short instruction for the model to generate an answer.

The reading examples for training are derived from the same knowledge retrieval results across models, for a fair comparison. Such retrieval results for training are obtained from zero-shot inferencing the [PreFLMR-G](https://huggingface.co/LinWeizheDragon/PreFLMR_ViT-G) model in its official manner.

### LLaVA Training

The [LLaVA-1.5](https://github.com/haotian-liu/LLaVA) models are trained to leverage a single image as the visual context.

We trained LLaVA-1.5 models for answer generators using the official LoRA fine-tuning script [finetune_lora.sh](https://github.com/haotian-liu/LLaVA/blob/main/scripts/v1_5/finetune_lora.sh), with super-parameters stated in the paper.

### VILA

The [VILA-1.5](https://github.com/NVlabs/VILA/tree/vila1.5) models are trained to handle contexts with multiple images as the visual context.

**Installation**

Following the [VILA-1.5/Installation](https://github.com/NVlabs/VILA/tree/vila1.5?tab=readme-ov-file#installation) instructions to install VILA-1.5 codebase first. We recommend to use an individual virtual environment for VILA-1.5 codebase since it does modifications to the `transformers` package in the installation.

Since the official VILA-1.5 codebase does not provide an official script for LoRA fine-tuning, we implemented on our side for this purpose.

**Training**

**Evaluation**


## Acknowledgements

We extend our sincere thanks for the data, models and codebases aforementioned, which made it possible for this project.