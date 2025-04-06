# MuKA

MuKA is intended to leverage multimodal passages into the knowledge retrieval and answer generation processes of a RAG pipeline to answer the visual information-seeking questions.

## Data Preparation

We leveraged the [M2KR](https://huggingface.co/datasets/BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR) suite for our experiments, which was raised along with PreFLMR models by [LinWeizheDragon/FLMR](https://github.com/LinWeizheDragon/FLMR).

The train split of InfoSeek is sub-sampled to meet the statistics: `data.shuffle(seed=42).select(range(100000))`.

The urls for the entity images are provided in the `data_preparation` folder. If image_url is null, a black image is used as the placeholder. Large images were scaled proportionally till the shorter side is 512px.

After downloading the entity images, a index json is required for the following scripts, which contains `title: image_path` key-value pairs. The titles are provided in the urls file.

> The `title` is actually the `title` column for the InfoSeek passages in M2KR and `passage_id` for EVQA. Change the `title_key` accordingly in the scripts.

## Knowledge Retrieval

The knowledge retrieval process is intended to provide retrieval results for building examples to train/test answer generators.

### Installation

Following the instructions of [FLMR/how-to-use-this-package](https://github.com/LinWeizheDragon/FLMR?tab=readme-ov-file#how-to-use-this-package) to clone and install FLMR first. We recommend to install it as editable using `pip -e` for development purposes.

Our modifications are provided in diffs. `flmr_diffs_mask` folder is for the MuKA retriever, `flmr_diffs_nomask` refer to the MuKA retriever without mask. The mask is only implemented for the [PreFLMR-G](https://huggingface.co/LinWeizheDragon/PreFLMR_ViT-G) model, modify accordingly for other models.

### Retriever Training

We implemented the training in `preflmr_train.py` using the huggingface trainer, which is paired with the `preflmr_train.sh` script.

### Retriever Testing

For indexing and testing, please refer to `preflmr_build_index.py`, which is pair with the `preflmr_test.sh` script.

> The `preflmr_build_index.py` was adapted from `examples/example_use_preflmr.py` from the [LinWeizheDragon/FLMR](https://github.com/LinWeizheDragon/FLMR) repo.

## Answer Generation

The answer generators are first trained on the reading examples and then tested. A reading example provides the question, retrieved documents and a short instruction for the model to generate an answer.

The reading examples for training are derived from the same knowledge retrieval results across models, for a fair comparison. Such retrieval results for training are obtained from a zero-shot inference of the [PreFLMR-G](https://huggingface.co/LinWeizheDragon/PreFLMR_ViT-G) model with its official codebase.

### LLaVA Training and Testing

The [LLaVA-1.5](https://github.com/haotian-liu/LLaVA) models are trained to leverage a single image as the visual context.

We trained LLaVA-1.5 models using the official LoRA fine-tuning script [finetune_lora.sh](https://github.com/haotian-liu/LLaVA/blob/main/scripts/v1_5/finetune_lora.sh), and tested with the official eval script [model_vqa_loader.py](https://github.com/haotian-liu/LLaVA/blob/main/llava/eval/model_vqa_loader.py).

### VILA

The [VILA-1.5](https://github.com/NVlabs/VILA/tree/vila1.5) models are trained to handle visual contexts with multiple images.

#### Installation

Following the [VILA-1.5/Installation](https://github.com/NVlabs/VILA/tree/vila1.5?tab=readme-ov-file#installation) instructions to clone and install first. We recommend to use a virtual environment since it does modifications to the `transformers` package. We recommend to install it as editable using `pip -e` for development purposes.

Since the official VILA-1.5 codebase does not provide an official script for LoRA fine-tuning, we implemented on our side for this purpose. Please refer to the `vila_diffs` folder.

#### Training

Please refer to `vila_train.py`, which is paired with `vila_train_lora.sh`.

#### Testing

Please refer to `vila_model_vqa_loader.py`, which is paired with `vila_eval_lora.sh`.

To test VILA with multiple images, use `<image>` as the placeholder in the text, and provide the image paths in a list.

## Acknowledgements

We extend our sincere thanks for the authors who created the resources aforementioned, which made it possible for this project.

## Disclaimer

The images we have collected are for research purposes only, and we shall not be held responsible for any issues arising from their use.

The scripts are cleaned but not tested, please check them before the run, and we shall not be held responsible for any issues arising from their use.