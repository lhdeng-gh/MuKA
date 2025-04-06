import transformers
from transformers import TrainingArguments, Trainer, HfArgumentParser
from transformers import AutoImageProcessor
from transformers import AutoConfig
from datasets import load_dataset

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.utils.data import random_split

import warnings
import random
import os
import json
from dataclasses import dataclass
from PIL import Image
from pprint import pformat

from flmr import FLMRQueryEncoderTokenizer, FLMRContextEncoderTokenizer, FLMRModelForRetrieval


@dataclass
class MyArguments:
    model_name_or_path :str = "LinWeizheDragon/PreFLMR_ViT-G"
    image_processor_name :str = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
    dataset_hf_path :str = "BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR"
    dataset :str = "Infoseek"
    sample_examples :int = -1
    num_negative_examples :int = 4
    image_root_dir :str = "./Infoseek/train_images"
    split_eval_from_train_examples :int = -1
    freeze_vision_encoder :bool = True
    freeze_text_encoder :bool = False
    # We use the Adam optimizer (Kingma and Ba, 2015) 
    # with a fixed learning rate of 10−4 for the mapping structure 
    # and 10−5 for the rest parameters in all experiments in all training stages.
    mapping_structure_lr :float = 1e-4
    non_mapping_structure_lr :float = 1e-5    
    doc_use_images :bool = False
    doc_image_root_dir :str = ""
    doc_image_title2image :str = ""
    title_key: str = ""


class PreFLMRTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs, return_dict=True)
        ib_loss = outputs["in_batch_negative_loss"]
        outputs["loss"] = ib_loss

        return (ib_loss, outputs) if return_outputs else ib_loss

    # https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py#L3353C5-L3353C90
    def save_model(self, output_dir=None, _internal_call: bool = False):
        super().save_model(output_dir, _internal_call)
        if output_dir is None:
            output_dir = self.args.output_dir
        
        self.query_tokenizer.save_pretrained(os.path.join(output_dir, 'query_tokenizer'))
        self.context_tokenizer.save_pretrained(os.path.join(output_dir, 'context_tokenizer'))


class PreFLMRDataset(Dataset):
    
    def __init__(self,
                 args,
                 data_df, passages_df, 
                 query_tokenizer, context_tokenizer, image_processor):
        self.args = args
        self.data_df = data_df
        self.passages_df = passages_df
        self.query_tokenizer = query_tokenizer
        self.context_tokenizer = context_tokenizer
        self.image_processor = image_processor
        
        self.unique_passage_ids = set(self.passages_df.index)
        
        if self.args.doc_use_images:
            self.doc_image_title2image = json.load(open(self.args.doc_image_title2image))
    
    
    def __len__(self):
        return len(self.data_df)
    
    
    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        query = row['instruction'] + row['question']
        
        pos_item_ids = row['pos_item_ids']
        pos_item_id = random.choice(pos_item_ids)
        pos_psg_row = self.passages_df.loc[pos_item_id]
        pos_passage = pos_psg_row['passage_content']
        
        query_image_path = os.path.join(self.args.image_root_dir, row['img_path'])
        query_image = Image.open(query_image_path).convert('RGB')
        query_pixel_values = self.image_processor(query_image, return_tensors='pt')['pixel_values'] # [1, 3, 224, 224]
        
        # negatives
        neg_item_ids = random.sample(list(self.unique_passage_ids - set(pos_item_ids)), 
                                     self.args.num_negative_examples)
        neg_psg_rows = [ self.passages_df.loc[neg_item_id] for neg_item_id in neg_item_ids ]
        neg_passages = [r['passage_content'] for r in neg_psg_rows]
        
        passages = [pos_passage] + neg_passages
        
        inputs = dict(
            query=query,
            passages=passages,
            query_pixel_values=query_pixel_values
        )
        
        if self.args.doc_use_images:
            pos_image_path = os.path.join(self.args.doc_image_root_dir, self.doc_image_title2image[pos_psg_row[self.args.title_key]])
            neg_image_paths = [ os.path.join(self.args.doc_image_root_dir, self.doc_image_title2image[r[self.args.title_key]])
                               for r in neg_psg_rows]
            context_images = [Image.open(image_path).convert('RGB') 
                              for image_path in [pos_image_path] + neg_image_paths]
            context_pixel_values = self.image_processor(context_images, return_tensors='pt')['pixel_values']
            
            inputs["context_pixel_values"] = context_pixel_values
        
        return inputs
        
    def collate_fn(self, batch):
        queries = [ex['query'] for ex in batch]
        passages = [] # [pos, neg, neg, neg, pos, ...]
        for ex in batch:
            passages.extend(ex['passages'])

        Q_encoding = self.query_tokenizer(queries)
        Q_pixel_values = torch.cat([ex['query_pixel_values'] for ex in batch], dim=0)
        D_encoding = self.context_tokenizer(passages)
        
        # according to `modeling_flmr.py:FLMRModelForRetrieval.forward`
        inputs = dict(
            query_input_ids=Q_encoding['input_ids'],
            query_attention_mask=Q_encoding['attention_mask'],
            query_pixel_values=Q_pixel_values,
            context_input_ids=D_encoding['input_ids'],
            context_attention_mask=D_encoding['attention_mask'],
            use_in_batch_negatives=True,
            in_batch_negatives_from_all_gpus=False,
            num_negative_examples=self.args.num_negative_examples,
            query_concat_output_from_vision_encoder=True,
            query_concat_output_from_text_encoder=True,
            context_concat_output_from_vision_encoder=False,
            context_concat_output_from_text_encoder=True,
        )
        
        if self.args.doc_use_images:
            context_pixel_values = torch.cat([ex['context_pixel_values'] for ex in batch], dim=0)
            inputs['context_pixel_values'] = context_pixel_values
            inputs['context_concat_output_from_vision_encoder'] = True
            
        return inputs


def main():
    parser = HfArgumentParser((MyArguments, TrainingArguments))
    my_args, training_args = parser.parse_args_into_dataclasses()
    
    ## setting up
    if dist.get_world_size() != 4:
        warnings.warn(
            'In paper, 4 Nvidia A100 GPUs were used with data parallel in all experiments. '
            f'Found {dist.get_world_size()} gpus.'
        )
    
    if dist.get_rank() == 0:
        print(f'## my_args: {pformat(my_args)}')
        print(f'## training_args: {pformat(training_args)}')
    
    transformers.set_seed(training_args.seed)
    
    ## setting up tokenizer
    query_tokenizer = FLMRQueryEncoderTokenizer.from_pretrained(
        my_args.model_name_or_path, subfolder="query_tokenizer")
    context_tokenizer = FLMRContextEncoderTokenizer.from_pretrained(
        my_args.model_name_or_path, subfolder="context_tokenizer")
    image_processor = AutoImageProcessor.from_pretrained(my_args.image_processor_name)

    ## setting up dataset
    data = load_dataset(my_args.dataset_hf_path, f'{my_args.dataset}_data')['train']
    if my_args.sample_examples != -1:
        print(f'## sampling examples with seed 42 into {my_args.sample_examples}')
        # for infoseek, to keep consistency with testing script
        data = data.shuffle(seed=42).select(range(my_args.sample_examples))
    data_df = data.to_pandas().set_index('question_id')
    
    passages_df = load_dataset(my_args.dataset_hf_path, f'{my_args.dataset}_passages')['train_passages']\
        .to_pandas()
    
    # evqa have duplicates in passage_id, deduplicate here
    if len(passages_df) != len(passages_df['passage_id'].unique()):
        print('## deduplicating passage_ids, before: {}, after: {}'.format(
            len(passages_df),
            len(passages_df['passage_id'].unique())
        ))
        passages_df.drop_duplicates('passage_id', inplace=True)
    
    # keep passage_id column, evqa needs it
    passages_df['passage_id_index'] = passages_df['passage_id']
    passages_df.set_index('passage_id_index', inplace=True)
        
    dataset = PreFLMRDataset(args=my_args,
                             data_df=data_df, passages_df=passages_df,
                             query_tokenizer=query_tokenizer, 
                             context_tokenizer=context_tokenizer,
                             image_processor=image_processor)
    collate_fn = dataset.collate_fn
    
    if my_args.split_eval_from_train_examples != -1:
        print(f'## splitting eval set of size {my_args.split_eval_from_train_examples} from training set...')
        torch.manual_seed(training_args.seed)
        dataset, eval_dataset = random_split(dataset, [
            len(dataset) - my_args.split_eval_from_train_examples,
            my_args.split_eval_from_train_examples
        ])
    elif training_args.do_eval:
        print(f'## building eval dataset...')
        eval_data_df = load_dataset(my_args.dataset_hf_path, f'{my_args.dataset}_data')['valid']\
            .to_pandas().set_index('question_id')
        eval_passages_df = load_dataset(my_args.dataset_hf_path, f'{my_args.dataset}_passages')['valid_passages']\
            .to_pandas().set_index('passage_id')
        eval_dataset = PreFLMRDataset(args=my_args,
                                    data_df=eval_data_df, passages_df=eval_passages_df,
                                    query_tokenizer=query_tokenizer, 
                                    context_tokenizer=context_tokenizer,
                                    image_processor=image_processor)
    else:
        eval_dataset = None
    
    if dist.get_rank() == 0:
        print(f'## len(dataset): {len(dataset)}')
        print(f'## dataset[0]: {pformat(dataset[0])}')
        print(f'## eval_dataset: {pformat(eval_dataset)}')
    
    ## setting up model
    model = FLMRModelForRetrieval.from_pretrained(
        my_args.model_name_or_path,
        query_tokenizer=query_tokenizer,
        context_tokenizer=context_tokenizer)
    
    ## setting up training
    # make modules into group
    vision_encoder_modules = [
        model.query_vision_encoder, # FLMRVisionModel
        model.context_vision_encoder # FLMRVisionModel
    ]
    text_encoder_modules = [
        model.query_text_encoder, # FLMRTextModel
        model.context_text_encoder, # FLMRTextModel
    ]
    mapping_structure_modules = [
        model.query_vision_projection, # FLMRMultiLayerPerceptron
        model.context_vision_projection, # FLMRMultiLayerPerceptron
        model.transformer_mapping_network, # BertEncoder
    ]
    non_mapping_structure_modules = [
        model.query_text_encoder_linear, # Linear
        model.context_text_encoder_linear, # Linear
        model.transformer_mapping_input_linear, # Linear
        model.transformer_mapping_output_linear, # Linear
    ]
    
    assert set(id(p) for p in model.parameters()) == set(id(p) \
        for module in vision_encoder_modules + text_encoder_modules \
            + mapping_structure_modules + non_mapping_structure_modules
        for p in module.parameters()), 'there are parameters in model that has not been grouped yet.'
    
    # build trainables    
    if my_args.freeze_vision_encoder:
        for module in vision_encoder_modules:
            for p in module.parameters():
                p.requires_grad = False
    else:
        non_mapping_structure_modules += vision_encoder_modules
    
    if my_args.freeze_text_encoder:
        for module in text_encoder_modules:
            for p in module.parameters():
                p.requires_grad = False
    else:
        non_mapping_structure_modules += text_encoder_modules
    
    if dist.get_rank() == 0:
        trainables = [pn for pn, p in model.named_parameters() if p.requires_grad]
        n_trainables = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'## trainables: {pformat(trainables)}')
        print(f'## n_trainables: {n_trainables:,}')
        print('## mapping_structure_modules parameters (lr={}): {}'.format(
            my_args.mapping_structure_lr,
            pformat([pn for module in mapping_structure_modules for pn, p in module.named_parameters()])
        ))
        print('## non_mapping_structure_modules parameters (lr={}): {}'.format(
            my_args.non_mapping_structure_lr,
            pformat([pn for module in non_mapping_structure_modules for pn, p in module.named_parameters()])
        ))
    
    # build optimizer and constant scheduler
    #   mapping_structure_modules: a 2-layer MLP_F^MLP and a Transformer block F_M^TR, lr = 1e-4
    #   non_mapping_structure_modules: remaining modules, mostly linears. lr = 1e-5
    # need to deduplicate modules (some modules may share parameters)
    optimizer_groups = []
    
    mapping_structure_modules_dedup = list({id(m) : m for m in mapping_structure_modules}.values())
    for module in mapping_structure_modules_dedup:
        optimizer_groups.append(dict(params=module.parameters(), lr=my_args.mapping_structure_lr))
    
    non_mapping_structure_modules_dedup = list({id(m) : m for m in (non_mapping_structure_modules)}.values())
    for module in non_mapping_structure_modules_dedup:
        optimizer_groups.append(dict(params=module.parameters(), lr=my_args.non_mapping_structure_lr))
    
    optimizer = torch.optim.Adam(optimizer_groups)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
    
    ## start training
    trainer = PreFLMRTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        optimizers=[optimizer, scheduler]
    )
    # for convenience
    trainer.query_tokenizer = query_tokenizer
    trainer.context_tokenizer = context_tokenizer
    # since modeling_flmr.forward does not have return_loss param, the trainer
    #   wrongly thought it can not return loss, override it
    trainer.can_return_loss = True
    
    trainer.train()
    trainer.save_state()
    trainer.save_model()


if __name__ == '__main__':
    main()