from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import DatasetDict
from evaluate import load as load_metric
from datasets import load_dataset
import torch
from torch import nn
from typing import Any, Dict, List
from transformers import default_data_collator
import numpy as np
import os

# Models to fine-tune
model_names = [
    't5-small',
    't5-base',
    'google/flan-t5-base',
    'facebook/bart-base',
]

# Datasets to fine-tune on
datasets_info = [
    'pii-masking-400k',
    'open-pii-masking-500k-ai4privacy'
]

# Preprocessing function
def get_preprocessor(tokenizer):
    def preprocess(example):
        input_text = 'Mask sensitive information: ' + example['source_text']
        target_text = example['masked_text']

        input_enc = tokenizer(input_text, truncation=True, padding='max_length', max_length=128, return_offsets_mapping=True)
        offset_mapping = input_enc['offset_mapping']
        input_ids = input_enc['input_ids']
        
        target_enc = tokenizer(target_text, truncation=True, padding='max_length', max_length=128)
        labels = target_enc['input_ids']
        labels = labels[:128] + [-100] * (128 - len(labels))

        span_mask = np.zeros(128, dtype=int)
        prefix_len = len('Mask sensitive information: ')
        for span in example.get('privacy_mask', []):
            start = span['start'] + prefix_len
            end = span['end'] + prefix_len
            for i, (token_start, token_end) in enumerate(offset_mapping):
                if token_start >= end:
                    break
                if token_end <= start:
                    continue
                if token_start < end and token_end > start:
                    span_mask[i] = 1

        return {
            'input_ids': input_enc['input_ids'],
            'attention_mask': input_enc['attention_mask'],
            'labels': labels,
            'span_labels': span_mask.tolist()
        }
    return preprocess

# Custom collator
def custom_data_collator(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    for f in features:
        if 'span_labels' not in f:
            f['span_labels'] = [0] * len(f['labels'])
    batch = default_data_collator(features)
    batch['span_labels'] = torch.tensor([f['span_labels'] for f in features])
    return batch

# Accuracy metric
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    correct = sum([pred.strip() == label.strip() for pred, label in zip(decoded_preds, decoded_labels)])
    accuracy = correct / len(decoded_preds)
    return {'accuracy': accuracy}

class PrivacyAwareTrainer(Seq2SeqTrainer):
    def __init__(self, penalty=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.penalty = penalty

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs['labels']
        span_labels = inputs['span_labels'].to(labels.device)
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=labels)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_span = span_labels[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(shift_labels.size())

        active_tokens = (shift_labels != -100).float()
        final_loss = (loss * active_tokens).sum() / active_tokens.sum()

        if self.penalty:
            penalty_weight = 5.0
            penalty_term = (penalty_weight * shift_span.float()).sum()
            final_loss = final_loss + penalty_term

        return (final_loss, outputs) if return_outputs else final_loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # REMOVE 'span_labels' before calling model.generate()
        inputs = {k: v for k, v in inputs.items() if k != 'span_labels'}
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

# Main loop to handle both normal loss and penalized loss
for dataset_name in datasets_info:
    print(f'\nLoading dataset: {dataset_name}')
    dataset = load_dataset(f'ai4privacy/{dataset_name}')
    # use 2000 examples for training and testeing
    english_dataset = dataset['train'].filter(lambda x: x['language'] == 'en')
    train_test = english_dataset.train_test_split(test_size=0.1, seed=42)
    
    for model_name in model_names:
        print(f'\nFine-tuning model: {model_name} on dataset: {dataset_name}')
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
        print(f'Using device: {device}')

        preprocessor = get_preprocessor(tokenizer)
        tokenized_train = train_test['train'].map(preprocessor, remove_columns=train_test['train'].column_names)
        tokenized_test = train_test['test'].map(preprocessor, remove_columns=train_test['test'].column_names)

        output_dir = f'./results-{dataset_name}-{model_name.replace("/", "-")}-full'

        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            learning_rate=5e-5,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            num_train_epochs=4,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=100,
            save_total_limit=1,
            save_steps=500,
            predict_with_generate=True,
            evaluation_strategy='epoch',
            report_to='none'
        )

        # Scenario 1: Fine-tuning with normal loss
        print(f'Fine-tuning with normal loss for {model_name} on dataset {dataset_name}')
        trainer_normal_loss = PrivacyAwareTrainer(
            penalty=False,  # No penalty for normal loss
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            tokenizer=tokenizer,
            data_collator=custom_data_collator,
            compute_metrics=compute_metrics
        )
        
        trainer_normal_loss.train()
        trainer_normal_loss.save_model(os.path.join(output_dir, 'normal-loss-final'))
        
        eval_results_normal = trainer_normal_loss.evaluate()
        print(f'Accuracy with normal loss on {dataset_name} with {model_name}: {eval_results_normal["eval_accuracy"]:.4f}')
        
        # Scenario 2: Fine-tuning with penalty on loss
        print(f'Fine-tuning with penalty on loss for {model_name} on dataset {dataset_name}')
        trainer_penalized_loss = PrivacyAwareTrainer(
            penalty=True,  # Penalty applied in the loss function
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            tokenizer=tokenizer,
            data_collator=custom_data_collator,
            compute_metrics=compute_metrics
        )
        
        trainer_penalized_loss.train()
        trainer_penalized_loss.save_model(os.path.join(output_dir, 'penalized-loss-final'))
        
        eval_results_penalized = trainer_penalized_loss.evaluate()
        print(f'Accuracy with penalized loss on {dataset_name} with {model_name}: {eval_results_penalized["eval_accuracy"]:.4f}')