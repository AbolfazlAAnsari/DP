from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator
)
from datasets import load_dataset
import torch
from torch import nn
from typing import Any, Dict, List
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
def get_preprocessor(tokenizer):
    def preprocess(example):
        input_text = 'Mask sensitive information: ' + example['source_text']
        target_text = example['masked_text']

        input_enc = tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_offsets_mapping=True
        )
        offset_mapping = input_enc.pop('offset_mapping')

        target_enc = tokenizer(
            target_text,
            truncation=True,
            padding='max_length',
            max_length=128
        )
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
                span_mask[i] = 1

        return {
            'input_ids': input_enc['input_ids'],
            'attention_mask': input_enc['attention_mask'],
            'labels': labels,
            'span_labels': span_mask.tolist(),
        }
    return preprocess

# Data collator
def custom_data_collator(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    for feature in features:
        if 'span_labels' not in feature:
            feature['span_labels'] = [0] * len(feature['labels'])
    batch = default_data_collator(features)
    batch['span_labels'] = torch.tensor([f['span_labels'] for f in features])
    return batch

# Metric
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    correct = sum(pred.strip() == label.strip() for pred, label in zip(decoded_preds, decoded_labels))
    accuracy = correct / len(decoded_preds)
    return {'accuracy': accuracy}

# Custom Trainer
class PrivacyAwareTrainer(Seq2SeqTrainer):
    def __init__(self, penalty=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.penalty = penalty

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs['labels']
        span_labels = inputs['span_labels'].to(labels.device)
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            labels=labels
        )
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
            penalty_term = (penalty_weight * shift_span.float()).sum() / active_tokens.sum()
            final_loss += penalty_term

        return (final_loss, outputs) if return_outputs else final_loss

# Main loop
for dataset_name in datasets_info:
    print(f'\nLoading dataset: {dataset_name}')
    dataset = load_dataset(f'ai4privacy/{dataset_name}')
    english_dataset = dataset['train'].filter(lambda x: x['language'] == 'en')

    # TAKE ONLY 1000 examples
    english_dataset = english_dataset.select(range(min(1000, len(english_dataset))))

    train_test = english_dataset.train_test_split(test_size=0.1, seed=42)

    for model_name in model_names:
        print(f'\nFine-tuning model: {model_name} on {dataset_name}')

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        preprocessor = get_preprocessor(tokenizer)
        tokenized_train = train_test['train'].map(
            preprocessor, remove_columns=train_test['train'].column_names
        )
        tokenized_test = train_test['test'].map(
            preprocessor, remove_columns=train_test['test'].column_names
        )

        output_dir = f'./results-small-{dataset_name}-{model_name.replace("/", "-")}'

        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            evaluation_strategy='epoch',
            save_strategy='epoch',
            learning_rate=5e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=1,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=50,
            save_total_limit=1,
            predict_with_generate=True,
            report_to='none',
            push_to_hub=False
        )

        # Normal loss training
        trainer_normal = PrivacyAwareTrainer(
            penalty=False,
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            tokenizer=tokenizer,
            data_collator=custom_data_collator,
            compute_metrics=compute_metrics,
        )

        print(f'\nTraining {model_name} (normal loss)...')
        trainer_normal.train()
        trainer_normal.save_model(os.path.join(output_dir, 'normal-loss-final'))

        eval_normal = trainer_normal.evaluate()
        print(f'Normal loss accuracy: {eval_normal["eval_accuracy"]:.4f}')