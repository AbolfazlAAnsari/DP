from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset
from evaluate import load as load_metric
from transformers import default_data_collator
from torch import nn
import numpy as np
import torch
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

        input_enc = tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_offsets_mapping=True
        )
        offset_mapping = input_enc.pop('offset_mapping')
        input_ids = input_enc['input_ids']

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
                if token_start is None or token_end is None:
                    continue
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
def custom_data_collator(features):
    for f in features:
        if 'span_labels' not in f:
            f['span_labels'] = [0] * len(f['labels'])
    batch = default_data_collator(features)
    batch['span_labels'] = torch.tensor([f['span_labels'] for f in features])
    return batch

class PrivacyAwareTrainer(Seq2SeqTrainer):
    def __init__(self, penalty=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.penalty = penalty

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs['labels']
        span_labels = inputs['span_labels'].to(labels.device)
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            labels=labels
        )
        logits = outputs.logits

        # Shift logits and labels
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_span = span_labels[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(shift_labels.size())

        active_tokens = (shift_labels != -100).float()
        base_loss = (loss * active_tokens).sum() / active_tokens.sum()

        if self.penalty:
            penalty_weight = 5.0
            penalty_term = (shift_span.float() * active_tokens).sum() / active_tokens.sum()
            base_loss += penalty_weight * penalty_term

        return (base_loss, outputs) if return_outputs else base_loss
# Main training loop
for dataset_name in datasets_info:
    print(f'\nLoading dataset: {dataset_name}')
    dataset = load_dataset(f'ai4privacy/{dataset_name}')
    english_dataset = dataset['train'].filter(lambda x: x['language'] == 'en')

    train_test = english_dataset.train_test_split(test_size=0.1, seed=42)

    for model_name in model_names:
        print(f'\nFine-tuning model: {model_name} on dataset: {dataset_name}')

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_init = lambda: AutoModelForSeq2SeqLM.from_pretrained(model_name)  # model re-init function

        preprocessor = get_preprocessor(tokenizer)
        tokenized_train = train_test['train'].map(preprocessor, remove_columns=train_test['train'].column_names)
        tokenized_test = train_test['test'].map(preprocessor, remove_columns=train_test['test'].column_names)

        output_dir = f'./results-{dataset_name}-{model_name.replace("/", "-")}-full'

        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            learning_rate=5e-5,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=100,
            save_total_limit=1,
            save_steps=500,
            predict_with_generate=True,
            evaluation_strategy='epoch',
            report_to='none'
        )

        # Define compute_metrics inside the loop to access tokenizer
        def compute_metrics(eval_pred):
            preds, labels = eval_pred
            preds = np.argmax(preds, axis=-1)
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            correct = sum(pred.strip() == label.strip() for pred, label in zip(decoded_preds, decoded_labels))
            return {'accuracy': correct / len(decoded_preds)}

        # Scenario 1: Normal loss
        print(f'Fine-tuning with normal loss for {model_name} on {dataset_name}')
        trainer_normal = PrivacyAwareTrainer(
            penalty=False,
            model=model_init(),
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            tokenizer=tokenizer,
            data_collator=custom_data_collator,
            compute_metrics=compute_metrics,
        )
        trainer_normal.train()
        trainer_normal.save_model(os.path.join(output_dir, 'normal-loss-final'))

        eval_normal = trainer_normal.evaluate()
        print(f'Accuracy (normal loss): {eval_normal["eval_accuracy"]:.4f}')

        # Scenario 2: Penalized loss
        print(f'Fine-tuning with penalty for {model_name} on {dataset_name}')
        trainer_penalized = PrivacyAwareTrainer(
            penalty=True,
            model=model_init(),  # fresh model
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            tokenizer=tokenizer,
            data_collator=custom_data_collator,
            compute_metrics=compute_metrics,
        )
        trainer_penalized.train()
        trainer_penalized.save_model(os.path.join(output_dir, 'penalized-loss-final'))

        eval_penalized = trainer_penalized.evaluate()
        print(f'Accuracy (penalized loss): {eval_penalized["eval_accuracy"]:.4f}')
