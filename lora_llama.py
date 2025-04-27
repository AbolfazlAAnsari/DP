from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
import torch
from torch import nn
from typing import Any, Dict, List
from peft import get_peft_model, LoraConfig, TaskType
from transformers import default_data_collator
import numpy as np
import os

# Model to fine-tune (LLaMA variant)
model_name = 'meta-llama/Llama-2-7b-chat-hf'

# Dataset
datasets_info = [
    'pii-masking-400k',
    'open-pii-masking-500k-ai4privacy'
]

# Preprocessing function
def get_preprocessor(tokenizer):
    def preprocess(example):
        input_text = 'Mask sensitive information: ' + example['source_text']
        target_text = example['masked_text']

        input_enc = tokenizer(input_text, truncation=True, padding='max_length', max_length=512, return_offsets_mapping=True)
        offset_mapping = input_enc.pop('offset_mapping', None)
        labels_enc = tokenizer(target_text, truncation=True, padding='max_length', max_length=512)

        labels = labels_enc['input_ids']
        labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]

        span_mask = np.zeros(512, dtype=int)
        if offset_mapping is not None:
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

# Metric
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    correct = sum(pred.strip() == label.strip() for pred, label in zip(decoded_preds, decoded_labels))
    accuracy = correct / len(decoded_preds)
    return {'accuracy': accuracy}

# Custom Trainer with span penalty
class PrivacyAwareTrainer(Trainer):
    def __init__(self, penalty=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.penalty = penalty

    def compute_loss(self, model, inputs, return_outputs=False):
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
            penalty_term = (penalty_weight * shift_span.float()).sum() / active_tokens.sum()
            final_loss = final_loss + penalty_term

        return (final_loss, outputs) if return_outputs else final_loss

# LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # typical for LLaMA
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# Main loop
for dataset_name in datasets_info:
    print(f'\nLoading dataset: {dataset_name}')
    dataset = load_dataset(f'ai4privacy/{dataset_name}')
    english_dataset = dataset['train'].filter(lambda x: x['language'] == 'en')
    train_test = english_dataset.train_test_split(test_size=0.1, seed=42)

    print(f'\nLoading model: {model_name}')
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="right")
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16, device_map='auto')

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    preprocessor = get_preprocessor(tokenizer)
    tokenized_train = train_test['train'].map(preprocessor, remove_columns=train_test['train'].column_names, batched=True)
    tokenized_test = train_test['test'].map(preprocessor, remove_columns=train_test['test'].column_names, batched=True)

    output_dir = f'./results-{dataset_name}-llama2-lora'

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=2e-4,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=50,
        save_total_limit=2,
        predict_with_generate=True,
        bf16=True,
        report_to='none'
    )

    print('Training without penalty loss first...')
    trainer = PrivacyAwareTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=custom_data_collator,
        compute_metrics=compute_metrics,
        penalty=False
    )

    trainer.train()
    trainer.save_model(os.path.join(output_dir, 'normal-loss-final'))

    eval_results = trainer.evaluate()
    print(f'Accuracy without penalty: {eval_results["eval_accuracy"]:.4f}')

    print('Training with penalty loss...')
    trainer.penalty = True
    trainer.train()
    trainer.save_model(os.path.join(output_dir, 'penalized-loss-final'))

    eval_results_penalized = trainer.evaluate()
    print(f'Accuracy with penalty: {eval_results_penalized["eval_accuracy"]:.4f}')
