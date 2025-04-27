from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import Dataset
from evaluate import load as load_metric
import torch
from torch import nn
from typing import Any, Dict, List
from transformers import default_data_collator
import numpy as np
import pandas as pd
import os

# Models to fine-tune
model_names = [
    't5-small',
    't5-base',
    'google/flan-t5-base',
]

def get_preprocessor(tokenizer):
    def preprocess(example):
        input_text = 'Mask sensitive information: ' + example['original_text']
        target_text = example['target_text']

        input_enc = tokenizer(input_text, truncation=True, padding='max_length', max_length=128, return_offsets_mapping=True)
        input_ids = input_enc['input_ids']
        
        target_enc = tokenizer(target_text, truncation=True, padding='max_length', max_length=128)
        labels = target_enc['input_ids']
        labels = labels[:128] + [-100] * (128 - len(labels))

        return {
            'input_ids': input_enc['input_ids'],
            'attention_mask': input_enc['attention_mask'],
            'labels': labels
        }
    return preprocess

def custom_data_collator(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    batch = default_data_collator(features)
    return batch

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    try:
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    except Exception as e:
        print(f"Error decoding predictions: {e}")
        # Some invalid tokens
        return {'accuracy': 0}  

    decoded_preds = [pred.replace('<pad>', '').replace('<unk>', '') for pred in decoded_preds]
    decoded_labels = [label.replace('<pad>', '').replace('<unk>', '') for label in decoded_labels]
    
    correct = sum([pred.strip() == label.strip() for pred, label in zip(decoded_preds, decoded_labels)])
    accuracy = correct / len(decoded_preds)
    return {'accuracy': accuracy}


class PrivacyAwareTrainer(Seq2SeqTrainer):
    def __init__(self, penalty=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.penalty = penalty

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs['labels']
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=labels)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(shift_labels.size())

        active_tokens = (shift_labels != -100).float()
        final_loss = (loss * active_tokens).sum() / active_tokens.sum()

        if self.penalty:
            penalty_weight = 5.0
            penalty_term = 0 
            final_loss = final_loss + penalty_term

        return (final_loss, outputs) if return_outputs else final_loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

# === Local dataset - PHI ===
print('Loading local dataset...')
csv_path = './data/Contextual_PII_Masking_Dataset.csv'
df = pd.read_csv(csv_path)

dataset = Dataset.from_pandas(df)

dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset['train']
test_dataset = dataset['test']

for model_name in model_names:
    print(f'\nFine-tuning model: {model_name} on local dataset')
    
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
    tokenized_train = train_dataset.map(preprocessor, remove_columns=train_dataset.column_names)
    tokenized_test = test_dataset.map(preprocessor, remove_columns=test_dataset.column_names)

    output_dir = f'./results-local-{model_name.replace("/", "-")}'

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        learning_rate=5e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=10,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=100,
        save_total_limit=1,
        save_steps=500,
        predict_with_generate=True,
        evaluation_strategy='epoch',
        report_to='none'
    )

    # Fine-tuning with normal loss
    print(f'Fine-tuning with normal loss for {model_name}')
    trainer_normal_loss = PrivacyAwareTrainer(
        penalty=False,
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=custom_data_collator,
        compute_metrics=compute_metrics
    )
    
    # trainer_normal_loss.train()
    # trainer_normal_loss.save_model(os.path.join(output_dir, 'normal-loss-final'))
    
    #eval_results_normal = trainer_normal_loss.evaluate()
    #print(f'Accuracy with normal loss: {eval_results_normal["eval_accuracy"]:.4f}')

    # Fine-tuning with penalized loss (no span_labels used)
    print(f'Fine-tuning with penalty on loss for {model_name}')
    trainer_penalized_loss = PrivacyAwareTrainer(
        penalty=True,
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
    
    #eval_results_penalized = trainer_penalized_loss.evaluate()
    #print(f'Accuracy with penalized loss: {eval_results_penalized["eval_accuracy"]:.4f}')
