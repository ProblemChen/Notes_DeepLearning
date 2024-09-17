from transformers import load_dataset
from transformers import AutoTokenizer,DataCollatorWithPadding
raw_datasets =load_dataset("glue","mrpc")
checkpoint= "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
	return tokenizer(example["train"],example["validation"],truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function,batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

from transformers import TrainingArguments
training_args =TrainingArguments("test-trainer")

from transformers import AutoModelForClassification
model = AutoModelForClassification.from_pretrained(checkpoint,num_labels=2)

from transformers import Trainer 
trainer = Trainer(model,training_args,train_dataset=tokenized_datasets["train"],eval_dataset=tokenized_datasets["validation"],data_collator=data_collator,tokenzier=tokenizer,)
trainer.train()

predictions = trainer.predict(tokenized_datasets["validation"])
print(predictions.predictions.shape,predictions.labels.shape)

import numpy as np 
preds = np.argmax(predictions.predictions,axis=-1)

import evaluate 
metric = evaluate.load("glue","mrpc")
metric.compute(predictions=preds,references=predictions.label_ids)

def compute_metrics(eval_preds):
	metric = evaluate.load("glue","mrpc")
	logits,labels = eval_preds
	predictions = np.argmax(logits,axis=-1)
	return metric.compute(predictions,references=labels)
	


