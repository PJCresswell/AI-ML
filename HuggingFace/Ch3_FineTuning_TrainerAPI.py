# First the data preparation as covered before

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Now the new bit

from transformers import TrainingArguments
# Training args class contains the hyper params plus defines where the model will be stored
training_args = TrainingArguments("test-trainer")

from transformers import AutoModelForSequenceClassification
# Define our model. Sequence classification with two labels
# Gives a warning that BERT not trained for that. That's what we're doing now !
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

# Now train the model
from transformers import Trainer
trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)
trainer.train()

# Finally, see how well the model is performing
predictions = trainer.predict(tokenized_datasets["validation"])
# The output of the predict() method is another named tuple with three fields: predictions, label_ids, and metrics
# The metrics field will just contain the loss on the dataset passed, as well as some time metrics (how long it took to predict, in total and on average)
# Predictions is a two-dimensional array with shape 408 x 2 (408 being the number of elements in the dataset we used)
# Those are the logits for each element of the dataset we passed to predict()
print(predictions.predictions.shape, predictions.label_ids.shape)
# To transform them into predictions that we can compare to our labels, we need to take the index with the maximum value on the second axis:
import numpy as np
preds = np.argmax(predictions.predictions, axis=-1)

# We can now compare those preds to the labels
# To build our compute_metric() function, we will rely on the metrics from the Evaluate library
# We can load the metrics associated with the MRPC dataset as easily as we loaded the dataset, this time with the evaluate.load() function.
# The object returned has a compute() method we can use to do the metric calculation
import evaluate
metric = evaluate.load("glue", "mrpc")
print(metric.compute(predictions=preds, references=predictions.label_ids))
