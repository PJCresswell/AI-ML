from datasets import load_dataset

# raw_datasets = load_dataset("openai/gsm8k", "main")


# Microsoft Research Paraphrase Corpus dataset
# The dataset consists of 5,801 pairs of sentences, with a label indicating if they are paraphrases or not (i.e., if both sentences mean the same thing)
raw_datasets = load_dataset("glue", "mrpc")
print(raw_datasets)

# raw_train_dataset = raw_datasets["train"]
# print(raw_train_dataset[0])
# print(raw_train_dataset.features)

# Tokenising one example

from transformers import AutoTokenizer
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
example = raw_datasets["train"][15]

# First individually
tokenized_sentences_1 = tokenizer(example["sentence1"])
print(tokenized_sentences_1)
tokenized_sentences_2 = tokenizer(example["sentence2"])
print(tokenized_sentences_2)

# Now together
inputs = tokenizer(example["sentence1"], example["sentence2"])
print(inputs)
decoded = tokenizer.convert_ids_to_tokens(inputs["input_ids"])
print(decoded)

# Tokenising the whole dataset using the map function. Applies the function to each value
# Adds new features to the dataset

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
print(tokenized_datasets)

# Final step - apply dynamic padding
# Dynamic means that only pad up to the longest example in each batch - NOT the longest example in the whole dataset

from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Grab some samples from our training set to batch together. Just the first 8
samples = tokenized_datasets["train"][:8]
# Removes the three columns not needed and which contain strings
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
# Have a look at the length of each entry in the batch
print([len(x) for x in samples["input_ids"]])

# Call the data collator with padding
batch = data_collator(samples)
# Have a look at the final shape - all now 67 which was the longest in the batch
print({k: v.shape for k, v in batch.items()})
