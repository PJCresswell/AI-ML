# Example using a BERT model for sentiment analysis

# Step 1 : Tokenisation
# Maps each token to an ID in the input dictionary of the model
# Can see how handled both sentences. Added padding for tee second as shorter and needs to be the same size when called
from transformers import AutoTokenizer
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")

print(inputs)

# Step 2 : Going through the model body. Encodes the input vectors
from transformers import AutoModel
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(checkpoint)
outputs = model(**inputs)
# Is the hidden state after encoding
# 2 indicates 2 examples
# 16 indicates the sequence length
# 768 is the vector dimension of each input
print(outputs.last_hidden_state.shape)

# Now we apply a model head to perform a task
from transformers import AutoModelForSequenceClassification
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
# We're doing sentiment analysis classification
print(model.config.id2label)
# Calculate the output
outputs = model(**inputs)
# Result is 2 by 2 - raw scores for each of the two examples against negative and positive classifications
print(outputs.logits.shape)
print(outputs.logits)
# Finally we turn into an actual probability score
import torch
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)
# First sentence: NEGATIVE: 0.0402, POSITIVE: 0.9598
# Second sentence: NEGATIVE: 0.9995, POSITIVE: 0.0005