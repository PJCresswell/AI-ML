# Example where we know the type of model that we want to use
# Not using the AutoModel class

from transformers import BertConfig, BertModel
# Building the config
config = BertConfig()
print(config)
# Building the model from the config
# Model is randomly initialized. Would output pure rubbish !
model = BertModel(config)

# Example where loaded from a checkpoint, saved locally and then used
from transformers import BertModel
model = BertModel.from_pretrained("bert-base-cased")
# model.save_pretrained("Models")
sequences = ["Hello!", "Cool.", "Nice!"]
encoded_sequences = [
    [101, 7592, 999, 102],
    [101, 4658, 1012, 102],
    [101, 3835, 999, 102],
]
import torch
model_inputs = torch.tensor(encoded_sequences)
output = model(model_inputs)
print(output)