# Get the SciQ dataset from HuggingFace
from datasets import load_dataset
dataset = load_dataset("sciq", split="train")

# Filter the dataset to only include questions with a support
dataset = dataset.filter(lambda x: x["support"] != "")
print("Number of questions with support: ", len(dataset))

import chromadb

client = chromadb.Client()
collection = client.create_collection("sciq_supports")
collection.add(
    ids=[str(i) for i in range(0, 100)],  # IDs are just strings
    documents=dataset["support"][:100],
    metadatas=[{"type": "support"} for _ in range(0, 100)
    ],
)
results = collection.query(
    query_texts=dataset["question"][:10],
    n_results=1)

# Print the question and the corresponding support
for i, q in enumerate(dataset['question'][:10]):
    print(f"Question: {q}")
    print(f"Retrieved support: {results['documents'][i][0]}")
    print()