from langchain_openai import OpenAIEmbeddings

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
l1 = embeddings_model.embed_query("dog")
l2 = embeddings_model.embed_query("Something that is a bit longer than a word.")

print(type(l1))
print(len(l1))
print(len(l2))
print(l1[:10])

import numpy as np

# Convert lists to numpy arrays
vec1 = np.array(l1)
vec2 = np.array(l2)

# Calculate the dot product
dot_product = np.dot(vec1, vec2)

# Calculate the magnitudes (L2 norms)
magnitude_vec1 = np.linalg.norm(vec1)
magnitude_vec2 = np.linalg.norm(vec2)

# Calculate cosine similarity
cosine_similarity = dot_product / (magnitude_vec1 * magnitude_vec2)

print(f"Cosine similarity: {cosine_similarity}")
print(magnitude_vec1 * magnitude_vec2)
print(np.dot(vec1, vec2))

def compare_str(embeddings_model, text1, text2):
    """
    This function returns the dot product of embeddings for two given text strings.

    Parameters:
    embeddings_model: The embeddings model to use for generating embeddings.
    text1 (str): The first text string.
    text2 (str): The second text string.

    Returns:
    float: The dot product of the embeddings for text1 and text2.
    """
    # Get the embeddings for the two text strings
    embedding1 = embeddings_model.embed_query(text1)
    embedding2 = embeddings_model.embed_query(text2)

    # Convert embeddings to numpy arrays for dot product calculation
    embedding1_array = np.array(embedding1)
    embedding2_array = np.array(embedding2)

    # Calculate and return the dot product
    dot_product = np.dot(embedding1_array, embedding2_array)
    return dot_product

compare_str(embeddings_model,
            "A machine that helps people to cut grass.",
            "Device with blades to cut plants under it.")

compare_str(embeddings_model,
            "A machine that helps people to cut grass.",
            "Vehicle that flys through the air.")