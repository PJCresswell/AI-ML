# Hard-coded examples of different tasks using the pipeline function
# Where no model is specified, used a default model
# Needs PyTorch to run as well as the Transformer library

from transformers import pipeline

classifier = pipeline("sentiment-analysis")
# result = classifier(
#    ["I've been waiting for a HuggingFace course my whole life.", "I hate this so much!"]
# )

classifier = pipeline("zero-shot-classification")
# result = classifier(
#    "This is a course about the Transformers library",
#    candidate_labels=["education", "politics", "business"],
# )

# Now we use a specific model
generator = pipeline("text-generation", model="distilgpt2")
# result = generator(
#    "In this course, we will teach you how to",
#    max_length=30,
#    num_return_sequences=2,
# )

unmasker = pipeline("fill-mask")
# result = unmasker("This course will teach you all about <mask> models.", top_k=2)

ner = pipeline("ner", grouped_entities=True)
# result = ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")

question_answerer = pipeline("question-answering")
# result = question_answerer(
#    question="Where do I work?",
#    context="My name is Sylvain and I work at Hugging Face in Brooklyn",
# )

summarizer = pipeline("summarization")
# result = summarizer(
"""
    America has changed dramatically during recent years. Not only has the number of 
    graduates in traditional engineering disciplines such as mechanical, civil, 
    electrical, chemical, and aeronautical engineering declined, but in most of 
    the premier American universities engineering curricula now concentrate on 
    and encourage largely the study of engineering science. As a result, there 
    are declining offerings in engineering subjects dealing with infrastructure, 
    the environment, and related issues, and greater concentration on high 
    technology subjects, largely supporting increasingly complex scientific 
    developments. While the latter is important, it should not be at the expense 
    of more traditional engineering.

    Rapidly developing economies such as China and India, as well as other 
    industrial countries in Europe and Asia, continue to encourage and advance 
    the teaching of engineering. Both China and India, respectively, graduate 
    six and eight times as many traditional engineers as does the United States. 
    Other industrial countries at minimum maintain their output, while America 
    suffers an increasingly serious decline in the number of engineering graduates 
    and a lack of well-educated engineers.
"""
# )

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
# result = translator("Ce cours est produit par Hugging Face.")

# OMG - great example of bias. The BERT model trying to fill in the blanks

unmasker = pipeline("fill-mask", model="bert-base-uncased")

result = unmasker("This man works as a [MASK].")
print([r["token_str"] for r in result])

result = unmasker("This woman works as a [MASK].")
print([r["token_str"] for r in result])

