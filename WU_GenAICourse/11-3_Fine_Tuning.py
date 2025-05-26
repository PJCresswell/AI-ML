import openai
from openai import OpenAI
import requests

client = OpenAI()

'''
fname = 'sarcastic.jsonl'
url = 'https://data.heatonresearch.com/data/t81-559/finetune/' + fname
r = requests.get(url)
print(r.status_code)
print(r.text)
with open("sarcastic.jsonl", "wb") as f:
  f.write(r.content)

obj = client.files.create(
  file=open("sarcastic.jsonl", "rb"),
  purpose="fine-tune"
)
print(obj.id)

import time

# Start the fine-tuning job
train = client.fine_tuning.jobs.create(
    training_file='file-W0yn1Ye8DqXbeBC6QxVIAzkA',
    model="gpt-4o-mini-2024-07-18"
)

done = False

# Initialize a set to store processed event IDs
processed_event_ids = set()

while not done:
    # Retrieve the latest status of the fine-tuning job
    status = client.fine_tuning.jobs.retrieve(train.id)
    print(f"Job status: {status.status}")

    # Fetch all events related to the fine-tuning job
    events = client.fine_tuning.jobs.list_events(fine_tuning_job_id=train.id)

    # Collect new events that haven't been processed yet
    new_events = []
    for event in events:
        if event.id not in processed_event_ids:
            new_events.append(event)
            processed_event_ids.add(event.id)

    # Sort the new events in chronological order
    new_events.sort(key=lambda e: e.created_at)

    # Display the new events in order
    for event in new_events:
        print(f"{event.created_at}: {event.message}")

    if status.status == "succeeded":
        done = True
        print("Done!")
    elif status.status == "failed":
        done = True
        print("Failed!")
    else:
        print("Waiting for updates...")
        time.sleep(20)  # Sleep for 20 seconds

model_id = status.fine_tuned_model
print(f"Trained model id: {model_id}")
'''

completion = client.chat.completions.create(
  model='ft:gpt-4o-mini-2024-07-18:personal::AD7rZq7T',
  messages=[
    {"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."},
    {"role": "user", "content": "What is the capital of the UK?"}
  ]
)
print(completion.choices[0].message)