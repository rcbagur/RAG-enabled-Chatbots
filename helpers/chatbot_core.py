import os
import pandas as pd
import numpy as np
from ast import literal_eval
from scipy.spatial import distance
from openai import OpenAI

# Setup OpenAI client securely
api_key = os.getenv("OPENAI_API_KEY")  # Set your OpenAI API key as an environment variable
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")
client = OpenAI(
  api_key = os.getenv("OPENAI_API_KEY")  # Set your OpenAI API key as an environment variable
)

def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    response = client.embeddings.create(input = [text], model=model)
    return response.data[0].embedding

def calculate_distances(query_embedding, embeddings, distance_metric="cosine"):
    distance_func = getattr(distance, distance_metric)
    return np.array([distance_func(query_embedding, embedding) for embedding in embeddings])

def create_context(question, df, max_len=1800):
    q_embedding = get_embedding(question)
    df['distances'] = calculate_distances(q_embedding, df['embedding'].tolist())
    sorted_df = df.sort_values('distances')
    texts, total_len = [], 0
    for _, row in sorted_df.iterrows():
        text_len = len(row['text'].split())
        if total_len + text_len > max_len:
            break
        texts.append(row['text'])
        total_len += text_len
    return "\n\n###\n\n".join(texts)

def answer_question(question, df, max_tokens=150, history=None, model="gpt-3.5-turbo-instruct"):
    context = create_context(question, df)
    prompt = f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext: {context}\n\n---\n\n"
    if history:
      prompt += f"Conversation history: {history}\n\n"
    prompt += f"Question: {question}\nAnswer:"
    try:
        # Create a completions using the question and context
        response = client.completions.create(
            model=model,
            prompt=prompt,
            temperature=0,
            max_tokens=max_tokens,
            stop=["\n"]
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print(e)
        return ""

def load_and_prepare_df(filepath):
    df = pd.read_csv(filepath, index_col=0)
    df['embedding'] = df['embedding'].apply(lambda x: np.array(literal_eval(x)))
    return df
