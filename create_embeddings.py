import os
import pandas as pd
import argparse
from helpers.text_processing import process_dataframe, save_pages, fetch_and_process_links
import tiktoken
from openai import OpenAI

# Setup OpenAI client securely
api_key = os.getenv("OPENAI_API_KEY")  # Set your OpenAI API key as an environment variable
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")
client = OpenAI(
  api_key = os.getenv("OPENAI_API_KEY")  # Set your OpenAI API key as an environment variable
)

def create_embeddings(url, model='text-embedding-ada-002'):
    # URL Configuration
    BASE_URL= "/".join(url.split('/')[:-1])
    SEARCH_PATH= "/"+"".join(url.split('/')[-1:])

    # Example usage to fetch, save, and process pages
    visited_pages = set()
    to_visit_pages = set([BASE_URL + SEARCH_PATH])

    while to_visit_pages:
        current_page = to_visit_pages.pop()
        visited_pages.add(current_page)

        new_pages = fetch_and_process_links(current_page, BASE_URL, SEARCH_PATH)
        if new_pages:
            new_pages = set(new_pages) - visited_pages  # Remove already visited or queued pages
            to_visit_pages.update(new_pages)  # Add new pages to visit

        if len(visited_pages) > 100:
            print("ERROR: Maximum of 100 links achieved")
            break

    save_pages(visited_pages)

    # Load and process saved text files into DataFrame
    df = pd.DataFrame([(f.split(".txt")[0], open(os.path.join('./database', f), 'r', encoding='utf-8').read()) 
                    for f in os.listdir('./database') if f.endswith('.txt')],
                    columns=['fname', 'text'])

    # Generate embeddings
    tokenizer = tiktoken.get_encoding("cl100k_base")
    split_df = process_dataframe(df, tokenizer, max_tokens=500)
    split_df['embedding'] = split_df['text'].apply(lambda x: client.embeddings.create(input=x, model=model).data[0].embedding)

    # Save DataFrame with embeddings
    split_df.to_csv('./database/processed_topics_embeddings.csv')
    print("Completed embeddings database creation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate text embeddings for a given URL.')
    parser.add_argument('url', type=str, help='The URL to process for text embeddings.')
    args = parser.parse_args()

    create_embeddings(args.url)
