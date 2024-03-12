import os
import pandas as pd
import requests
import time
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

# Initialize session for HTTP requests within fetch_and_process_links if needed
session = requests.Session()
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}
session.headers.update(headers)

# Assuming tiktoken and client (OpenAI) are initialized outside this script
def fetch_page_content(url):
    """Fetches the content of a given URL with a delay to avoid hammering the server."""
    time.sleep(1)  # Respectful delay
    response = session.get(url)
    if response.status_code == 200:
        return response.text
    else:
        print(f"Failed to retrieve {url}")
        return None

def save_content_from_url(url):
    """Extracts and saves either the full text or paragraphs from a URL based on content_type, keeping the biggest file in case of duplicates."""
    content = fetch_page_content(url)
    if content:
        soup = BeautifulSoup(content, 'html.parser')
        title = soup.find('title').text.strip().replace('/', '-').replace('\\', '-')
        filename = os.path.join(f"./database/{title}.txt")
        paragraphs = soup.find_all('p')
        text_content = '\n'.join([p.get_text() for p in paragraphs])
        if text_content:
            # Check if the file exists and compare sizes
            if os.path.exists(filename):
                existing_file_size = os.path.getsize(filename)
                new_file_size = len(text_content.encode('utf-8'))
                if new_file_size <= existing_file_size:
                    print(f"Existing file {filename} is larger or equal in size; skipping overwrite.")
                    return

            # Save or overwrite the file with the new content
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(text_content)
            print(f"Saved {filename}")


def fetch_and_process_links(url, BASE_URL, SEARCH_PATH):
    """Fetches a page, extracts all relevant links, and returns them."""
    print(f">> {SEARCH_PATH}" + url)
    content = fetch_page_content(url)
    if content:
        soup = BeautifulSoup(content, 'html.parser')
        links = [BASE_URL + a['href'] for a in soup.find_all('a', href=True) if url.removeprefix(BASE_URL) in a['href'] and not a['href'].startswith("https")]
        return links

def save_pages(links, content_type='paragraphs'):
    """Saves pages from a list of links using multiple threads."""
    with ThreadPoolExecutor(max_workers=4) as executor:
      executor.map(lambda url: save_content_from_url(url), links)

# Example usage of splitting and processing text for embeddings
def split_text_into_chunks(text, tokenizer, max_tokens):
    """Splits text into chunks, ensuring each chunk does not exceed max_tokens."""
    tokens = tokenizer.encode(text)
    chunks = []
    current_chunk = []
    tokens_in_chunk = 0

    for token in tokens:
        if tokens_in_chunk + 1 > max_tokens:  # Start a new chunk if adding a token exceeds max_tokens
            chunks.append(tokenizer.decode(current_chunk))
            current_chunk = []
            tokens_in_chunk = 0
        current_chunk.append(token)
        tokens_in_chunk += 1

    if current_chunk:  # Add the last chunk if it's not empty
        chunks.append(tokenizer.decode(current_chunk))

    return chunks

def process_dataframe(df, tokenizer, max_tokens):
    """Process a dataframe to tokenize and split texts, ensuring each chunk is within max_tokens."""
    split_texts = []
    for _, row in df.iterrows():
        text = row['text']
        if pd.isnull(text):
            continue  # Skip null texts
        text_chunks = split_text_into_chunks(text, tokenizer, max_tokens)
        split_texts.extend(text_chunks)

    split_df = pd.DataFrame(split_texts, columns=['text'])
    split_df['n_tokens'] = split_df['text'].apply(lambda x: len(tokenizer.encode(x)))
    return split_df