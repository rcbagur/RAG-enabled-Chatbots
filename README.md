# Interactive Chatbot Using OpenAI

This project demonstrates how to create an interactive chatbot that leverages OpenAI's embeddings and GPT models to answer questions based on a provided context. The chatbot finds the most relevant context from a pre-processed dataset and generates answers to user-input questions.

## Getting Started

These instructions will guide you through the setup and running of the interactive chatbot on your local machine.

### Prerequisites

- Python 3.6+
- An OpenAI API key
- The dataset file in CSV format containing your pre-processed embeddings or a website url to scrape data from

### Installation

1. **Clone the Repository**

   Clone this repository to your local machine using:
   ```git clone https://github.com/rcbagur/RAG-enabled-Chatbots.git```

2. **Install Required Python Packages**

Navigate to the cloned repository's directory and install the required packages using:
```pip install -r requirements.txt```

3. **Set Up Environment Variables**

You need to set the following environment variables:
- `OPENAI_API_KEY`: Your OpenAI API key.

On Unix-based systems, you can set these variables in your shell:
```export OPENAI_API_KEY='your_openai_api_key_here'```

On Windows, you can set these variables in Command Prompt (CMD) as follows:
```set OPENAI_API_KEY=your_openai_api_key_here```

### Running the script to build a vector database

After setting up the environment variables and installing the required packages, you can web scrapping by running:
```python create_embeddings.py <url>```

### Dataset Format

Your dataset CSV file should contain at least two columns:
- `text`: The text content you want to use for context.
- `embedding`: The embeddings of the text content, stored as lists.

Example:

```csv
text,embedding
"Some text content.",[0.123, 0.456, ...]
```

### Running the Interactive Chat

After setting up the environment variables and installing the required packages, you can start the interactive chat by running:
```python interactive_chat.py```

Follow the on-screen prompts to ask questions. Type 'quit' to exit the chat.










