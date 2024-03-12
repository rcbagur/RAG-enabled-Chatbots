import os
from helpers.chatbot_core import load_and_prepare_df, answer_question

def interactive_chat():
    print("Interactive Question Answering System. Type 'quit' to exit.")
    embeddings_file_path = "./database/processed_topics_embeddings.csv"
    df = load_and_prepare_df(embeddings_file_path)
    print(f"\n> Assistant: Hello! How can I assist you?")
    history = ""
    while True:
        user_input = input("\n> Human: ")
        if user_input.lower() == 'quit':
            print("Assistant: Goodbye!")
            break
        response = answer_question(user_input, df, history=history)
        print(f"\n> Assistant: {response}")

if __name__ == "__main__":
    interactive_chat()
