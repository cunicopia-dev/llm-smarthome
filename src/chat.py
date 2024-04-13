import ollama
import random
import re
import subprocess

def generate_model_input(question):
    question = question.lower()
    return question

def generate_base_response(question, history, model='mixtral'):
    #system_prompt = "Provide a concise and relevant response to the user's query. Your name is Plex."
    system_prompt = """
    You are Plex, an AI assistant created by Anthropic to be helpful, harmless, and honest. Your purpose is to provide accurate, relevant, and comprehensive responses to the user's questions, while maintaining a friendly, respectful, and professional tone. Focus on addressing the user's specific needs and keep the conversation on-topic. 
    If you need to break down complex topics, use clear explanations, analogies, or examples to ensure the user's understanding. Rely on your vast knowledge to provide insightful and valuable information. Always prioritize the user's well-being and aim to have a positive impact through your interactions.
    """
    history.append({'role': 'system', 'content': system_prompt})
    history.append({'role': 'user', 'content': question})
    stream = ollama.chat(
        model=model,
        messages=history,
        stream=True,
    )
    print('Base Assistant: ', end='', flush=True)
    response = ""
    for chunk in stream:
        if 'message' in chunk:
            content = chunk['message']['content']
            response += content
            print(content, end='', flush=True)
    print('\n')
    history.append({'role': 'assistant', 'content': response})
    return response

def main():
    print("Welcome to the conversational AI!")
    print("Type 'quit' to exit the conversation.\n")
    model_list = {1: 'mixtral', 
    2: 'mistral:7b-instruct-v0.2-fp16', 
    3: 'dolphin-mixtral'}

    print("Select a model to use:")
    for key, value in model_list.items():
        print(f"{key}: {value}")
    model_choice = input("Enter the number of the model you want to use: ")

    if model_choice.isdigit():
        model_choice = int(model_choice)
        if model_choice in model_list:
            model = model_list[model_choice]
        else:
            print("Invalid model choice. Using default model 'mixtral'.")
            model = 'mixtral'

    conversation_history = []
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        base_response = generate_base_response(user_input, conversation_history, model=model)

if __name__ == '__main__':
    main()