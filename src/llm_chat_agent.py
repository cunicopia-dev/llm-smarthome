import ollama
import random
import re
import subprocess

def generate_model_input(question):
    question = question.lower()
    return question

def generate_base_response(question, history, model='mixtral'):
    system_prompt = "Provide a concise and relevant response to the user's query. Your name is Plex."
    # system_prompt = """
    # You are Claude, an AI assistant created by Anthropic to be helpful, harmless, and honest. Your purpose is to provide accurate, relevant, and comprehensive responses to the user's questions, while maintaining a friendly, respectful, and professional tone. Focus on addressing the user's specific needs and keep the conversation on-topic. 
    # If you need to break down complex topics, use clear explanations, analogies, or examples to ensure the user's understanding. Rely on your vast knowledge to provide insightful and valuable information. Always prioritize the user's well-being and aim to have a positive impact through your interactions.
    # """
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

def generate_refiner_response(base_response, history, model='mixtral'):

    system_prompt = """
    YOU SHOULD NEVER OUTPUT MORE THAN A FEW SENTENCES.
    As a refiner assistant, your role is to provide concise, actionable feedback to improve the base assistant's responses, focusing on enhancing clarity, relevance, and overall helpfulness. Analyze the response and identify specific areas for improvement, such as:
    - Removing unnecessary or redundant information
    - Improving the organization and structure of the response
    - Ensuring the response directly addresses the user's question
    - Suggesting ways to make the explanation clearer or easier to understand
    - Recommending additional relevant information or examples to include
    Provide your feedback in a clear, bullet-pointed format, and prioritize the most impactful changes. Avoid engaging in conversations with the base assistant or making subjective comments.
    """
    history_subset = history[-3:]  # Use only the last 3 messages for the refiner
    history_subset.append({'role': 'system', 'content': system_prompt})
    history_subset.append({'role': 'user', 'content': base_response})
    stream = ollama.chat(
        model=model,
        messages=history_subset,
        stream=True,

    )
    print('Refiner Assistant: ', end='', flush=True)
    response = ""
    for chunk in stream:
        if 'message' in chunk:
            content = chunk['message']['content']
            response += content
            print(content, end='', flush=True)
    print('\n')
    history.append({'role': 'assistant', 'content': response})
    return response

def generate_final_response(question, base_response, refiner_response, history, model='mixtral'):
    system_prompt = """You are an agent meant to take the data from the refiner, and provide a final message back to the user. The user doesn't know you are talking with a refiner agent. Ensure you focus on providing a concise and meaningful answer to the user, never mentioning the refiner agents. Enhance the response for clarity and conciseness, focusing on directly answering the user's question with minimal additional information."""

    # system_prompt = """
    # As Claude, your primary goal is to provide a concise, relevant, and helpful response to the user's question or request. Carefully review the refiner assistant's feedback and incorporate their suggestions to improve your response, without mentioning the refiner's involvement.

    # When crafting your refined response:
    # - Greet the user warmly and acknowledge their question or request
    # - Directly address the user's specific needs or inquiries in a clear and concise manner
    # - Provide well-structured information that is easy for the user to understand
    # - Use examples, analogies, or explanations only when necessary to clarify complex topics
    # - If relevant, offer additional insights or resources that add value to the user's query
    # - Maintain a friendly and conversational tone, while keeping the response focused and on-topic
    # - Conclude by asking if there's anything else you can assist the user with

    # Remember, your aim is to deliver a high-quality, user-centered response that demonstrates your expertise and commitment to being helpful, honest, and concise. Keep the conversation focused on the user's needs and strive to provide a valuable interaction in as few words as possible. Make the user feel heard and supported, without overwhelming them with unnecessary details.    """
    history.append({'role': 'system', 'content': system_prompt})
    history.append({'role': 'user', 'content': question})
    history.append({'role': 'assistant', 'content': base_response})
    history.append({'role': 'user', 'content': refiner_response})
    stream = ollama.chat(
        model=model,
        messages=history,
        stream=True,
    )
    print('Final Response: ', end='', flush=True)
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
        refiner_response = generate_refiner_response(base_response, conversation_history, model=model)
        final_response = generate_final_response(user_input, base_response, refiner_response, conversation_history, model=model)

if __name__ == '__main__':
    main()