import json
from ollama import chat

def load_prompts(file_path):
    with open(file_path, 'r') as file:
        prompts = json.load(file)
    return prompts

def get_system_prompt(prompts, prompt_id=None):
    if prompt_id and prompt_id in prompts:
        return prompts[prompt_id]['description']
    else:
        return prompts['1']['description']  # Default to base assistant prompt

def save_chat_history(history):
    with open('chat.json', 'w') as file:
        json.dump(history, file, indent=4)

def load_chat_history():
    try:
        with open('chat.json', 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return []

def generate_base_response(question, history, model, assistant_name, system_prompt):
    history.append({'role': 'system', 'content': system_prompt})
    history.append({'role': 'user', 'content': question})
    stream = chat(
        model=model,
        messages=history,
        stream=True,
    )
    print(f'{assistant_name}: ', end='', flush=True)
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
    load_history = input("Do you want to load the previous chat history? (yes/no): ").strip().lower()
    conversation_history = load_chat_history() if load_history == 'yes' else []

    print("Type 'quit' to exit the conversation.\n")
    model_list = {1: 'mixtral', 2: 'mistral:7b-instruct-v0.2-fp16', 3: 'dolphin-mixtral'}
    prompts = load_prompts('./src/prompts/system_prompts.json')

    # Model selection handling
    print("Select a model to use:")
    for key, value in model_list.items():
        print(f"{key}: {value}")
    model_choice = input("Enter the number of the model you want to use: ")
    if model_choice.isdigit() and int(model_choice) in model_list:
        model = model_list[int(model_choice)]
    else:
        model = model_list[2]  # Default to 'mistral:7b-instruct-v0.2-fp16'
        print(f"Invalid model choice. Using default model '{model}'.")

    # System prompt selection
    print("Available System Prompts:")
    for key, value in prompts.items():
        print(f"{key}: {value['one_word_description']} - {value['description']}")
    prompt_id = input("Enter the ID of the system prompt you want to use (leave blank for default): ")
    system_prompt = get_system_prompt(prompts, prompt_id)

    assistant_name = input("Enter the name for the assistant: ")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            save_chat_history(conversation_history)
            print("Goodbye!")
            break
        generate_base_response(user_input, conversation_history, model, assistant_name, system_prompt)

if __name__ == '__main__':
    main()
