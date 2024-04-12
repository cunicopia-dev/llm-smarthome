import ollama
import random
import re
import subprocess

def generate_model_input(question):
    question = question.lower()
    return question

def generate_response(question, history, model='mixtral'):
    question = generate_model_input(question)
    history.append({'role': 'user', 'content': question})
    if question.startswith("shell "):  # Check if the command is intended to be a shell command
        command = question[6:]  # Extract the command part
        try:
            output = subprocess.run(command, shell=True, check=True, text=True, stdout=subprocess.PIPE)
            response = output.stdout
            print(response)
        except subprocess.CalledProcessError as e:
            response = f"Error: {e}"
    else:
        stream = ollama.chat(
            model=model,
            messages=history,
            stream=True,
        )
        print('A: ', end='', flush=True)
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

    conversation_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break

        generate_response(user_input, conversation_history)

if __name__ == '__main__':
    main()
