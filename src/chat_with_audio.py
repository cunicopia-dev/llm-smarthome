import json
import subprocess
import threading
import sounddevice as sd
from pocketsphinx import LiveSpeech
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

def generate_base_response(question, history, model, assistant_name, system_prompt, language, add_system_prompt=False):
    if add_system_prompt:
        # Add system prompt only on the first interaction of each session
        history.append({'role': 'system', 'content': system_prompt})
    history.append({'role': 'user', 'content': question})
    stream = chat(
        model=model,
        messages=history,
        stream=True,
    )
    response = ""
    for chunk in stream:
        if 'message' in chunk:
            content = chunk['message']['content']
            response += content
            print({content}', end='', flush=True)
    print('\n')
    history.append({'role': 'assistant', 'content': response})
    text_to_speech_pico(response, language)
    save_chat_history(history)  # Save chat history after each interaction
    return response

def text_to_speech_pico(text, language="en-US"):
    output = "speech.wav"
    command = ["pico2wave", "--wave", output, "--lang", language, text]
    subprocess.run(command, check=True)
    enhance_audio_quality(output)

def enhance_audio_quality(input_file):
    output_file = "enhanced_speech.wav"
    command = ["ffmpeg", "-y", "-i", input_file, "-ac", "2", "-ar", "44100", "-sample_fmt", "s16", output_file]
    subprocess.run(command, check=True)
    play_audio(output_file)

def play_audio(file_path):
    def run_player():
        subprocess.run(["aplay", file_path])
    player_thread = threading.Thread(target=run_player)
    player_thread.start()
    input("Press Enter to stop playback...\n")
    subprocess.run(["pkill", "aplay"])

def record_and_recognize():
    print("Recording... Speak now. Press Enter to stop.")
    speech = LiveSpeech(
        verbose=False,
        sampling_rate=16000,
        buffer_size=2048,
        no_search=False,
        full_utt=True
    )
    recognized_text = ""
    try:
        for phrase in speech:
            print('Recognized:', str(phrase))
            recognized_text += str(phrase) + " "
    except KeyboardInterrupt:
        print("Recording manually stopped.")
    return recognized_text.strip()


def main():
    print("\n")
    print("Welcome to the conversational AI!")
    load_history = input("Do you want to load the previous chat history? (yes/no): ").strip().lower()
    print("\n")
    conversation_history = load_chat_history() if load_history == 'yes' else []
    add_system_prompt = not conversation_history  # Add system prompt only if no history loaded


    prompts = load_prompts('./src/prompts/system_prompts.json')
    print("Available System Prompts:")
    for key, value in prompts.items():
        print(f"{key}: {value['one_word_description']} - {value['description']}")
    print("\n")
    prompt_id = input("Enter the ID of the system prompt you want to use (leave blank for default): ")
    system_prompt = get_system_prompt(prompts, prompt_id)
    print("\n")

    voices = {
        '1': 'en-GB',
        '2': 'en-US',
        '3': 'de-DE',
        '4': 'es-ES',
        '5': 'fr-FR',
        '6': 'it-IT'
    }
    model_list = {
        1: 'mistral:latest',
        2: 'mistral:7b-instruct-v0.2-fp16', 
        3: 'dolphin-mixtral:latest'
    }

    print("Select a model to use:")
    print("\n")
    for key, value in model_list.items():
        print(f"{key}: {value}")
        
    print("\n")
    model_choice = input("Enter the number of the model you want to use: ").strip()
    if model_choice.isdigit() and int(model_choice) in model_list:
        model = model_list[int(model_choice)]
    else:
        model = model_list[2]  # Default model if invalid choice
        print(f"Invalid or no model choice. Using default model '{model}'.")

    print("Select a voice:")
    print("\n")
    for key, value in voices.items():
        print(f"{key}: {value}")
    print("\n")
    choice = input("Enter the number of the voice you want to use: ").strip()
    language = voices.get(choice, 'en-GB')  # Default voice if invalid choice

    assistant_name = input("Enter the name for the assistant: ").strip()
    if not assistant_name:
        assistant_name = 'Plex'  # Default name if no input
        print("No name entered. Using default name 'Plex'.")

    while True:
        user_input = input("Type 'speak' to record, enter your text directly, or type 'quit' to exit:")
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        elif user_input.lower() == 'speak':
            recognized_text = record_and_recognize()
            if recognized_text:
                print("Recognized text:", recognized_text)
                generate_base_response(recognized_text, conversation_history, model, assistant_name, system_prompt, language, add_system_prompt)
                add_system_prompt = False  # Prevent further system prompt inclusion
            else:
                print("No speech recognized. Please try again.")
        else:
            generate_base_response(user_input, conversation_history, model, assistant_name, system_prompt, language, add_system_prompt)
            add_system_prompt = False  # Prevent further system prompt inclusion

if __name__ == '__main__':
    main()
