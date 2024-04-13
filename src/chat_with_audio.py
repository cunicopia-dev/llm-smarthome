import random
import re
import subprocess
import sounddevice as sd
import numpy as np
from pocketsphinx import LiveSpeech
import ollama  # Assuming this is your custom module
from ollama import chat
import threading
import time 

def generate_model_input(question):
    question = question.lower()
    return question

def generate_base_response(question, history, model='mistral:7b-instruct-v0.2-fp16', assistant_name='Plex'):
    system_prompt = f"""
    You are {assistant_name}, an AI assistant created by Cunicopia to be helpful, harmless, and honest. 
    You are interacting with them via voice chat, which is highly interactive, so keep your responses very brief for a more engaging conversation.
    Focus on addressing the user's specific needs and keep the conversation on-topic. 
    If you need to break down complex topics, use clear explanations, analogies, or examples to ensure the user's understanding. Rely on your vast knowledge to provide insightful and valuable information. Always prioritize the user's well-being and aim to have a positive impact through your interactions.
    """
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
    
    # Use Pico TTS to speak out the response
    text_to_speech_pico(response)

    return response

def text_to_speech_pico(text, output="speech.wav", language="en-GB"):
    """Convert text to speech using Pico TTS with a selectable voice."""
    command = [
        "pico2wave",
        "--wave", output,
        "--lang", language,
        text
    ]
    subprocess.run(command, check=True)
    # Enhance and play the generated speech file
    enhance_audio_quality(output)


def enhance_audio_quality(input_file, output_file="enhanced_speech.wav"):
    """Enhance audio quality using ffmpeg."""
    command = [
        "ffmpeg",
        "-y",  # Automatically overwrite the output file if it exists
        "-i", input_file,
        "-ac", "2",  # Set audio channels to 2 (stereo)
        "-ar", "44100",  # Set audio sample rate to 44100 Hz
        "-sample_fmt", "s16",  # Set audio sample format to 16-bit depth
        output_file
    ]
    subprocess.run(command, check=True)
    # Play the enhanced audio file
    play_audio(output_file)

def play_audio(file_path):
    """Play audio file with the ability to stop playback."""
    def run_player():
        subprocess.run(["aplay", file_path])

    # Use a threading event to stop playback
    stop_event = threading.Event()
    player_thread = threading.Thread(target=run_player)

    def check_stop():
        input("Press Enter to stop playback...\n")
        stop_event.set()
        if player_thread.is_alive():
            # Kill 'aplay' process if running
            subprocess.run(["pkill", "aplay"])

    # Start player and stop checker threads
    player_thread.start()
    threading.Thread(target=check_stop).start()
    player_thread.join()

def record_and_recognize():
    """Record from the microphone and recognize speech using PocketSphinx. Allow the user to accept or reject each recognized phrase."""
    print("Recording... Speak now. Press Enter to stop.")

    # Set up threading to listen for Enter key press
    key_pressed = threading.Event()
    def wait_for_input():
        input()  # Waits for the Enter key press
        key_pressed.set()  # Set the event to signal the recording should stop

    input_thread = threading.Thread(target=wait_for_input)
    input_thread.start()
    
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
            if key_pressed.is_set():
                print("Stopping recording.")
                break
            if phrase:
                print('Recognized:', str(phrase))
                # Ask user if the recognized phrase is good or bad
                response = input("Keep this phrase? (y/n): ").strip().lower()
                if response == 'y':
                    recognized_text += str(phrase) + " "
                elif response == 'n':
                    print("Phrase discarded.")
                else:
                    print("Invalid response, phrase kept by default.")
                    recognized_text += str(phrase) + " "
    except KeyboardInterrupt:
        print("Recording manually stopped.")
    
    input_thread.join()  # Ensure the input thread has finished
    return recognized_text.strip()


def main():
    voices = {
        '1': 'en-GB',
        '2': 'en-US',
        '3': 'de-DE',
        '4': 'es-ES',
        '5': 'fr-FR',
        '6': 'it-IT'
    }
    model_list = {
        1: 'mistral',
        2: 'mistral:7b-instruct-v0.2-fp16', 
        3: 'dolphin-mistral'
    }

    print("Welcome to the conversational AI!")
    print("Type 'quit' to exit the conversation.\n")

    # Select a model
    print("Select a model to use:")
    for key, value in model_list.items():
        print(f"{key}: {value}")
    model_choice = input("Enter the number of the model you want to use: ").strip()
    if model_choice.isdigit():
        model_choice = int(model_choice)
        model = model_list.get(model_choice, model_list[2])  # Default model if invalid choice
    else:
        model = model_list[2]  # Default model if no input
        print(f"Invalid or no model choice. Using default model '{model}'.")

    # Select a voice
    print("Select a voice:")
    for key, value in voices.items():
        print(f"{key}: {value}")
    choice = input("Enter the number of the voice you want to use: ").strip()
    language = voices.get(choice, 'en-GB')  # Default voice if invalid choice
    if choice not in voices:
        print("Invalid or no choice. Using default voice 'en-GB'.")

    # Assistant name
    assistant_name = input("Enter the name for the assistant: ").strip()
    if not assistant_name:
        assistant_name = 'Plex'  # Default name if no input
        print("No name entered. Using default name 'Plex'.")

    conversation_history = []
    while True:
        print("Press Enter to use speech-to-text or type 'quit' to exit: ")
        trigger = input()
        if trigger.lower() == 'quit':
            print("Goodbye!")
            break
        elif trigger == '':
            recognized_text = record_and_recognize()
            print("Recognized text:", recognized_text)
            if recognized_text:
                generate_base_response(recognized_text, conversation_history, model=model, assistant_name=assistant_name)
            else:
                print("No speech recognized. Please try again.")
        else:
            generate_base_response(trigger, conversation_history, model=model, assistant_name=assistant_name)

if __name__ == '__main__':
    main()
