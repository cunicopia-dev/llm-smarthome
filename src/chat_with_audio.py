import random
import re
import subprocess
import sounddevice as sd
import numpy as np
from pocketsphinx import LiveSpeech
import ollama  # Assuming this is your custom module
from ollama import chat

def generate_model_input(question):
    question = question.lower()
    return question

def generate_base_response(question, history, model='mistral:7b-instruct-v0.2-fp16', assistant_name='Plex'):
    system_prompt = f"""
    You are {assistant_name}, an AI assistant created by Cunicopia to be helpful, harmless, and honest. Your purpose is to provide accurate, relevant, and comprehensive responses to the user's questions, while maintaining a friendly, respectful, and professional tone. Focus on addressing the user's specific needs and keep the conversation on-topic. 
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

def text_to_speech_pico(text, output="speech.wav"):
    """Convert text to speech using Pico TTS."""
    command = [
        "pico2wave", 
        "--wave", output, 
        "--lang", "en-GB", 
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
    subprocess.run(["aplay", output_file])

def record_and_recognize():
    """Record from the microphone and recognize speech using PocketSphinx."""
    print("Press Enter to start recording, then speak. Press Enter again to stop.")
    input()  # Wait for Enter key to start
    print("Recording...")
    audio = sd.rec(int(5 * 16000), samplerate=16000, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    print("Stopped recording.")
    audio = np.int16(audio).tobytes()

    speech = LiveSpeech(
        audio,
        verbose=False,
        sampling_rate=16000,
        buffer_size=2048,
        no_search=False,
        full_utt=False
    )
    for phrase in speech:
        print('Recognized:', phrase)
        break  # Stop after the first phrase recognized

def main():
    print("Welcome to the conversational AI!")
    print("Type 'quit' to exit the conversation.\n")
    model_list = {1: 'mixtral', 2: 'mistral:7b-instruct-v0.2-fp16', 3: 'dolphin-mixtral'}

    print("Select a model to use:")
    for key, value in model_list.items():
        print(f"{key}: {value}")
    model_choice = input("Enter the number of the model you want to use: ")
    if model_choice.isdigit():
        model_choice = int(model_choice)
        if model_choice in model_list:
            model = model_list[model_choice]
        else:
            model = model_list[2]  # Default to 'mistral:7b-instruct-v0.2-fp16'
            print(f"Invalid model choice. Using default model '{model}'.")

    assistant_name = input("Enter the name for the assistant: ")

    conversation_history = []
    while True:
        print("Press Enter to use speech-to-text or type 'quit' to exit: ")
        trigger = input()
        if trigger.lower() == 'quit':
            print("Goodbye!")
            break
        elif trigger == '':
            record_and_recognize()
        else:
            user_input = trigger
            generate_base_response(user_input, conversation_history, model=model, assistant_name=assistant_name)

if __name__ == '__main__':
    main()
