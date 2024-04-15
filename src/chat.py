import streamlit as st
import json
from ollama import chat
import datetime
import uuid
import os

class ConversationalAIApp:
    def __init__(self):
        self.load_prompts('./src/prompts/system_prompts.json')
        self.model_list = {
            1: 'llama2:13b-chat-q8_0',
            2: 'gemma:7b-instruct-v1.1-fp16',
            2: 'mistral:7b-instruct-v0.2-fp16',
            3: 'dolphin-mixtral',
            5: 'mixtral:latest',
            6: 'codegemma:7b-instruct-fp16'
        }

        self.colors = {
            'system': '#E8E8E8',  # Light grey for system messages
            'user': '#D6EAF8',    # Very light blue for user messages
            'assistant': '#D5F5E3'  # Very light green for assistant messages
        }


    def load_prompts(self, file_path):
        with open(file_path, 'r') as file:
            self.prompts = json.load(file)

    def get_system_prompt(self, prompt_id=None):
        return self.prompts.get(prompt_id, self.prompts['1'])['description']

    def save_chat_history(self):
        if 'chat_title' not in st.session_state or not st.session_state['chat_title']:
            initial_message = st.session_state['conversation_history'][1]['content']  # Assuming the second entry is the user's first message

            # Generate chat title after first user message
            
            st.session_state['chat_title'] = self.generate_chat_title()
        
        # Now use the chat title in the filename
        filename = f"{st.session_state['chat_title']}.json".replace(" ", "_")
        filepath = os.path.join('./chats', filename)
        with open(filepath, 'w') as file:
            json.dump(st.session_state['conversation_history'], file, indent=4)
        return filename

    # TODO: Make this work. Right now something strange is happening with the model
    def generate_chat_title(self):
        if 'conversation_history' not in st.session_state or not st.session_state['conversation_history']:
            return 'New_Chat'

        # Directly use the conversation history; no need to transform it into a single string
        title_messages = st.session_state['conversation_history']

        # Add a message to clearly instruct the model its job is to generate a chat title
        # title_messages.append({'role': 'system', 'content': 'Please generate a title for this chat, based on the conversation so far. Make sure it is maximum 3 words long.'})


        # Assuming 'chat' is a method for interacting with your language model
        try:
            response = chat(model=self.model_list[1], messages=title_messages)
            if response:
                title_words = response.split()
                title = '_'.join(title_words[:min(5, len(title_words))])  # Up to 5 words
                return title
            else:
                raise ValueError("No valid response from the model.")
        except Exception as e:
            # Properly handle errors, log them if necessary
            print(f"Error generating chat title: {str(e)}")
            return 'Chat_' + uuid.uuid4().hex[:8]


    def load_chat_histories(self):
        chat_files = [f for f in os.listdir('./chats') if f.endswith('.json')]
        chat_files.sort(reverse=True)  # Optional: sort by creation date, newest first
        return chat_files

    def load_chat_history(self, filename):
        # Extract chat_id from the filename and store it in session state
        chat_id = filename.split('_')[1].split('.')[0]
        st.session_state['chat_id'] = chat_id
        # Load the chat history as before
        filepath = os.path.join('./chats', filename)
        with open(filepath, 'r') as file:
            st.session_state['conversation_history'] = json.load(file)
            st.session_state['set_system_prompt'] = False

    def display_chat_selector(self):
        chat_files = self.load_chat_histories()
        current_chat = f'chat_{st.session_state.get("chat_id", "")}.json'
        
        # Set a default index for the selectbox
        default_index = 0
        
        # If the current chat is in the chat_files, set that as the default index
        if current_chat in chat_files:
            default_index = chat_files.index(current_chat) + 1  # +1 because of the empty string at the start

        # Display the select box with the default index
        selected_chat = st.sidebar.selectbox("Select a previous chat:", [""] + chat_files, index=default_index)
        
        # Load the chat history if a chat is selected
        if selected_chat and selected_chat != current_chat:
            self.load_chat_history(selected_chat)


    def generate_response(self, question, model, system_prompt):
        if question.strip():  # Ensure we don't process empty questions
            st.session_state['request_in_progress'] = True
            conversation_history = st.session_state.get('conversation_history', [])
            if st.session_state['set_system_prompt']:
                conversation_history.append({'role': 'system', 'content': system_prompt})
                st.session_state['set_system_prompt'] = False  # Reset after adding system prompt
            conversation_history.append({'role': 'user', 'content': question})
            response = ""
            stream = chat(model=model, messages=conversation_history, stream=True)
            for chunk in stream:
                if 'message' in chunk:
                    content = chunk['message']['content']
                    response += content
            if response:  # Append response only if it's non-empty
                conversation_history.append({'role': 'assistant', 'content': response})
            st.session_state['conversation_history'] = conversation_history
            st.session_state['request_in_progress'] = False
            return response
        return ""

    def display_chat(self):
        if 'conversation_history' in st.session_state:
            for message in st.session_state['conversation_history']:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                role = message['role'].capitalize()
                color = self.colors[message['role']]
                name_map = {'user': 'User', 'assistant': 'Assistant', 'system': 'System'}
                name = name_map[message['role']]
                
                if message['role'] == 'system':
                    html_content = f"<div style='padding: 8px;'><strong>{name}</strong> <small>({timestamp})</small><hr style='margin-top: 3px; margin-bottom: 6px;'><div style='background-color: {color}; padding: 6px; border-radius: 8px;'>{message['content']}</div></div>"
                else:
                    align = 'right' if message['role'] == 'user' else 'left'
                    html_content = f"<div style='text-align: {align}; padding: 8px;'><strong>{name}</strong> <small>({timestamp})</small><hr style='margin-top: 3px; margin-bottom: 6px;'><div style='background-color: {color}; padding: 6px; border-radius: 8px;'>{message['content']}</div></div>"
                    
                st.markdown(html_content, unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)  # Visual separator at the bottom of the chat


    def run(self):
        st.set_page_config(page_title="Conversational AI", page_icon=":robot_face:")
        st.title("Chat with AI")

        with st.sidebar:
            model_choice = st.selectbox("Select a model:", list(self.model_list.values()))
            prompt_id = st.selectbox("Select a prompt:", list(self.prompts.keys()), format_func=lambda x: self.prompts[x]['one_word_description'])
            system_prompt = self.get_system_prompt(prompt_id)

            self.display_chat_selector()

        chat_container = st.container()
        with chat_container:
            self.display_chat()

        if 'input_key' not in st.session_state:
            st.session_state.input_key = 0

        if 'chat_id' not in st.session_state:
            st.session_state['chat_id'] = uuid.uuid4().hex 

        user_input = st.text_input("You:", key=f"user_input_{st.session_state.input_key}", disabled=st.session_state.get('request_in_progress', False))

        if st.button("Send", disabled=st.session_state.get('request_in_progress', False)) and user_input and not st.session_state.get('last_input', '') == user_input:
            response = self.generate_response(user_input, model_choice, system_prompt)
            if len(st.session_state['conversation_history']) == 2:  # One system prompt and one user message
                self.save_chat_history()  # Save the chat history after the first user message
            st.session_state['last_input'] = user_input  # Track last input to prevent duplication
            self.save_chat_history()  # Save after sending message
            st.session_state.input_key += 1  # Increment the key to reset the input box
            st.rerun()

        if st.button("Clear Chat"):
            st.session_state['conversation_history'] = []
            st.session_state['request_in_progress'] = False
            st.session_state['set_system_prompt'] = True
            st.session_state['last_input'] = ""  # Reset the last input
            st.session_state.input_key += 1  # Ensure input field is reset
            st.rerun()
        
if __name__ == '__main__':
    if 'conversation_history' not in st.session_state:
        st.session_state['conversation_history'] = []
    if 'request_in_progress' not in st.session_state:
        st.session_state['request_in_progress'] = False
    if 'set_system_prompt' not in st.session_state:
        st.session_state['set_system_prompt'] = True  # Initialize this state to manage system prompts
    app = ConversationalAIApp()
    app.run()