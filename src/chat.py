import streamlit as st
import json
from ollama import chat

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

    def load_prompts(self, file_path):
        with open(file_path, 'r') as file:
            self.prompts = json.load(file)

    def get_system_prompt(self, prompt_id=None):
        return self.prompts.get(prompt_id, self.prompts['1'])['description']

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
                if message['role'] == 'system':
                    st.markdown(f"<div style='text-align: center; color: red; font-size: 16px;'>⚙️ {message['content']}</div>", unsafe_allow_html=True)
                elif message['role'] == 'user':
                    st.markdown(f"<div style='text-align: right; color: white; font-size: 16px;'>👤 {message['content']}</div>", unsafe_allow_html=True)
                else:  # 'assistant'
                    st.markdown(f"<div style='text-align: left; color: grey; font-size: 16px;'>🤖 {message['content']}</div>", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)  # Visual separator

    def run(self):
        st.set_page_config(page_title="Conversational AI", page_icon=":robot_face:")
        st.title("Chat with AI")

        with st.sidebar:
            model_choice = st.selectbox("Select a model:", list(self.model_list.values()))
            prompt_id = st.selectbox("Select a prompt:", list(self.prompts.keys()), format_func=lambda x: self.prompts[x]['one_word_description'])
            system_prompt = self.get_system_prompt(prompt_id)

        chat_container = st.container()
        with chat_container:
            self.display_chat()

        if 'input_key' not in st.session_state:
            st.session_state.input_key = 0

        user_input = st.text_input("You:", key=f"user_input_{st.session_state.input_key}", disabled=st.session_state.get('request_in_progress', False))

        if st.button("Send", disabled=st.session_state.get('request_in_progress', False)) and user_input and not st.session_state.get('last_input', '') == user_input:
            response = self.generate_response(user_input, model_choice, system_prompt)
            st.session_state['last_input'] = user_input  # Track last input to prevent duplication
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