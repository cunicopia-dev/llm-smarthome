import unittest
from unittest.mock import patch
from io import StringIO
from chat import load_prompts, get_system_prompt, save_chat_history, load_chat_history, generate_base_response

class ConversationalAITestCase(unittest.TestCase):

    def test_load_prompts(self):
        prompts = load_prompts('./src/prompts/system_prompts.json')
        self.assertIsNotNone(prompts)
        self.assertTrue('1' in prompts)
        self.assertEqual(prompts['1']['description'], 'Base Assistant Prompt')

    def test_get_system_prompt(self):
        prompts = load_prompts('./src/prompts/system_prompts.json')
        system_prompt = get_system_prompt(prompts, '1')
        self.assertEqual(system_prompt, 'Base Assistant Prompt')

        system_prompt = get_system_prompt(prompts, 'invalid_id')
        self.assertEqual(system_prompt, prompts['1']['description'])

    def test_save_chat_history(self):
        history = [{'role': 'user', 'content': 'Hello'}, {'role': 'assistant', 'content': 'Hi'}]
        save_chat_history(history)
        self.assertTrue(os.path.exists('chat.json'))

    def test_load_chat_history(self):
        history = [{'role': 'user', 'content': 'Hello'}, {'role': 'assistant', 'content': 'Hi'}]
        save_chat_history(history)
        loaded_history = load_chat_history()
        self.assertEqual(loaded_history, history)

    @patch('sys.stdout', new=StringIO())
    def test_generate_base_response(self, mock_stdout):
        history = [{'role': 'user', 'content': 'Hello'}]
        model = 'test_model'
        assistant_name = 'Test Assistant'
        system_prompt = 'Test System Prompt'
        question = 'What is your name?'

        generate_base_response(question, history, model, assistant_name, system_prompt)

        expected_output = f'{assistant_name}: '
        self.assertEqual(mock_stdout.getvalue(), expected_output)

if __name__ == '__main__':
    unittest.main()