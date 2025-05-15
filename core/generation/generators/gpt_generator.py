import openai
from .. import BaseGenerator
from typing import List, Dict, Any

class GPTGenerator(BaseGenerator):
    def load_model(self):
        """加载OpenAI GPT模型"""
        openai.api_key = self.config.get('api_key')
        self.model = self.config.get('model_name', 'gpt-3.5-turbo')

    def generate(self, context: List[str], query: str) -> str:
        """使用GPT生成回答"""
        prompt = self.format_prompt(context, query)
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.get('temperature', 0.7),
                max_tokens=self.config.get('max_tokens', 512)
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating response: {e}")
            return "An error occurred while generating the response." 