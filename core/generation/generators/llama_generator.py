from .. import BaseGenerator
from typing import List, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class LlamaGenerator(BaseGenerator):
    def load_model(self):
        """加载Llama模型"""
        model_name = self.config.get('model_name', 'llama-2-7b-chat-hf')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else "cpu"
        )

    def generate(self, context: List[str], query: str) -> str:
        """使用Llama生成回答"""
        prompt = self.format_prompt(context, query)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=self.config.get('max_length', 1024),
                temperature=self.config.get('temperature', 0.7),
                do_sample=True
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.replace(prompt, "").strip() 