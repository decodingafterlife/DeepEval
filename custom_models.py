# custom_models.py

from typing import List, Optional, Union
from pydantic import BaseModel, ValidationError
import ollama
import json
from sentence_transformers import SentenceTransformer
from deepeval.models import DeepEvalBaseLLM, DeepEvalBaseEmbeddingModel

class CustomEmbeddingModel(DeepEvalBaseEmbeddingModel):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        super().__init__(model_name)

    def load_model(self):
        return self.model

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts)
        return [embedding.tolist() for embedding in embeddings]

    def embed_text(self, text: str) -> List[float]:
        embedding = self.model.encode(text)
        return embedding.tolist()

    async def a_embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self.embed_texts(texts)

    async def a_embed_text(self, text: str) -> List[float]:
        return self.embed_text(text)

    def get_model_name(self) -> str:
        return self.model_name

class CustomOllamaLLM(DeepEvalBaseLLM):
    def __init__(self, model: str, host: str = "http://localhost:11434", **kwargs):
        self.model_name = model
        self.host = host
        self.kwargs = kwargs
        super().__init__(model)

    def load_model(self):
        try:
            return ollama.Client(host=self.host, **self.kwargs)
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Ollama at {self.host}.") from e

    def _create_enhanced_prompt(self, prompt: str, schema: BaseModel) -> str:
        schema_dict = schema.model_json_schema()
        required_keys = list(schema_dict.get('properties', {}).keys())
        enhanced_prompt = f"{prompt}\n\nProvide a response in JSON format. The JSON object must include these keys: {required_keys}. Do not add any extra text, explanations, or Markdown formatting. Your response must be a single, valid JSON object.\n\nJSON Response:"
        return enhanced_prompt

    def generate(self, prompt: str, schema: Optional[BaseModel] = None) -> Union[str, BaseModel]:
        client = self.load_model()
        if schema:
            enhanced_prompt = self._create_enhanced_prompt(prompt, schema)
            response = client.chat(model=self.model_name, messages=[{"role": "user", "content": enhanced_prompt}], format='json')
            json_string = response['message']['content']
            data = json.loads(json_string)
            if 'response' in data and isinstance(data['response'], dict):
                data['response'] = json.dumps(data['response'])
                json_string = json.dumps(data)
            return schema.model_validate_json(json_string)
        else:
            response = client.chat(model=self.model_name, messages=[{"role": "user", "content": prompt}])
            return response['message']['content']

    async def a_generate(self, prompt: str, schema: Optional[BaseModel] = None) -> Union[str, BaseModel]:
        client = ollama.AsyncClient(host=self.host, **self.kwargs)
        if schema:
            enhanced_prompt = self._create_enhanced_prompt(prompt, schema)
            response = await client.chat(model=self.model_name, messages=[{"role": "user", "content": enhanced_prompt}], format='json')
            json_string = response['message']['content']
            data = json.loads(json_string)
            if 'response' in data and isinstance(data['response'], dict):
                data['response'] = json.dumps(data['response'])
                json_string = json.dumps(data)
            try:
                return schema.model_validate_json(json_string)
            except ValidationError as e:
                raise RuntimeError(f"Ollama's JSON output failed validation: {e}\nOutput:\n{json_string}") from e
        else:
            response = await client.chat(model=self.model_name, messages=[{"role": "user", "content": prompt}])
            return response['message']['content']

    def get_model_name(self) -> str:
        return f"{self.model_name} (Ollama)"