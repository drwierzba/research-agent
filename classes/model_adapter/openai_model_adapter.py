from langchain.chat_models import init_chat_model
from classes.model_adapter.model_adapter import ModelAdapter


class OpenAIModelAdapter(ModelAdapter):
    def __init__(self, model_str = "gpt-4o-mini"):
        self.model = init_chat_model(model_str, model_provider="openai", temperature=0)

    def invoke(self, prompt):
        return self.model.invoke(prompt)

    def invoke_with_images(self, prompt, images):
        message = {"role": "user", "content": [{"type": "text", "text": prompt}]}
        for img in images:
            message["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img}"}
            })
        return self.model.invoke([message])

    def with_structured_output(self, output_type, prompt):
        return self.model.with_structured_output(output_type).invoke(prompt)