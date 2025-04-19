# classes/model_adapter/model_adapter_factory.py
class ModelAdapterFactory:
    @staticmethod
    def create_adapter(adapter_type, model_str=None):
        """Create a model adapter based on type"""
        if adapter_type.lower() == "claude":
            from classes.model_adapter.claude_model_adapter import ClaudeModelAdapter
            return ClaudeModelAdapter(model_str)
        elif adapter_type.lower() == "openai":
            from classes.model_adapter.openai_model_adapter import OpenAIModelAdapter
            return OpenAIModelAdapter(model_str)
        else:
            raise ValueError(f"Unknown adapter type: {adapter_type}")