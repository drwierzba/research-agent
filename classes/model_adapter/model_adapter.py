from abc import ABC, abstractmethod

class ModelAdapter(ABC):
    """Abstract adapter for different LLM model interfaces"""

    @abstractmethod
    def invoke(self, prompt):
        """Invoke the model with a text prompt and images"""
        pass

    @abstractmethod
    def invoke_with_images(self, prompt, images):
        """Invoke the model with a text prompt and images"""
        pass

    @abstractmethod
    def with_structured_output(self, output_type, prompt):
        """Invoke the model with a text prompt, returning structured output"""
        pass
