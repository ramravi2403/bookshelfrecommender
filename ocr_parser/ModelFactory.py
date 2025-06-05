from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models import BaseChatModel
class ModelFactory:

    @classmethod
    def create_model(cls, model_name: str) -> BaseChatModel:
        model_name = model_name.lower()
    
        if "claude" in model_name:
            return ChatAnthropic(model_name=model_name)
        elif "gpt" in model_name:
            return ChatOpenAI(model_name=model_name)
        elif "gemini" in model_name:
            return ChatGoogleGenerativeAI(model=model_name)
        else:
            raise NotImplementedError(f"Unsupported model name: {model_name}. Must contain 'claude', 'gpt', or 'gemini'.")
