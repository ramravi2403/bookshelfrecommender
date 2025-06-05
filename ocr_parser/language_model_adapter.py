from langchain_core.prompts import PromptTemplate
from .ModelFactory import ModelFactory
from ocr_parser.interfaces import ILanguageModel
from dotenv import load_dotenv


class LangChainModelAdapter(ILanguageModel):

    def __init__(self, model_name: str, prompt_template_path: str):
        load_dotenv()
        self.model = ModelFactory.create_model(model_name)
        self.prompt = self._load_prompt_template(prompt_template_path)

    def _load_prompt_template(self, template_path: str) -> PromptTemplate:
        """Load prompt template from file."""
        with open(template_path, 'r', encoding='utf-8') as f:
            template = f.read()

        return PromptTemplate(
            input_variables=["text"],
            template=template
        )

    async def generate_async(self, prompt: str, **kwargs) -> str:
        """Generate text asynchronously."""
        chain = self.prompt | self.model
        response = await chain.ainvoke({"text": prompt})
        return response.content.strip()
