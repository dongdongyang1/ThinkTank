import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()

@dataclass
class LLMConfig:
    base_url : str
    api_key : str
    vl_model : str
    llm_model : str
    item_model : str
    llm_temperature : str


lm_config = LLMConfig(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    vl_model=os.getenv("VL_MODEL"),
    llm_model=os.getenv("LLM_DEFAULT_MODEL"),
    item_model=os.getenv("ITEM_MODEL"),
    llm_temperature=float(os.getenv("LLM_DEFAULT_TEMPERATURE"))
)