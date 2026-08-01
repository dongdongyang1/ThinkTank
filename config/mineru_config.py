import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class MineruConfig:
    api_token : str
    base_url : str

mineru_config = MineruConfig(
    api_token = os.getenv("MINERU_API_TOKEN", ""),
    base_url = os.getenv("MINERU_BASE_URL", "")
)



