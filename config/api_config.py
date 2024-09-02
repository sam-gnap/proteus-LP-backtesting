import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Key for data fetching
API_KEY = os.getenv("API_KEY")
