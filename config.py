import os
from dotenv import load_dotenv

# Load the .env file once
load_dotenv()

# Read the variables and store them in clean Python constants
API_KEY = os.getenv("GEMINI_API_KEY")
