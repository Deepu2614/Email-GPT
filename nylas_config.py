import os
from nylas import APIClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve Nylas API credentials from environment variables
nylas_client_id = os.getenv("NYLAS_CLIENT_ID")
nylas_client_secret = os.getenv("NYLAS_CLIENT_SECRET")
access_token = os.getenv("NYLAS_ACCESS_TOKEN")

# Initialize the Nylas API client
nylas = APIClient(nylas_client_id, nylas_client_secret, access_token=access_token)
