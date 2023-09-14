import os
import numpy as np
import pinecone
import requests  # Import the requests library
from dotenv import load_dotenv
import nylas
from nylas import APIClient

# Define the function to embed text
def embed_text(text):
    # Replace this with your text embedding logic
    # For demonstration, we are using a random embedding
    return np.random.rand(256).tolist()

# Define the main function
def main():
    # Load environment variables from .env file
    load_dotenv()

    # Initialize Pinecone client
    pinecone.init(api_key=str(os.getenv("PINECONE_API_KEY")), environment="gcp-starter")

    index_name = "email-gpt"  # Replace with your desired index name

    # Define the number of recent emails you want to retrieve
    num_emails_to_fetch = 2

    # Retrieve the recent emails
    emails = nylas.messages.where(limit=num_emails_to_fetch)

    # Initialize an empty list to store email data
    email_data = []

    # Iterate through the retrieved emails
    for email in emails:
        # Check if the email already exists in the Pinecone database by message ID
        index = pinecone.Index(index_name)
        result = index.query(queries=[email.id], top_k=1)  # Added top_k parameter

        # If the email is not in the database, embed and add it
        if not result or not result[0].matches:
            embeddings = embed_text(email.body)

            # Get sender information
            sender_name = None
            sender_email = None
            if email.from_ and len(email.from_) > 0:
                sender_name = email.from_[0].get("name")
                sender_email = email.from_[0].get("email")

            # Get recipient information
            to_name = None
            to_email = None
            if email.to and len(email.to) > 0:
                recipient = email.to[0]  # Assuming there is one recipient
                to_name = recipient.get("name")
                to_email = recipient.get("email")

            # Create a dictionary for this email with required data
            email_dict = {
                "id": email.id,  # Message ID
                "values": embeddings,  # Embedding as list of floats
                "metadata": {
                    "message_snippet": email.body[:100],  # Adjust the length as needed
                    "from_name": sender_name,
                    "from_email": sender_email,
                    "to_name": to_name,
                    "to_email": to_email,
                    "subject": email.subject,
                    "date": email.date,
                },
            }
            email_data.append(email_dict)
            # Add the email to the Pinecone database
            index.upsert(email_data)

    # Get a query from the user at runtime
    user_query = input("Enter your query: ")

    # Embed the user query
    query_embedding = embed_text(user_query)

    # Define the Pinecone query payload
    pinecone_payload = {
        "vector": query_embedding,
        "topK": 5,
        "includeMetadata": True,
        "includeValues": True,
        "namespace": ""
    }

    # Define the Pinecone query endpoint URL
    pinecone_url = f"https://email-gpt-3d030fb.svc.gcp-starter.pinecone.io"

    # Define the headers for the request
    headers = {
        "Content-Type": "application/json",
        "Api-Key": os.getenv("PINECONE_API_KEY")
    }

    # Send the query request to Pinecone
    response = requests.post(pinecone_url, json=pinecone_payload, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse and print the query results
        results = response.json()
        for result in results:
            email_id = result["id"]
            email_metadata = result["metadata"]
            email_values = result["values"]
            print(f"Message ID: {email_id}")
            print(f"Message Body: {email_metadata['message_snippet']}")
            print(f"Embedding Values: {email_values}\n")
    else:
        print(f"Query request failed with status code: {response.status_code}")

if __name__ == "__main__":
    main()
