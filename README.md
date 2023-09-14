# Email-GPT

Email-GPT is a tool that leverages recent emails from your inbox to answer questions related to your email content.

## Getting Started

To use Email-GPT, follow these steps:

### Prerequisites

Make sure you have the following prerequisites installed on your system:

- Python 3.x
- [Pinecone API Key](https://www.pinecone.io/docs/api-keys/)
- [Nylas API Credentials](https://www.nylas.com/platform/api-keys)

### Installation

1. Clone the Email-GPT repository to your local machine:

   ```bash
   git clone https://github.com/your-username/email-gpt.git
   cd email-gpt


# email_gpt/config.py

PINECONE_API_KEY = "YOUR_PINECONE_API_KEY"
NYLAS_API_CREDENTIALS = {
    "client_id": "YOUR_NYLAS_CLIENT_ID",
    "client_secret": "YOUR_NYLAS_CLIENT_SECRET",
}


### Usage

Once you have installed and configured Email-GPT, you can start using it by running the following command:

bash
python email_gpt.py


Email-GPT will retrieve your recent emails, create embeddings for each email, and store them in the Pinecone index.

You can then interact with Email-GPT by providing a user query. It will search for the most relevant email and generate an answer based on the provided query.

### Example:


User query: What is the deadline for the new project?

Email-GPT response: The deadline for the new project is Friday, September 22nd.
