import os
import nylas
from transformers import AutoTokenizer, AutoModel, GPT2LMHeadModel, GPT2Tokenizer
import torch
from dotenv import load_dotenv
from nylas import APIClient
from pinecone import Pinecone

load_dotenv()

# Initialize Pinecone
pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment="gcp-starter")
pinecone.init()

# List Pinecone indexes
indexes = pinecone.list_indexes()
print("Indexes:", indexes)

# Retrieve Nylas API credentials from environment variables
nylas_client_id = os.getenv("NYLAS_CLIENT_ID")
nylas_client_secret = os.getenv("NYLAS_CLIENT_SECRET")
access_token = os.getenv("NYLAS_ACCESS_TOKEN")

# Initialize the Nylas API client
nylas = APIClient(client_id=nylas_client_id, client_secret=nylas_client_secret)
nylas.get_access_token(code=access_token)

# Retrieve email messages
def retrieve_messages():
    messages = nylas.messages.list(limit=1)
    message = messages[0]
    print("Message:", message)

    id = message.id
    metadata = {
        'snippet': message.snippet,
        'fromName': message.from_name if message.from_ else None,
        'fromEmail': message.from_email if message.from_ else None,
        'toName': message.to_name if message.to_ else None,
        'toEmail': message.to_email if message.to_ else None,
        'subject': message.subject,
        'date': message.date,
        'id': message.id,
    }
    print("Message Metadata:", metadata)
    return message, metadata, id

message_data = retrieve_messages()

# Create embeddings for the email message
def create_embeddings(input_text):
    model_name = "bert-base-uncased"  # Specify the model name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Tokenize and convert input text to embeddings
    input_tokens = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = model(**input_tokens)

    # Extract embeddings (CLS token embedding for BERT)
    embeddings = output.last_hidden_state.mean(dim=1).squeeze().numpy()

    return embeddings

embedding = create_embeddings(message_data[0])

# Store the vector in Pinecone
def store_vector(embedding, id, metadata):
    upsert_response = pinecone.upsert(
        index_name="email-gpt",
        upsert_requests=[{
            "id": id,
            "values": embedding.tolist(),
            "metadata": metadata
        }]
    )
    print("Upsert Response:", upsert_response)

store_vector(embedding, message_data[2], message_data[1])

# Create a query vector
def create_query_vector(input_text):
    model_name = "bert-base-uncased"  # Specify the model name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Tokenize and convert input text to embeddings
    input_tokens = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = model(**input_tokens)

    # Extract embeddings (CLS token embedding for BERT)
    embeddings = output.last_hidden_state.mean(dim=1).squeeze().numpy()

    return embeddings

user_query = "Did Blag email me about a meeting?"
query_embedding = create_query_vector(user_query)

# Search for contexts in Pinecone
def context_search(query_embedding):
    search_response = pinecone.query(
        index_name="email-gpt-2",
        query_request={
            "vector": query_embedding.tolist(),
            "top_k": 2,
            "include_metadata": True
        }
    )
    print("Search Response:", search_response)
    matches = search_response.get("matches", [])
    contexts = [match["metadata"]["snippet"] for match in matches]
    print("Contexts:", contexts)
    return contexts

contexts = context_search(query_embedding)

# Create a prompt with contexts
def create_prompt(contexts, user_query):
    prompt_start = "Answer the question based on the context below.\n\nContext:\n"
    prompt_end = f"\n\nQuestion: {user_query}\nAnswer:"

    joined_contexts = "\n\n---\n\n"
    limit = 3750

    for context in contexts:
        if len(joined_contexts) + len(context) >= limit:
            print("Total Length:", len(joined_contexts) + len(context[:contexts.index(context)]))
            break
        joined_contexts += context + "\n\n---\n\n"

    query_with_contexts = prompt_start + joined_contexts + prompt_end
    print(query_with_contexts)
    return query_with_contexts

prompt = create_prompt(contexts, user_query)

# Generate a response using the modified prompt
def send_prompt(prompt):
    # Load the pre-trained GPT-2 model and tokenizer
    model_name = "gpt2"  # You can use other models like "gpt2-medium", "gpt2-large", etc.
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Encode the prompt text
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Generate text based on the prompt
    with torch.no_grad():
        output = model.generate(input_ids, max_length=150, top_p=1.0, pad_token_id=50256)  # 50256 is the id for the [PAD] token in GPT-2

    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    print(generated_text)

    return generated_text

answer = send_prompt(prompt)
