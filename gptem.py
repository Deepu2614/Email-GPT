import os
import pinecone
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel, GPT2LMHeadModel, GPT2Tokenizer
import torch
import numpy as np

# Load environment variables from .env file
load_dotenv()

# Function to create embeddings using BERT
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

# Function to set up Pinecone
api_key = os.getenv("PINECONE_API_KEY")
pinecone.init(api_key=api_key, environment="gcp-starter")

async def setup_pinecone():
    # List existing indexes
    index_list = pinecone.list_indexes()
    print("Indexes:", index_list)

    # Set up the Pinecone vector index
    index_name = "email-gpt"
    index = pinecone.Index(index_name)

    return index

# Function to store vectors in Pinecone
async def store_vector(index, embedding, id, metadata):
    # Upsert the vector data
    upsert_response = index.upsert(
        [
            {
                "id": id,
                "values": embedding.tolist(),  # Convert NumPy array to a list
                "metadata": metadata,
            }
        ]
    )
    print("Upsert Response:", upsert_response)

    # Get index stats
    index_stats = index.describe_index_stats()
    print("Index Stats:", index_stats)

# Function to create a query vector using BERT
def create_query_vector(query):
    model_name = "bert-base-uncased"  # Specify the BERT model name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Tokenize and convert the query text to embeddings
    input_tokens = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = model(**input_tokens)

    # Extract embeddings (CLS token embedding for BERT)
    embeddings = output.last_hidden_state.mean(dim=1).squeeze().numpy()

    return embeddings

# Function to search for context in Pinecone
async def context_search(query_embedding, index):
    # Search the Pinecone index
    search_response = await index.query(
        {
            "vector": query_embedding.tolist(),  # Convert NumPy array to a list
            "top_k": 5,  # Adjust this value as needed
            "include_metadata": True,
        },
        top_k=5  # Ensure that the top_k parameter is correctly added
    )

    # Log out the possible search responses
    print("Search Response:", search_response)

    # Retrieve the matches from the search response
    matches = search_response.get("matches", [])

    # Retrieve the contexts from the matches
    contexts = [result["metadata"]["snippet"] for result in matches]

    print("Contexts:", contexts)

    return contexts

# Function to create a prompt for GPT-2
def create_prompt(contexts, user_query):
    # Create the prompts
    prompt_start = """
    Answer the question based on the context below.

    Context:
    """
    
    prompt_end = f"""
    Question: {user_query}
    Answer:
    """
    
    joined_contexts = "\n\n---\n\n"
    limit = 3750
    
    # Join the contexts together
    for context in contexts:
        if len(joined_contexts) + len(context) >= limit:
            print("Total length:", len(joined_contexts) + len(context[:contexts.index(context)]))
            break
        joined_contexts += context + "\n\n---\n\n"
    
    # Create the final prompt
    query_with_contexts = prompt_start + joined_contexts + prompt_end
    
    print(query_with_contexts)
    
    return query_with_contexts

# Function to send the prompt to GPT-2 and get a response
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

# Main function to execute the entire process
async def main():
    # Assuming you have a list of messages with text to be embedded
    messages = [
        "This is the first message.",
        "This is the second message."
    ]

    # Initialize the Pinecone index
    pinecone_index = await setup_pinecone()

    # Initialize a list to store embeddings for each message
    embeddings_list = []

    # Iterate through each message in messages
    for message in messages:
        try:
            # Generate embeddings for the current message
            embedding = create_embeddings(message)
            
            # Append the embedding to the embeddings_list
            embeddings_list.append(embedding)
        except Exception as e:
            print(f"Error processing message: {e}")
            continue  # Skip this message and continue with the next one

    your_data_list = []

    for embedding, message in zip(embeddings_list, messages):
        # Extract relevant information from message and metadata
        id = message[:10]  # You can use a unique ID for each message
        metadata = {"snippet": message}

        # Create a dictionary for the current data item
        data_item = {
            "embedding": embedding.tolist(),  # Convert NumPy array to a list
            "id": id,
            "metadata": metadata,
        }

        # Append the data item to your_data_list
        your_data_list.append(data_item)

    # Assuming you have a list of embeddings, ids, and metadata
    for data_item in your_data_list:
        embedding = np.array(data_item["embedding"])
        await store_vector(pinecone_index, embedding, data_item["id"], data_item["metadata"])

    ###############################################
    user_query = "This is my query text."
    query_embedding = create_query_vector(user_query)
    ###############################################

    contexts = await context_search(query_embedding, pinecone_index)  # Assuming you have a Pinecone index object

    prompt = create_prompt(contexts, user_query)

    generated_response = send_prompt(prompt)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
