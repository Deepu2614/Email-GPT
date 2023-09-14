import os
import pinecone
import nylas
from nylas import APIClient
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables from .env file
load_dotenv()

pinecone.init(api_key=str(os.getenv("PINECONE_API_KEY")), environment="gcp-starter")
index_name = "email-gpt"
index = pinecone.Index(index_name)



# Retrieve Nylas API credentials from environment variables
nylas_client_id = os.getenv("NYLAS_CLIENT_ID")
nylas_client_secret = os.getenv("NYLAS_CLIENT_SECRET")
access_token = os.getenv("NYLAS_ACCESS_TOKEN")

# Initialize the Nylas API client
nylas = APIClient(nylas_client_id, nylas_client_secret, access_token=access_token)

def retrieve_messages():
    # Retrieve a list of the last two messages
    messages = nylas.messages.where(limit=2)
    
    # Initialize an empty list to store message data
    message_data = []
    
    for message in messages:
        metadata = {
            'snippet': message['snippet'],
            'fromName': message['from'][0]['name'] if message.get('from') else None,
            'fromEmail': message['from'][0]['email'] if message.get('from') else None,
            'toName': message['to'][0]['name'] if message.get('to') else None,
            'toEmail': message['to'][0]['email'] if message.get('to') else None,
            'subject': message['subject'],
            'date': message['date'],
            'id': message.id,
        }
        message_data.append({'message': message, 'metadata': metadata, 'id': message.id})

    return message_data





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

    return embeddings  # Make sure it returns NumPy array





pinecone.init(api_key=str(os.getenv("PINECONE_API_KEY")), environment="gcp-starter")


async def store_vector(index, embedding, id, metadata):
    # Upsert the vector data
    upsert_response = index.upsert(
        [
            {
                "id": id,
                "values": np.array(embedding).tolist(),  # Convert NumPy array to a list
                "metadata": metadata,
            }
        ]
    )
    # print("Upsert Response:", upsert_response)

    # Get index stats
    index_stats = index.describe_index_stats()
    # print("Index Stats:", index_stats)



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


async def context_search(query_embedding, index):
    # Search the Pinecone index
    search_response = await index.query(
        {
            "vector": query_embedding.tolist(),  # Convert NumPy array to a list
            "top_k": 2,  # Set the top_k value to 2
            "include_metadata": True,
        }
    )

    # Log out the possible search responses
    # print("Search Response:", search_response)

    # Retrieve the matches from the search response
    matches = search_response.get("matches", [])

    # Retrieve the contexts from the matches
    contexts = [result["metadata"]["snippet"] for result in matches]

    # print("Contexts:", contexts)

    return contexts


def create_prompt(similarity_data, user_query):
    # Create the prompts
    prompt_start = "Answer the question based on the context below.\n\nContext:\n\n"

    # Iterate through similarity_data and append it to prompt_start
    for score, metadata in similarity_data:
        snippet = metadata.get('snippet', 'N/A')
        prompt_start += f"---\n\nSimilarity Score: {score}\nEmail Snippet: {snippet}\n\n"

    prompt_end = f"Question: {user_query}\nAnswer:\n"

    limit = 3750

    # Ensure that the total length of the prompt does not exceed the limit
    if len(prompt_start) + len(prompt_end) > limit:
        print("Total length exceeds the limit.")

    # Create the final prompt
    query_with_contexts = prompt_start + prompt_end

    return query_with_contexts

########################

# NOT WORKING DA... generate_answer_from_prompt

########################

# def generate_answer_from_prompt(prompt):
#     # Load the pre-trained GPT-2 model and tokenizer
#     model_name = "gpt2"  # You can use other models like "gpt2-medium", "gpt2-large", etc.
#     model = GPT2LMHeadModel.from_pretrained(model_name)
#     tokenizer = GPT2Tokenizer.from_pretrained(model_name)

#     # Encode the prompt text
#     input_ids = tokenizer.encode(prompt, return_tensors="pt")

#     # Generate text based on the prompt
#     with torch.no_grad():
#         output = model.generate(input_ids, max_length=150, top_p=1.0, pad_token_id=50256)  # 50256 is the id for the [PAD] token in GPT-2

#     # Decode the generated text and return it
#     generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

#     return generated_text





########################################################################################################################################################################################################

# ...
# ...

async def main():
    # Call the retrieve_messages function
    message_data = retrieve_messages()
    print(type(message_data))

    message_data = retrieve_messages()

    # Initialize a list to store embeddings for each message
    embeddings_list = []

    your_data_list = []  # Create a list to store data items (including embedding and metadata)

    # Iterate through each message in message_data
    for message in message_data:
        metadata = message['metadata']  # Get the metadata dictionary
        if 'snippet' not in metadata:
            print(f"Skipping message without snippet: {metadata}")
            continue
        
        input_text = metadata['snippet']
        embedding = create_embeddings(input_text)
        embeddings_list.append(embedding)

        id = metadata['id']
        
        data_item = {
            "embedding": embedding.tolist(),  # Convert NumPy array to a list
            "id": id,
            "metadata": metadata,
        }

        your_data_list.append(data_item)

    # Assuming you have a list of embeddings, ids, and metadata
    for data_item in your_data_list:
        await store_vector(index, data_item["embedding"], data_item["id"], data_item["metadata"])

    user_query = "Yahoo users need to authenticate using an app password instead of their email password."
    query_embedding = create_query_vector(user_query)

    def similarity_check(query_embedding, embeddings_list):
        # Calculate cosine similarity between the query embedding and all email embeddings
        similarity_scores = cosine_similarity(query_embedding.reshape(1, -1), embeddings_list)

        # Create a list of tuples containing (similarity_score, email_metadata)
        similarity_data = [(score, data_item["metadata"]) for score, data_item in zip(similarity_scores[0], your_data_list)]

        # Sort the similarity data by similarity score in descending order
        similarity_data.sort(key=lambda x: x[0], reverse=True)

        return similarity_data
    
    similarity_data = similarity_check(query_embedding, embeddings_list)
    
    print(similarity_data)

    # for score, metadata in similarity_data:
    #     print(f"Similarity Score: {score}")
    #     print(f"Email Snippet: {metadata.get('snippet', 'N/A')}")
    #     print(f"Email Subject: {metadata.get('subject', 'N/A')}")
    #     print("\n")

    # Extract snippets from similarity data and join them
    # context_snippets = [metadata.get('snippet', '') for _, metadata in similarity_data]
    # combined_contexts = "\n\n---\n\n".join(context_snippets)

    # print(combined_contexts)

    # prompt = create_prompt(similarity_data, user_query)
    # print(prompt)
    
    # Generate a response from GPT-2
    # generated_response = generate_answer_from_prompt(prompt)

    # print("Generated Response:", generated_response)

# Run the main coroutine
if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
