import torch
from transformers import AutoModel, AutoTokenizer

def embed(inp):
    # Input text
    input_text = inp

    # Load pre-trained model and tokenizer
    model_name = "bert-base-uncased"  # You can choose any model from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Tokenize and convert input text to embeddings
    input_tokens = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = model(**input_tokens)
    
    # Extract embeddings (CLS token embedding for BERT)
    embeddings = output.last_hidden_state.mean(dim=1).squeeze().numpy()

    # Print the embeddings (you can save them to a file or use them in your project)
    # print("Text Embeddings:")
    # print(embeddings)
    return embeddings
