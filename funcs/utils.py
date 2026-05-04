from PyPDF2 import PdfReader
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def read_document(path):
    text = ''
    with open(path, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + " "
    return text

''' this method simulates retrieval from a vector database
1. converts texts into vectors
2. converts question into a numerical representation
3. cosine_similarity computes the similarity betweeb query and each chunks
4. retrieve the top k  chunks 
returns the best chunks
# '''

# use for other chunking except subdoc and pageindex
def retrieve_top_k(chunks, query, embeddings, k=5):
    chunk_embeds = embeddings.embed_documents(chunks)
    query_embed = embeddings.embed_query(query)

    sims = cosine_similarity([query_embed], chunk_embeds)[0]
    top_k_idx = np.argsort(sims)[-k:][::-1]

    return [chunks[i] for i in top_k_idx]    

def retrieve_top_k_alt(chunks, query, embeddings, k=5):
    chunks_to_embed = [c["chunk_content"] for c in chunks]
    chunk_embeds = embeddings.embed_documents(chunks_to_embed)
    query_embed = embeddings.embed_query(query)

    sims = cosine_similarity([query_embed], chunk_embeds)[0]
    top_k_idx = np.argsort(sims)[-k:][::-1]

    return [chunks[i] for i in top_k_idx]

