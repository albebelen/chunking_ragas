from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings

def semantic_chunking(document, model="qwen3-embedding:0.6b"): 
    embeddings = OllamaEmbeddings(model=model)

    text_splitter = SemanticChunker(
        embeddings, 
        breakpoint_threshold_type="percentile" #??
    )
    
    chunks = text_splitter.create_documents([document])    
    return chunks[0].page_content