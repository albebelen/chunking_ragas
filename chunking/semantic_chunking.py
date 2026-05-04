from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings

def semantic_chunking(document, model="qwen3-embedding:0.6b"): 
    embeddings = OllamaEmbeddings(model=model)

    text_splitter = SemanticChunker(
        embeddings, 
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95.0 # ???
    )
    
    chunks = text_splitter.create_documents([document])    
    
    #return chunks[0].page_content
    return [{
            "chunk_content": c.page_content,
            "metadata": {"source": "docs/CELEX_32006L0054_IT_TXT.pdf"}
        }
        for c in chunks]