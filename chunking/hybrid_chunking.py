## use recursive chunking to "clean" file
## give clean file to semantic

from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

OLLAMA_EMBEDDING_MODEL = "qwen3-embedding:0.6b"

def hybrid_chunking(document, max_chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["TITOLO", "Articolo"],
        chunk_size = max_chunk_size,
        chunk_overlap=chunk_overlap, # overlap chunk to mitigate loss of info
        length_function=len,
        is_separator_regex=False
    )

    chunks = text_splitter.split_text(document)
    
    # semantic, 
    embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)

    text_splitter = SemanticChunker(
        embeddings, 
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95.0
    )
    
    docs = text_splitter.create_documents(chunks)
    
    return [{
            "chunk_content": d.page_content,
            "metadata": {"source": "docs/CELEX_32006L0054_IT_TXT.pdf"}
        }
        for d in docs]