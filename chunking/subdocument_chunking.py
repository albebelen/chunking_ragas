from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid

def subdocument_chunking(document, document_path,parent_size=2000, child_size=400):    
    # splits by Article headers to keep logic intact
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_size,
        chunk_overlap=0,
        separators=["Article ", "\n\n", "\n", " "]
    )
    
    # creates smaller windows for better retrieval
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_size,
        chunk_overlap=50
    )
    
    parent_docs = parent_splitter.split_text(document)
    chunks = []

    for parent_text in parent_docs:
        # Generate a unique ID to link children back to this parent
        parent_id = str(uuid.uuid4())
        
        # Create child chunks from this specific parent
        child_chunks = child_splitter.split_text(parent_text)
        
        for child_text in child_chunks:
            chunks.append({
                "chunk_content": child_text,
                "metadata": {
                    "parent_id": parent_id,
                    "full_context": parent_text,  # This is passed to the LLM
                    "source": document_path
                }
            })
            
    return chunks