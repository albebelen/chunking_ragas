from langchain_text_splitters import RecursiveCharacterTextSplitter

def recursive_chunking(document, max_chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["TITLE", "Article"],
        chunk_size = max_chunk_size,
        chunk_overlap=chunk_overlap, # overlap chunk to mitigate loss of info
        length_function=len,
        is_separator_regex=False
    )

    chunks = text_splitter.split_text(document)
    return chunks