def paragraph_chunking(document):

    chunks = [p.strip() for p in document.split('CAPITOLO') if len(p.strip()) > 100]
    return chunks