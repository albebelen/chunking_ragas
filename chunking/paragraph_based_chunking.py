def paragraph_chunking(document):

    chunks = [p.strip() for p in document.split('CHAPTER') if len(p.strip()) > 100]
    return chunks