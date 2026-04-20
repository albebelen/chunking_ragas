
def fixed_size_chunking(document, chunk_size):
    return [document[i:i+chunk_size] for i in range(0, len(document), chunk_size)]