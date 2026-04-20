def sliding_window_chunking(document, window_size = 1000, step_size = 500):
    chunks = []
    start = 0
    text_len = len(document)

    while start < text_len:
        end = start + window_size
        chunk = document[start:end]
        chunks.append(chunk.strip())

        start += step_size

        if start >= text_len - (window_size // 4):
            break

    return chunks