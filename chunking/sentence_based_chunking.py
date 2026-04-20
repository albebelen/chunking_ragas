import nltk

nltk.download('punkt_tab')

def sentence_chunking(document):
    chunks = nltk.sent_tokenize(document.strip())
    return chunks