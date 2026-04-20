from ragas import evaluate 
from ragas.metrics import context_precision, context_recall
from datasets import Dataset
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama

from funcs.utils import read_document, retrieve_top_k
from chunking.context_chunking import context_enriched_chunking

OLLAMA_CLOUD_URL = "https://ollama.com"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_BASE_MODEL = "gpt-oss:120b-cloud"
OLLAMA_EMBEDDING_MODEL = "qwen3-embedding:0.6b"
OLLAMA_API_KEY = "79dd5063a8764b26ab2ce8892f76ae1b.BT4rLtrSGsWDf7jPWQO9efht"

llm = Ollama(
    model=OLLAMA_BASE_MODEL,
    base_url=OLLAMA_CLOUD_URL,
    temperature=0,
    headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"}
)
embeddings = OllamaEmbeddings(
    model=OLLAMA_EMBEDDING_MODEL, 
    base_url=OLLAMA_BASE_URL
)

document = read_document('docs/CELEX_32006L0054_EN_TXT.pdf')
chunks = context_enriched_chunking(document)

questions = [
    "What is the purpose of the directive DIRECTIVE 2006/54/EC?",
    "How many articles are in the directive",
    "What does Article 15 say about return from maternity leave?"
]

ground_truths = [
    "The purpose of this Directive is to ensure the implementation of the principle of equal opportunities and equal treatment of men and women in matters of employment and occupation.",
    "There are 36 articles in total.",
    "A woman on maternity leave shall be entitled, after the end of her period of maternity leave, to return to her job or to an equivalent post on terms and conditions which are no less favourable to her and to benefit from any improvement in working conditions to which she would have been entitled during her absence."
]

answers = []

contexts = []

for q in questions:
    retrieved = retrieve_top_k(chunks, q, embeddings, k=5)
    contexts.append(retrieved)

    provided_context = "\n\n".join(retrieved)

    prompt = f"""
    Answer only based one provided context.

    Context: {provided_context}

    Question: {q}
    """

    response = llm.invoke(prompt)

    if isinstance(response, str):
        answers.append(response.strip())
    else:
        answers.append(str(response).strip())

    print("Question: " + q + "\n")
    print("Answer: " + response + "\n")

dataset = Dataset.from_dict({
    "user_input": questions,
    "question": questions,
    "answer": answers,
    "ground_truth": ground_truths,
    "contexts": contexts
})

result = evaluate(
    dataset,
    metrics=[context_precision, context_recall],
    llm=llm,
    embeddings=embeddings,
    raise_exceptions=False
)

print(result)

i = 1

with open('outputs/context_chunking.txt', 'w') as output:
    for chunk in chunks:
        output.write('chunk ' + str(i) + ' : ' + chunk + '\n')
        i+= 1

