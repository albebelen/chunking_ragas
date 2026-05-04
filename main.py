from ragas import evaluate 
from ragas.metrics import context_precision, context_recall, faithfulness
from datasets import Dataset
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from time import time 
import os 

from funcs.utils import read_document, retrieve_top_k
from questions.questions_set_eng import questions

from chunking.paragraph_based_chunking import paragraph_chunking

OLLAMA_CLOUD_URL = "https://ollama.com"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_BASE_MODEL = "gpt-oss:120b-cloud"
OLLAMA_EMBEDDING_MODEL = "qwen3-embedding:0.6b"
OLLAMA_API_KEY = "c036e8dbcabc49e28bcdeb7ca52cb800.IIPzB42NuCYS0gXLELXKzjtz"

llm = Ollama(
    model=OLLAMA_BASE_MODEL,
    base_url=OLLAMA_CLOUD_URL,
    temperature=0,
    headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"}
)
embeddings = OllamaEmbeddings(
    model=OLLAMA_EMBEDDING_MODEL, 
    base_url=OLLAMA_BASE_URL,
)

document = read_document('docs/CELEX_32006L0054_IT_TXT.pdf')

start_time = time()

chunks = paragraph_chunking(document)

''' region old params
questions = [
    "What is the purpose of the directive DIRECTIVE 2006/54/EC?",
    "How many articles are in the directive?",
    "What specific employment rights are guaranteed to a woman returning from maternity leave under the provisions of this Directive?",
    "What is the difference between direct and indirect discrimination?",
    "According to Directive 2006/54/EC, can Member States place a prior upper limit on the compensation or reparation awarded to a victim of sex discrimination?"
]

ground_truths = [
    "The purpose of this Directive is to ensure the implementation of the principle of equal opportunities and equal treatment of men and women in matters of employment and occupation.",
    "There are 36 articles in total.",
    "A woman on maternity leave is entitled to return to her job or to an equivalent post at the end of her leave period. This return must be on terms and conditions that are no less favourable to her. Furthermore, she is entitled to benefit from any improvement in working conditions to which she would have been entitled had she not been absent.",
    "direct discrimination is when one person is treated less favourably on grounds of sex than another is, has been or would be treated in a comparable situation; whereas indirect discrimination " +
    "is when an apparently neutral provision, criterion or practice would put persons of one sex at a particular disadvantage compared with persons of the other sex, unless that provision, criterion " + 
    "or practice is objectively justified by a legitimate aim, and the means.",
    "Member States must ensure real and effective compensation or reparation for loss and damage sustained by a person injured as a result of sex discrimination. Such compensation may not be restricted by a prior upper limit. The only exception to this rule is in cases where the employer can prove that the only damage suffered by the applicant was the refusal to take his or her job application into consideration."
]

answers = []

contexts = []

'''

for q in questions:
    retrieved = retrieve_top_k(chunks, q["question"], embeddings, k=5)
    q["contexts"].append(retrieved)

    provided_context = "\n\n".join(retrieved)

    prompt = f"""
    Answer only based on provided context.

    Answer format: {q["answer_type"]}

    Context: {provided_context}

    Question: {q["question"]}
    """

    #come aggiungo l'articolo di riferimento?

    response = llm.invoke(prompt)

    if isinstance(response, str):
        q["answer"] = response.strip()
    else:
        q["answer"] = str(response).strip()

    print("Question: " + q["question"] + "\n")
    print("Answer: " + response + "\n")

stop_time = time()
elapsed_time = stop_time - start_time

dataset = Dataset.from_dict({
    "user_input": [q["question"] for q in questions],
    "question": [q["question"] for q in questions],
    "answer": [q["answer"] for q in questions],
    "ground_truth": [q["ground_truth"] for q in questions],
    "contexts": [[str(c) for c in q["contexts"]] for q in questions],
    "metadata": [{"difficulty": q["difficulty"], "adversarial": q["adversarial"], "language": "IT"} for q in questions]
})

result = evaluate(
    dataset,
    metrics=[context_precision, context_recall, faithfulness],
    llm=llm,
    embeddings=embeddings,
    raise_exceptions=False
)

i = 1

with open('outputs/paragraph_based_output.txt', 'w') as output:
    for chunk in chunks:
        output.write('chunk ' + str(i) + ' : ' + chunk  + '\n')
        i+= 1

file_size = os.path.getsize('outputs/paragraph_based_output.txt')

print(result)
print(f"Elapsed time: {elapsed_time:.2f} seconds")
print("File size: " + str(file_size))


''' other metrics
prompt = f'Give me a short evaluation of the chunking strategy used in this app based on chunk sizes (what kind of consequences does it have on memory efficiency and retrieval latency?). " \
    "Also does each chunk make sense on their own and are the full answer contained inside a single chunk? Here are the chunks: {chunks}, the questions {questions} and the answers: {answers} " \
    "and the contexts: {contexts}'

response_extra = llm.invoke(prompt)

print("Extra metrics \n" + response_extra)
'''