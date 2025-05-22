from sentence_transformers import SentenceTransformer
import time

sentences = [
    "BERT is a powerful transformer model.",
    "Embedding sentences helps in many NLP tasks.",
    "Docker makes ML deployment easier."
]

# Fast and accurate transformer
model = SentenceTransformer('all-MiniLM-L6-v2')
start_time = time.time()

embeddings = model.encode(sentences)
end_time = time.time()
elapsed = end_time - start_time
for sentence, embedding in zip(sentences, embeddings):
    print(f"Sentence: {sentence}")
    print(f"Embedding (first 5 dims): {embedding[:5]}")
    print("------")
print(f"Time taken to encode {len(sentences)} sentences: {elapsed:.3f} seconds")
