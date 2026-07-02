# tests/test_retrieval.py
from agent.retrieval import HybridRetriever
import json

# Test queries from conversation traces
test_queries = [
    {"role": "Java developer", "skills": ["java"], "test_types": ["K"]},
    {"role": "Python developer", "skills": ["python", "aws"], "test_types": ["K", "P"]},
    {"skills": ["leadership"], "test_types": ["P"]},
]

retriever = HybridRetriever()

for i, context in enumerate(test_queries):
    results = retriever.search(context, k=5)
    print(f"\nQuery {i+1}: {context}")
    for j, r in enumerate(results):
        print(f"  {j+1}. {r['name']} (type: {r['test_type']}, score: {r['score']:.3f})")