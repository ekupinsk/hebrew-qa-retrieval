---
title: Hebrew QA Search
emoji: 🔎
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "6.30.0"
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Hebrew question answering with hybrid search and a reranker
models:
  - ElyKok/hebrew-qa-retriever-e5-large-nofilter
  - ElyKok/hebrew-qa-reranker
  - intfloat/multilingual-e5-large
  - dicta-il/dictabert-heq
  - dicta-il/DictaLM-3.0-1.7B-Instruct
datasets:
  - ElyKok/hebrew-qa-retrieval-data
---

# Hebrew QA Search

Ask a question in Hebrew and get the paragraph that answers it, and the answer itself.
Hybrid search (BM25 + a fine-tuned multilingual-e5-large) returns 20 candidates, a fine-tuned reranker
orders them, and DictaBERT-HeQ extracts the answer. You can also upload your own document and ask about it.

Full project, notebooks and results: [GitHub](https://github.com/ekupinsk/hebrew-qa-retrieval)
