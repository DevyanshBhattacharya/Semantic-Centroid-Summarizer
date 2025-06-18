# Semantic-Centroid-Summarizer

## Overview

**Semantic-Centroid-Summarizer** is a powerful and scalable text summarization pipeline designed to handle large documents such as research papers, reports, and books. It combines semantic embeddings, clustering, and large language model (LLM) summarization to produce concise and coherent summaries by focusing on the most representative parts of the document.

---

## Pipeline Architecture

### 1. Document Loading & Chunking
- **Library:** LangChain’s `PyPDFLoader` and `RecursiveCharacterTextSplitter`
- **Description:** Extracts raw text from PDF files and splits it into overlapping chunks of manageable size.
- **Purpose:** Enables processing of large documents without exceeding LLM context window limits, while preserving semantic continuity.

### 2. Semantic Embeddings
- **Model:** `all-MiniLM-L6-v2` from SentenceTransformers
- **Description:** Transformer-based sentence embedding model producing fixed-size dense vectors that capture semantic meaning of each chunk.
- **Purpose:** Provides a numerical representation of text enabling similarity comparison and clustering.

### 3. Clustering
- **Algorithm:** KMeans clustering from scikit-learn
- **Description:** Groups semantically similar chunks based on their embedding vectors.
- **Purpose:** Identifies thematic clusters to organize document content and reduce redundancy.

### 4. Centroid Selection
- **Method:** For each cluster, selects the chunk closest to the cluster centroid by minimizing cosine distance.
- **Purpose:** Represents each cluster by its most central, and thus most informative, text chunk.

### 5. Summarization
- **Model:** Google Gemini 2.5 Flash (via Google Generative AI API)
- **Description:** An LLM that generates fluent, abstractive summaries in bullet points from representative chunks.
- **Purpose:** Produces concise, human-readable summaries focusing on key insights.

### 6. (Optional) Global Summary
- Concatenates cluster summaries to generate an overall summary of the entire document.

---

## Why Semantic-Centroid-Summarizer?

- **Scalable:** Effectively processes large documents by dividing them into semantically coherent parts.
- **Efficient:** Summarizes only representative chunks, reducing API calls and computational cost.
- **Accurate:** Uses semantic embeddings and clustering to preserve meaning and topical relevance.
- **Modular:** Each component can be independently improved or swapped with alternative models.

---

## Usage

1. Load a PDF document using LangChain.
2. Split text into chunks with semantic overlap.
3. Compute embeddings for each chunk.
4. Cluster the embeddings with KMeans.
5. Identify representative chunks by proximity to cluster centroids.
6. Summarize these chunks using Gemini.
7. Optionally, generate a global summary from the cluster summaries.

---

## Requirements

- Python 3.8+
- `langchain`, `langchain-community`, `google-generativeai`
- `sentence-transformers`
- `scikit-learn`
- `matplotlib` (for visualization)

---

## Future Improvements

- Replace SentenceTransformer embeddings with Gemini’s embedding API for unified model use.
- Implement adaptive clustering algorithms like HDBSCAN for dynamic cluster count.
- Develop a user-friendly UI for interactive summarization.
- Add hybrid extractive-abstractive summarization approaches.

