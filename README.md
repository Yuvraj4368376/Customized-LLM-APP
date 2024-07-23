# Sustainable(ish) Living Advisor 🌿🏡

## Overview

The Sustainable(ish) Living Advisor is a Retrieval-Augmented Generation (RAG) chatbot that provides practical, achievable advice on eco-friendly practices for everyday life. Based on 'The Sustainable(ish) Living Guide' by Jen Gale, this application demonstrates how RAG can enhance a language model's performance by incorporating external knowledge.

## How RAG Enhances LLM Performance

RAG improves language models by augmenting them with external documents. It retrieves relevant information based on user queries and combines it with the original prompt before generating responses. This approach ensures access to up-to-date and domain-specific information without extensive retraining.

The RAG process involves:

1. **Input**: The user's question.
2. **Indexing**: Related documents are chunked, embedded, and indexed in a vector store.
3. **Retrieval**: Relevant documents are obtained by comparing the query against indexed vectors.
4. **Generation**: Retrieved documents are combined with the original prompt as context for response generation.

## Features

- Interactive chatbot interface powered by Gradio
- RAG implementation using FAISS for efficient document retrieval
- PDF processing and text extraction using PyMuPDF
- Sentence embeddings with SentenceTransformer (all-MiniLM-L6-v2)
- Integration with Hugging Face's Inference API (Zephyr 7B Beta model)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/sustainable-living-advisor.git
   cd sustainable-living-advisor
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Ensure you have the PDF file "SustainableishLivingGuide.pdf" in the same directory as the script.

## Usage

Run the application:

```
python app.py
```

Open your web browser and navigate to the URL provided in the console (usually `http://127.0.0.1:7860`).

## Example Queries

- How can I start reducing my plastic usage?
- What are some easy ways to save energy at home?
- Can you suggest sustainable alternatives for everyday products?
- What are some tips for reducing food waste?
- How can I make my cleaning routine more eco-friendly?
- How can I involve my kids in sustainable living?

## Components

1. **Knowledgebase**: 'The Sustainable(ish) Living Guide' by Jen Gale (PDF)
2. **requirements.txt**: Lists necessary Python packages
3. **app.py**: Main application file
4. **Hugging Face Account**: Required for accessing the Inference API

## Disclaimer

This chatbot is intended for educational purposes and to demonstrate RAG implementation. For personalized advice, please consult with environmental experts.

## Dependencies

- gradio
- huggingface_hub
- PyMuPDF
- sentence-transformers
- numpy
- faiss-cpu

## Acknowledgements

- Jen Gale for 'The Sustainable(ish) Living Guide'
- Hugging Face for the Inference API and model hosting
- The open-source community for the amazing tools and libraries used in this project
sed (Zephyr LLM and all-MiniLM-L6-v2 sentence transformer). It also retains the essential information about the project setup, usage, and other relevant details.
