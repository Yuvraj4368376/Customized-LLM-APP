import gradio as gr
from huggingface_hub import InferenceClient
from typing import List, Tuple
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util
import numpy as np
import faiss

client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

# Placeholder for the app's state
class SustainableLivingApp:
    def __init__(self) -> None:
        self.documents = []
        self.embeddings = None
        self.index = None
        self.load_pdf("SustainableishLivingGuide.pdf")
        self.build_vector_db()

    def load_pdf(self, file_path: str) -> None:
        """Extracts text from a PDF file and stores it in the app's documents."""
        doc = fitz.open(file_path)
        self.documents = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            self.documents.append({"page": page_num + 1, "content": text})
        print("Sustainable(ish) Living Guide processed successfully!")

    def build_vector_db(self) -> None:
        """Builds a vector database using the content of the PDF."""
        model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = model.encode([doc["content"] for doc in self.documents])
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(np.array(self.embeddings))
        print("Sustainable living vector database built successfully!")

    def search_documents(self, query: str, k: int = 3) -> List[str]:
        """Searches for relevant documents using vector similarity."""
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode([query])
        D, I = self.index.search(np.array(query_embedding), k)
        results = [self.documents[i]["content"] for i in I[0]]
        return results if results else ["No relevant information found."]

app = SustainableLivingApp()

def respond(
    message: str,
    history: List[Tuple[str, str]],
    system_message: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
):
    system_message = "You are a knowledgeable Sustainable Living Advisor based on 'The Sustainable(ish) Living Guide' by Jen Gale. You provide practical, achievable advice on eco-friendly practices for everyday life. Discuss one sustainable practice at a time, be concise, and avoid long responses. Ask questions like a real environmental consultant, be a good listener, and use verbal cues sparingly. Consider the users as your clients seeking guidance on making sustainable choices. When needed, ask one follow-up question to guide the conversation. If any dangerous activities are mentioned, avoid giving suggestions and recommend seeking expert advice."
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    # RAG - Retrieve relevant documents
    retrieved_docs = app.search_documents(message)
    context = "\n".join(retrieved_docs)
    messages.append({"role": "system", "content": "Relevant information: " + context})

    response = ""
    for message in client.chat_completion(
        messages,
        max_tokens=100,
        stream=True,
        temperature=0.98,
        top_p=0.7,
    ):
        token = message.choices[0].delta.content
        response += token
        yield response

demo = gr.Blocks()

with demo:
    gr.Markdown(
        "‼️Disclaimer: This chatbot is based on 'The Sustainable(ish) Living Guide' by Jen Gale. It's intended for educational purposes and to demonstrate RAG implementation. For personalized advice, please consult with environmental experts.‼️"
    )
    
    chatbot = gr.ChatInterface(
        respond,
        examples=[
            ["How can I start reducing my plastic usage?"],
            ["What are some easy ways to save energy at home?"],
            ["Can you suggest sustainable alternatives for everyday products?"],
            ["What are some tips for reducing food waste?"],
            ["How can I make my cleaning routine more eco-friendly?"],
            ["How can I involve my kids in sustainable living?"]
        ],
        title='Sustainable(ish) Living Advisor 🌿🏡'
    )

if __name__ == "__main__":
    demo.launch()