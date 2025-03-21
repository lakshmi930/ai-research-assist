from fastapi import FastAPI, UploadFile, File, HTTPException, Form
import fitz  # PyMuPDF
import subprocess

app = FastAPI()

# Store extracted text in memory
document_text = ""

@app.get("/")
def home():
    return {"message": "AI Research Assistant is Running!"}

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...), topic: str = Form(None)):
    """Uploads a PDF file and summarizes it."""
    global document_text
    global topic_text
    try:
        pdf_reader = fitz.open(stream=await file.read(), filetype="pdf")
        text = ""
        for page in pdf_reader:
            text += page.get_text()

        if len(text) < 100:
            raise HTTPException(status_code=400, detail="PDF text is too short for analysis.")

        document_text = text  # Store text for Q&A
        topic_text = topic
        summary = summarize_text(text, topic)

        return {"filename": file.filename, "summary": summary}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def summarize_text(text, topic):
    """Calls Llama 3.2 via Ollama for summarization."""
    try:
        if topic:
            command = f'ollama run llama3.2 "You are my research assistant.\nSummarize this research paper concisely with the topic-{topic} relevancy:\n{text[:3000]}"'
        else:
            print("No topic")
            command = f'ollama run llama3.2 "Summarize this research paper concisely:\n{text}"'
        result = subprocess.run(command, shell=True, capture_output=True, text=True)

        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return f"Error: {result.stderr.strip()}"

    except Exception as e:
        return f"Exception: {str(e)}"

@app.post("/ask/")
def ask_question(question: str = Form(...)):
    """Allows users to ask questions about the uploaded research paper."""
    global document_text
    global topic_text
    if not document_text:
        raise HTTPException(status_code=400, detail="No document uploaded. Please upload a PDF first.")

    return {"answer": ask_llama(document_text, question, topic_text)}

def ask_llama(text, question, topic):
    """Calls Llama 3.2 via Ollama to answer questions about the document."""
    try:
        prompt = f"Based on this research paper, answer the following question:\n\n{text[:3000]}\n\nQuestion: {question}. {"Keep the answer relevant to {topic}." if topic else ""}"
        command = f'ollama run llama3.2 "{prompt}"'
        result = subprocess.run(command, shell=True, capture_output=True, text=True)

        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return f"Error: {result.stderr.strip()}"

    except Exception as e:
        return f"Exception: {str(e)}"
    
@app.post("/keywords/")
def get_keywords():
    """Allows users to find the keywords about the uploaded research paper."""
    global document_text
    global topic_text
    if not document_text:
        raise HTTPException(status_code=400, detail="No document uploaded. Please upload a PDF first.")

    return {"answer": ask_llama(document_text, "What are the keywords relevant to the research? Order them in decreasing order of frequency. Do not include random words just because they could be commonly occuring. Also, include the frequency of each keyword. Mention the keywords (if exists) and the ones that are extracted from the text.", topic_text)}
