from fastapi import FastAPI, UploadFile, File, HTTPException, Form
import fitz  # PyMuPDF
import subprocess

app = FastAPI()

# Store extracted text in memory
document_text = ""
topic_text = ""

@app.get("/")
def home():
    return {"message": "AI Research Assistant is Running!"}

@app.post("/topic/")
async def get_topic(topic: str = Form(None)):
    """Gets the topic of the researcher."""
    global topic_text
    topic_text = topic
    if (topic_text == "" or topic_text == None):
        return {"message": "Providing a topic helps with personalized responses."}
    else:
        return {"message": "Topic is set"}
    

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    """Uploads a PDF file and summarizes it."""
    global document_text
    try:
        pdf_reader = fitz.open(stream=await file.read(), filetype="pdf")
        text = ""
        for page in pdf_reader:
            text += page.get_text()

        if len(text) < 100:
            raise HTTPException(status_code=400, detail="PDF text is too short for analysis.")

        document_text = text[:3000]  # Store text for Q&A
        summary = summarize_text()

        return {"filename": file.filename, "summary": summary}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def summarize_text():
    """Calls Llama 3.2 via Ollama for summarization."""
    if not document_text:
        raise HTTPException(status_code=400, detail="No document uploaded. Please upload a PDF first.")
    try:
        prompt = f"""
            You are personal research assistant.
            Summarize given research paper concisely.
            {"Keep the answer relevant to {topic_text}." if topic_text else ""}
            Paper content:\n{document_text}
        """
        command = f'ollama run llama3.2 "{prompt}"'
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
    if not document_text:
        raise HTTPException(status_code=400, detail="No document uploaded. Please upload a PDF first.")

    return {"answer": ask_llama(question)}

def ask_llama(question):
    """Calls Llama 3.2 via Ollama to answer questions about the document."""
    if not document_text:
        raise HTTPException(status_code=400, detail="No document uploaded. Please upload a PDF first.")
    try:
        prompt = f"""
            You are personal research assistant.
            Based on this research paper, answer the following question:{document_text}
            Question: {question}. 
            {"Keep the answer relevant to {topic_text}." if topic_text else ""}
        """
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
    """Extracts the keywords from the uploaded research paper."""
    if not document_text:
        raise HTTPException(status_code=400, detail="No document uploaded. Please upload a PDF first.")
    
    prompt = f"""
            You are personal research assistant.
            Extract the keywords from the research paper.
            {"Keep the answer relevant to {topic_text}." if topic_text else ""}
            Avoid random words just because they could be commonly occuring.
            Order the keywords in decreasing order of frequency.
            Include the frequency of each keyword.
            Mention the keywords section (if exists) and the ones that are extracted from the text by you separately.
            Paper content:\n{document_text}
        """
    command = f'ollama run llama3.2 "{prompt}"'
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        return result.stdout.strip()
    else:
        return f"Error: {result.stderr.strip()}"
    
@app.post("/idea/")
def get_idea():
    """Extracts the critical idea from the uploaded research paper."""
    if not document_text:
        raise HTTPException(status_code=400, detail="No document uploaded. Please upload a PDF first.")
    
    prompt = f"""
            You are personal research assistant.
            Extract the critical idea from the research paper.
            {"Keep the answer relevant to {topic_text}." if topic_text else ""}
            Paper content:\n{document_text}
        """
    command = f'ollama run llama3.2 "{prompt}"'
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        return result.stdout.strip()
    else:
        return f"Error: {result.stderr.strip()}"
