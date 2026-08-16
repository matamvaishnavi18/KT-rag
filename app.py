import os
import tempfile
import uvicorn

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)

from langchain_core.runnables import RunnableLambda
from langserve import add_routes


# ---------------------------------------------------------
# App
# ---------------------------------------------------------

app = FastAPI(
    title="Knowledge Transfer PDF RAG Agent"
)


# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY environment variable is not set."
    )


# ---------------------------------------------------------
# Models
# ---------------------------------------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY,
)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
)


# ---------------------------------------------------------
# Vector store
# ---------------------------------------------------------

VECTORSTORE_PATH = "./faiss_index"

vectorstore = None


# ---------------------------------------------------------
# Request schema
# ---------------------------------------------------------

class QuerySchema(BaseModel):
    question: str = Field(
        description="Question about the uploaded Knowledge Transfer document"
    )


# ---------------------------------------------------------
# Load existing FAISS index
# ---------------------------------------------------------

def load_vectorstore():
    global vectorstore

    if os.path.exists(VECTORSTORE_PATH):
        try:
            vectorstore = FAISS.load_local(
                VECTORSTORE_PATH,
                embeddings,
                allow_dangerous_deserialization=True,
            )

            print("Loaded existing FAISS index.")

        except Exception as e:
            print(f"Could not load existing FAISS index: {e}")


# ---------------------------------------------------------
# Upload PDF
# ---------------------------------------------------------

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    global vectorstore

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file provided."
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp:

        content = await file.read()
        tmp.write(content)
        file_path = tmp.name

    try:

        # -------------------------------------------------
        # Load PDF
        # -------------------------------------------------

        loader = PyPDFLoader(file_path)
        docs = loader.load()

        if not docs:
            raise HTTPException(
                status_code=400,
                detail="The PDF contains no readable content."
            )

        # -------------------------------------------------
        # Split into chunks
        # -------------------------------------------------

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
        )

        splits = text_splitter.split_documents(docs)

        if not splits:
            raise HTTPException(
                status_code=400,
                detail="Could not create text chunks from the PDF."
            )

        # -------------------------------------------------
        # Create FAISS vector store
        # -------------------------------------------------

        vectorstore = FAISS.from_documents(
            splits,
            embeddings,
        )

        # -------------------------------------------------
        # Persist vector store
        # -------------------------------------------------

        vectorstore.save_local(VECTORSTORE_PATH)

        return {
            "message": "PDF successfully indexed.",
            "filename": file.filename,
            "pages": len(docs),
            "chunks": len(splits),
        }

    finally:

        # Remove temporary PDF
        if os.path.exists(file_path):
            os.remove(file_path)


# ---------------------------------------------------------
# Optional status endpoint
# ---------------------------------------------------------

@app.get("/status")
async def status():

    if vectorstore is None:
        return {
            "ready": False,
            "message": "No PDF has been uploaded."
        }

    return {
        "ready": True,
        "message": "Knowledge base is ready."
    }


# ---------------------------------------------------------
# RAG function
# ---------------------------------------------------------

def answer_kt_query(inputs: QuerySchema) -> str:

    global vectorstore

    if vectorstore is None:

        return (
            "No Knowledge Transfer PDF has been uploaded yet. "
            "Please upload a PDF using POST /upload first."
        )

    q = inputs.question

    # -----------------------------------------------------
    # Retrieve relevant chunks
    # -----------------------------------------------------

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 4
        }
    )

    context_docs = retriever.invoke(q)

    if not context_docs:
        return (
            "I could not find relevant information in "
            "the uploaded Knowledge Transfer document."
        )

    # -----------------------------------------------------
    # Build context
    # -----------------------------------------------------

    context = "\n\n".join(
        doc.page_content
        for doc in context_docs
    )

    # -----------------------------------------------------
    # Prompt
    # -----------------------------------------------------

    prompt = f"""
You are a Knowledge Transfer documentation assistant.

Answer the user's question using ONLY the information
contained in the provided context.

If the answer cannot be found in the context, say:

"I could not find this information in the uploaded
Knowledge Transfer document."

Do not invent commands, environment variables,
configuration values, file paths, or procedures.

Context:
----------------
{context}
----------------

Question:
{q}

Answer:
"""

    # -----------------------------------------------------
    # Generate answer
    # -----------------------------------------------------

    response = llm.invoke(prompt)

    if hasattr(response, "content"):
        return response.content

    return str(response)


# ---------------------------------------------------------
# LangServe
# ---------------------------------------------------------

rag_chain = (
    RunnableLambda(answer_kt_query)
    .with_types(
        input_type=QuerySchema,
        output_type=str,
    )
)

add_routes(
    app,
    rag_chain,
    path="/kt-agent",
)


# ---------------------------------------------------------
# Startup
# ---------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    load_vectorstore()


# ---------------------------------------------------------
# Run
# ---------------------------------------------------------

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            8000
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
    )
