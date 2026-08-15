import os
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda
from langserve import add_routes

app = FastAPI(title="Knowledge Transfer PDF RAG Agent")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)

vectorstore = None

class QuerySchema(BaseModel):
    question: str = Field(description="Question about the uploaded document context")

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global vectorstore
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    file_path = f"/tmp/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    
    vectorstore = FAISS.from_documents(splits, embeddings)
    return {"message": f"Successfully indexed {len(splits)} text chunks from {file.filename}."}

def answer_kt_query(inputs: QuerySchema) -> str:
    global vectorstore
    if vectorstore is None:
        return "Please upload a PDF document first via the `/upload` endpoint."
    
    q = inputs.question if isinstance(inputs, QuerySchema) else inputs.get("question", "")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    context_docs = retriever.invoke(q)
    context = "\n\n".join([doc.page_content for doc in context_docs])
    
    prompt = f"Answer the following Knowledge Transfer question based only on context:\n\nContext:\n{context}\n\nQuestion: {q}"
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)

rag_chain = RunnableLambda(answer_kt_query).with_types(input_type=QuerySchema, output_type=str)
add_routes(app, rag_chain, path="/kt-agent")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
