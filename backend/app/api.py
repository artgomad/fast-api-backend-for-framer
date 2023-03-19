from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
"""
from langchain.document_loaders import PagedPDFSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import dotenv_values
import os

config = dotenv_values(".env.local")
os.environ['OPENAI_API_KEY'] = config['OPENAI_API_KEY']
"""

todos = [
    {
        "id": "1",
        "item": "Read a book."
    },
    {
        "id": "2",
        "item": "Cycle around town."
    }
]


app = FastAPI()

origins = [
    "http://localhost:3000",
    "localhost:3000",
    "https://project-zszmhke1xyd6rlfxgg1i.framercanvas.com",  # Framer Canvas
    "https://comprehensive-value-405432.framer.app"  # A framer publised website
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/", tags=["root"])
async def read_root() -> dict:
    return {"message": "Welcome to your todo list."}


@app.get("/todo", tags=["todos"])
async def get_todos() -> dict:
    return {"data": todos}


@app.post("/todo", tags=["todos"])
async def add_todo(todo: dict) -> dict:
    todos.append(todo)
    return {
        "data": {"Todo added."}
    }


@app.put("/todo/{id}", tags=["todos"])
async def update_todo(id: int, body: dict) -> dict:
    for todo in todos:
        if int(todo["id"]) == id:
            todo["item"] = body["item"]
            return {
                "data": f"Todo with id {id} has been updated."
            }

    return {
        "data": f"Todo with id {id} not found."
    }


@app.delete("/todo/{id}", tags=["todos"])
async def delete_todo(id: int) -> dict:
    for todo in todos:
        if int(todo["id"]) == id:
            todos.remove(todo)
            return {
                "data": f"Todo with id {id} has been removed."
            }

    return {
        "data": f"Todo with id {id} not found."
    }
"""
loader = PagedPDFSplitter("nl-employee-handbook-local-v12-0.pdf")
pages = loader.load_and_split()
faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings())


@app.get("/pdf", tags=["pdf"])
async def pdf(request: Request, question: str):

    # question = "How many vacation days do I get?";

    docs = faiss_index.similarity_search(question, k=2)

    response = ""

    print("----------------------------------------------------------------")
    print("question: ", question)

    for doc in docs:
        print(doc.page_content)
        response += doc.page_content

    print("----------------------------------------------------------------")

    return '{"success":"true", "response": %s}' % response
"""