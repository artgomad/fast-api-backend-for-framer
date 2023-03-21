from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import openai
import os
import pickle

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


@app.post("/question", tags=["openAI"])
async def openAI_chatbot(chatlog: list):
    vectorstores_dir = os.path.abspath('vectorstores')
    if "vectorstore_faqs_ing.pkl" in os.listdir(vectorstores_dir):
        print("Found vectorstore in " + vectorstores_dir)
        with open(vectorstores_dir + "/vectorstore_faqs_ing.pkl", "rb") as f:
            vectorstore = pickle.load(f)
            print("loading vectorstore...")
    else:
        print("vectorstore not found")

    user_question = chatlog[-1]['content']
    print('User question: ' + user_question)

    # Get the top 5 documents from the vectorstore
    # similarity_search() is being retreived from langchain.vectorstores.faiss
    docs = vectorstore.similarity_search(user_question, 5)
    docs_headers = ""
    docs_content = ""
    for doc in docs:
        print(doc.metadata)
        docs_headers += "- " + doc.metadata["heading"] + "\n\n"
        docs_content += doc.page_content + "\n\n"
    print(docs_headers)

    # Add context snippets to the system prompt
    system_prompt = chatlog[0]['content'].format(docs_content)
    chatlog[0] = {'role': 'system', 'content': system_prompt}
    print('System prompt: ' + chatlog[0]['content'])

    params = {
        "model": "gpt-3.5-turbo",
        "temperature": 0,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "messages": chatlog,
    }

    return {
        "data": openai.ChatCompletion.create(**params)
    }

