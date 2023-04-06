from fastapi import FastAPI, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from utils.wiki_to_csv import recursively_find_all_pages, extract_sections
import openai
from dotenv import load_dotenv
import pandas as pd
import os
import pickle
import json
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain.agents import AgentExecutor
from app.agents.mortgage_agent import agent, tools
from app.agents.callbacks.custom_callbacks import MyAgentCallback, MyAgentCallback_works

# import tiktoken

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

load_dotenv()
openai.api_key = os.environ.get('OPENAI_API_KEY')
wolframalpha_app_key = os.environ.get('WOLFRAMALPHA_APP_ID')
serpapi_api_key = os.environ.get('SERPAPI_API_KEY')

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

wolfram = WolframAlphaAPIWrapper()


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


@app.options("/question", tags=["openAI"])
async def options_handler(request: Request):
    """Handle HTTP OPTIONS requests for /question."""
    print("OPTIONS request received")
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "*",
        "Access-Control-Allow-Headers": "*",
    }
    return JSONResponse(content={}, headers=headers)


@app.post("/question", tags=["openAI"])
async def openAI_chatbot(chatlog: list):
    vectorstores_dir = os.path.abspath('vectorstores')
    # vectorstore_faqs_ing.pkl
    # olympics_sections.pkl
    vectorstore = 'olympics_sections.pkl'
    if vectorstore in os.listdir(vectorstores_dir):
        print("Found vectorstore in " + vectorstores_dir)
        with open(vectorstores_dir + "/" + vectorstore, "rb") as f:
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
        docs_headers += "- " + \
            list(doc.metadata.values())[0] + ", " + \
            list(doc.metadata.values())[1] + "\n\n"
        docs_content += doc.page_content + "\n\n"
    print(docs_headers)

    # Add context snippets to the system prompt
    system_prompt = chatlog[0]['content'].format(docs_content)
    chatlog[0] = {'role': 'system', 'content': system_prompt}
    # print('System prompt: ' + chatlog[0]['content'])

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

# @app.post("/agentCall", tags=["openAI"])

"""
TO BE DONE:
FIX WEBSOCKET CONNECTION
"""
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        # Process the received data from the client
        chatlog = json.loads(data)['chatlog']

        vectorstores_dir = os.path.abspath('vectorstores')
        # vectorstore_faqs_ing.pkl
        # olympics_sections.pkl
        vectorstore = 'olympics_sections.pkl'
        if vectorstore in os.listdir(vectorstores_dir):
            print("Found vectorstore in " + vectorstores_dir)
            with open(vectorstores_dir + "/" + vectorstore, "rb") as f:
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
            docs_headers += "- " + \
                list(doc.metadata.values())[0] + ", " + \
                list(doc.metadata.values())[1] + "\n\n"
            docs_content += doc.page_content + "\n\n"
        print(docs_headers)

        # Add context snippets to the system prompt
        system_prompt = chatlog[0]['content'].format(docs_content)
        chatlog[0] = {'role': 'system', 'content': system_prompt}
        # print('System prompt: ' + chatlog[0]['content'])

        # Run agent and get last action and observation
        callback = MyAgentCallback(websocket)
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent, tools=tools, verbose=False, callback_manager=callback, return_intermediate_steps=True,
        )
        agent_output = await agent_executor.acall(user_question)
        #agent_output = agent_executor(user_question)
        """
        #This one works if I remove return_intermediate_steps=True from agent_executor
        agent_output = agent_executor.run(user_question)
        """

        print('agent_input = ' + agent_output['input'])
        print('agent_output = ' + agent_output['output'])
        print('agent_intermediate_steps = ')
        print(agent_output['intermediate_steps'])

        """
        if callback.last_action_text:
            print(f"LAST ACTION: {callback.last_action_text}")
        if callback.last_observation:
            print(f"LAST OBSERVATION: {callback.last_observation}")
        """

        data = ""
        try:
            data = wolfram.run(user_question)
            print('\n\nWolfram response:')
            print(data)
        except Exception as e:
            print("Error:", e)

        await websocket.send_json({
            # "data": openai.ChatCompletion.create(**params)
            "data":  agent_output['output'],
            "intermediate_steps": agent_output['intermediate_steps'],
        })


@app.post("/agentCall", tags=["openAI"])
async def agent_chatbot(chatlog: list):
    vectorstores_dir = os.path.abspath('vectorstores')
    # vectorstore_faqs_ing.pkl
    # olympics_sections.pkl
    vectorstore = 'olympics_sections.pkl'
    if vectorstore in os.listdir(vectorstores_dir):
        print("Found vectorstore in " + vectorstores_dir)
        with open(vectorstores_dir + "/" + vectorstore, "rb") as f:
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
        docs_headers += "- " + \
            list(doc.metadata.values())[0] + ", " + \
            list(doc.metadata.values())[1] + "\n\n"
        docs_content += doc.page_content + "\n\n"
    print(docs_headers)

    # Add context snippets to the system prompt
    system_prompt = chatlog[0]['content'].format(docs_content)
    chatlog[0] = {'role': 'system', 'content': system_prompt}
    # print('System prompt: ' + chatlog[0]['content'])

    # Run agent and get last action and observation
    callback = MyAgentCallback_works()
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=False, callback_manager=callback, return_intermediate_steps=True,
    )
    agent_output = agent_executor(user_question)
    """
        #This one works if I remove return_intermediate_steps=True from agent_executor
        agent_output = agent_executor.run(user_question)
        """

    print('agent_input = ' + agent_output['input'])
    print('agent_output = ' + agent_output['output'])
    print('agent_intermediate_steps = ')
    print(agent_output['intermediate_steps'])

    """
        if callback.last_action_text:
            print(f"LAST ACTION: {callback.last_action_text}")
        if callback.last_observation:
            print(f"LAST OBSERVATION: {callback.last_observation}")
        """

    data = ""
    try:
        data = wolfram.run(user_question)
        print('\n\nWolfram response:')
        print(data)
    except Exception as e:
        print("Error:", e)

    return {
        # "data": openai.ChatCompletion.create(**params)
        "data":  agent_output['output'],
        "intermediate_steps": agent_output['intermediate_steps'],
    }


# To run this functio can take up to 30 minutes
@app.post("/wiki", tags=["openAI"])
async def wiki_search():
    print("Starting wiki search...")
    pages = recursively_find_all_pages(["2020 Summer Olympics"])
    print("Pages length: ", len(pages))

    res = []
    for page in pages:
        res += extract_sections(page.content, page.title)
    df = pd.DataFrame(res, columns=["title", "heading", "content", "tokens"])
    df = df[df.tokens > 40]
    df = df.drop_duplicates(['title', 'heading'])
    df = df.reset_index().drop('index', axis=1)  # reset index
    df.head()

    df.to_csv('../data/olympics_sections.csv', index=False)

    return {
        "data": "Done."
    }
