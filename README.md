# Developing a Single Page App with FastAPI and React

### Want to learn how to build this?

Check out the [post](https://testdriven.io/blog/fastapi-react/).

## Want to use this project?

1. Fork/Clone

2. Run the server-side FastAPI app locally from the terminal window:

    ```sh
    $ cd backend
    $ python3.9 -m venv env (only the first time)
    $ source env/bin/activate
    (env)$ pip install -r requirements.txt (only when requirements change)
    (env)$ python main.py
    ```

    Navigate to [http://localhost:8000](http://localhost:8000)

3. Initialise your OpenAI API key by writing this in your backend terminal:

    ```sh
    $ cd backend
    $ export OPENAI_API_KEY="your-key"
    ```

4. Run the client-side React app in a different terminal window:

    ```sh
    $ cd frontend
    $ npm install
    $ npm run start
    ```

    Navigate to [http://localhost:3000](http://localhost:3000)

5. Debug Heroku

    Heroku backend is always running. Doesn't need to be activated or deactivated

    ```sh
    $ heroku logs --tail
    ```
    
    
## Making an LLM agent with Langchain

Let's look at the key scripts that make the repo work:

1. **Main:** Runs the application, defining the port where the application runs either locally or with Heroku.
    [/backend/main.py](https://github.com/artgomad/fast-api-backend-for-framer/blob/main/backend/main.py)

2. **API definition:** Defines the API calls that can be called in the front end, using Fast API. We are going to use a Websocket endpoint.
    [/backend/app/api.py](https://github.com/artgomad/fast-api-backend-for-framer/blob/main/backend/app/api.py)
    ```python
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
    ```
    The websocket endpoint recieves the chatlog data from the frontend and returns the output of the LLM agent
    ```python
    #...
    while True:
        data = await websocket.receive_text()
        # Process the received data from the client
        chatlog = json.loads(data)['chatlog']
        
        #...
        
        await websocket.send_json({
            "data":  agent_output['output'],
            # "intermediate_steps": agent_output['intermediate_steps'],
        })
    ```
    
    The agent is defined in [mortgage_agent_conversational.py](https://github.com/artgomad/fast-api-backend-for-framer/blob/main/backend/app/agents/mortgage_agent_conversational.py) and gets the chat history as input
    
    ```python
    from app.agents.mortgage_agent_conversational import create_agent
    #...
    agent_executor = create_agent(websocket)
    #...
    agent_output = await agent_executor.acall({'input': user_question, 'chat_history': chatlog_strings})
    ```
    
3. **Agent definition:** The agent is defined in
    [/backend/app/agents/mortgage_agent_conversational.py](https://github.com/artgomad/fast-api-backend-for-framer/blob/main/backend/app/agents/mortgage_agent_conversational.py)
    To learn more about langchain agents check [here](https://python.langchain.com/en/latest/modules/agents/agents/custom_llm_agent.html)
    
    An LLM agent consists of three parts:
    **- CustomPromptTemplate:** This class adds the conversation history, tools and tool names to the prompt that will instruct the language model on what to do. 
    **- LlmChain:** This is a call to the defined language model, that sends the prompt as input and returns an text output
    **- Stop sequence:** Instructs the LLM to stop generating as soon as this string is found
    **- CustomOutputParser:** This determines how to parse the LLMOutput into an AgentAction or AgentFinish object

    The LLMAgent is used in an **AgentExecutor**. This AgentExecutor can largely be thought of as a loop that:
    - Passes user input and any previous steps to the Agent (in this case, the LLMAgent)
    - If the Agent returns an AgentFinish, then return that directly to the user
    - If the Agent returns an AgentAction, then use that to call a tool and get an Observation
    - Repeat, passing the AgentAction and Observation back to the Agent until an AgentFinish is emitted.


    AgentAction is a response that consists of action and action_input. action refers to which tool to use, and action_input refers to the input to that tool. log can also be provided as more context (that can be used for logging, tracing, etc).

    AgentFinish is a response that contains the final message to be sent back to the user. This should be used to end an agent run.

    
    ```python
    
    #...
    
    ```

