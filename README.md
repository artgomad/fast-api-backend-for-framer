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

1. FastAPI definition:
    [https://github.com/artgomad/fast-api-backend-for-framer/blob/main/backend/app/api.py](/backend/app/api.py)

