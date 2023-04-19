# Developing a Single Page App with FastAPI and React

### Want to learn how to build this?

Check out the [post](https://testdriven.io/blog/fastapi-react/).

## Want to use this project?

1. Fork/Clone

2. Run the server-side FastAPI app locally from the terminal window:

    ```sh
    $ cd backend
    $ python3 -m venv env (only the first time)
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