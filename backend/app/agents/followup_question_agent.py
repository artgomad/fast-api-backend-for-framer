from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import BaseChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from dotenv import load_dotenv


load_dotenv()

RED = "\033[1;31m"
GREEN = "\033[0;32m"
BLUE = "\033[34m"
YELLOW = "\033[1;33m"
RESET = "\033[0m"

user_message = """
{transcript}
"""

system_message = """
You are a qualitative researcher conducting a user test in the following context:
{context}

The research questions of the study are:
{research_questions}

Come up with {n_questions} follow-up question(s) to ask the user based on the transcript provided. The questions should address different topics to maximize insight generation.
"""

multiple_shot_prompting = """
EXAMPLE USER TRANSCRIPT:
User: "I'm not sure what this 'comprehensive coverage' means, and I don't know if I need all these extras. The price seems reasonable, but I'm a bit confused about the whole process."

EXAMPLE OF FOLLOW-UP QUESTIONS:
- Can you explain which part of the process you find confusing?
- What do you think 'comprehensive coverage' includes?
- Do you feel that the pricing and coverage options available provide a good value for the cost, or are there any aspects that make it difficult to determine if you're getting a good deal?

Now, generate follow-up questions based on the provided user transcript.
USER TRANSCRIPT:
"""


class CustomPromptTemplate(BaseChatPromptTemplate):
    template: str

    # This should return ChatPromptTemplate
    def format_messages(self, **kwargs) -> str:
        # Add input_variables to kwargs
        kwargs["n_questions"] = kwargs.get("n_questions", "")
        kwargs["research_questions"] = kwargs.get("research_questions", "")
        kwargs["context"] = kwargs.get("context", "")
        kwargs["transcript"] = kwargs.get("transcript", "")

        formatted_system_message = system_message.format(**kwargs)
        formatted_user_message = user_message.format(**kwargs)

        llm_prompt_input = [SystemMessage(
            content=formatted_system_message), HumanMessage(content=formatted_user_message)]

        print('FORMATED PROMPT AS RECEIVED BY THE LLM\n')
        print(llm_prompt_input)

        return llm_prompt_input


def create_followup_question_agent():
    # Declare a dynamic prompt formatting class that adds tools, input, chat_history
    prompt = CustomPromptTemplate(
        template=system_message,
        input_variables=[
            "n_questions",
            "research_questions",
            "context",
            "transcript"],
    )

    llm = ChatOpenAI(temperature=0.7, model='gpt-4')

    # Declare a chain that will trigger an openAI completion with the given prompt
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
    )

    return llm_chain
