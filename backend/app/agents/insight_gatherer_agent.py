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
**TRANSCRIPT:**
* Participant: {transcript}

* Interviewer: {question1}

* Participant: {answer1}

* Interviewer: {question2}

* Participant: {answer2}

* Interviewer: {question3}

* Participant: {answer3}
"""

system_message = """
You are a qualitative research assistant. Your goal is to gather observations from a given user transcript.
The goal of the research study is to gather insights around the following research questions.

RESEARCH QUESTIONS:
{research_questions}

INSTRUCTIONS:
- The maximum amount of observations you can generate from a transcript is 3
- The minimum amount of observations you can generate from a transcript 1
- Format each observations EXACTLY like this: "- {{observations}}" 

CONTEXT OF THE TRANSCRIPT:
{context}

"""


class CustomPromptTemplate(BaseChatPromptTemplate):
    template: str

    # This should return ChatPromptTemplate
    def format_messages(self, **kwargs) -> str:
        # Add input_variables to kwargs
        kwargs["transcript"] = kwargs.get("transcript", "")
        kwargs["question1"] = kwargs.get(
            "followup_questions", {}).get("question1", "")
        kwargs["question2"] = kwargs.get(
            "followup_questions", {}).get("question2", "")
        kwargs["question3"] = kwargs.get(
            "followup_questions", {}).get("question3", "")
        kwargs["answer1"] = kwargs.get("user_answers", {}).get("answer1", "")
        kwargs["answer2"] = kwargs.get("user_answers", {}).get("answer2", "")
        kwargs["answer3"] = kwargs.get("user_answers", {}).get("answer3", "")
        kwargs["research_questions"] = kwargs.get("research_questions", "")
        kwargs["context"] = kwargs.get("context", "")

        formatted_system_message = system_message.format(**kwargs)
        formatted_user_message = user_message.format(**kwargs)

        llm_prompt_input = [SystemMessage(
            content=formatted_system_message), HumanMessage(content=formatted_user_message)]

        print('FORMATED PROMPT AS RECEIVED BY THE LLM\n')
        print(llm_prompt_input)

        return llm_prompt_input


def create_insights_agent():
    # Declare a dynamic prompt formatting class that adds tools, input, chat_history
    prompt = CustomPromptTemplate(
        template=system_message,
        input_variables=["transcript", "research_questions",
                         "context", "followup_questions", "user_answers"],
    )

    llm = ChatOpenAI(temperature=0.7, model='gpt-4')

    # Declare a chain that will trigger an openAI completion with the given prompt
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
    )

    return llm_chain
