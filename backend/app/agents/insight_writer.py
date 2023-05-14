from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import BaseChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain

user_message = """
You are a qualitative research assistant. Your goal is to tag a set of similar observations from different user interviews to speed up the qualitative analysis of the results.

Write a single tag for the provided observations following these rules:
- The tag should encapsulate the underlying insight within the given set of observations, and help researchers navigate the data.
- Use keywords from the observations when helpful
- The tag has to be 1 to 5 words long, with a "-" instead of a space between words.
- The tag has to be in lowercase.

OBSERVATIONS:
{observations}

TAG:
"""


class CustomPromptTemplate(BaseChatPromptTemplate):
    template: str

    def format_messages(self, **kwargs) -> str:
        # Add input_variables to kwargs
        kwargs["observations"] = kwargs.get("observations", "")

        formatted_user_message = user_message.format(**kwargs)

        llm_prompt_input = [HumanMessage(content=formatted_user_message)]

        #print('FORMATED PROMPT AS RECEIVED BY THE LLM\n')
        #print(llm_prompt_input)

        return llm_prompt_input


def write_insights(model, temperature):
    prompt = CustomPromptTemplate(
        template=user_message,
        input_variables=["observations"],
    )

    llm = ChatOpenAI(temperature=temperature, model=model)

    # Declare a chain that will trigger an openAI completion with the given prompt
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
    )
    return llm_chain
