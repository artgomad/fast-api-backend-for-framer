class AskHumanWrapper():
    """Tool that adds the capability to ask user for input.""" 
    def run(self, query: str) -> str:
        print("\nAsk human:")
        print(query)

class BasicAnswer():
    """Tool that represents a simple llms answer""" 
    def run(self, query: str) -> str:
        print("\nBasic answer")
        print(query)

