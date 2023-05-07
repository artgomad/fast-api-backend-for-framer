import os
import pickle

class Qa_RetrievalWrapper():
     """
     #IT MIGHT BE BETTER TO INITIALISE THE VECTORSTORE IN THE INIT FUNCTION

      def __init__(self):
        vectorstores_dir = os.path.abspath('vectorstores')
        vectorstore_name = 'olympics_sections.pkl'
        vectorstore_path = os.path.join(vectorstores_dir, vectorstore_name)
        with open(vectorstore_path, 'rb') as f:
            self.vectorstore = pickle.load(f)
     """

     def run(self, query: str) -> str:
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

        print('User question: ' + query)

        # Get the top 5 documents from the vectorstore
        # similarity_search() is being retreived from langchain.vectorstores.faiss
        docs = vectorstore.similarity_search(query, 3)
        docs_headers = ""
        docs_content = ""
        for doc in docs:
            docs_headers += "- " + \
                list(doc.metadata.values())[0] + ", " + \
                list(doc.metadata.values())[1] + "\n\n"
            docs_content += doc.page_content + "\n\n"
        print(docs_headers)

        return docs_content