from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle
import os
import csv
from typing import List
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = os.environ.get('OPENAI_API_KEY')

def embed_doc():
    contents: List[str] = []
    metadata: List[dict] = []
    csv_file_path = '../data/' + file_name + '.csv'

    with open(csv_file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            print(row['content'])
            content = row['content']

            contents.append(content)

            metadata_dict = {
                'title': row['title'],
                'heading': row['heading']
            }
            metadata.append(metadata_dict)

    # Load Data to vectorstore
    embedding = OpenAIEmbeddings()

    # I'm using add_texts to run every row the embeddings and add to the Chroma vectorstore.
    # vectorstore = vectorstore.add_texts(contents, metadata)
    vectorstore = FAISS.from_texts(
        texts=contents, embedding=embedding, metadatas=metadata)
    
    
    # Save vectorstore to a pickle file
    with open(vector_file_path, "wb") as f:
        pickle.dump(vectorstore, f)

file_name = "olympics_sections"
vector_file_path = '../vectorstores/' + file_name + '.pkl'

# Check if the vectorstore exists
if os.path.exists(vector_file_path):
    with open(vector_file_path, "rb") as f:
        docsearch = pickle.load(f)
else:
    embed_doc()
