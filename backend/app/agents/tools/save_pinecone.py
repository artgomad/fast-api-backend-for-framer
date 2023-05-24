import os
import json
import nomic
import numpy as np
import openai
import pandas as pd
import pinecone
from dotenv import load_dotenv
from nomic import atlas
from sklearn.cluster import KMeans
from app.agents.insight_writer import write_insights

userID = 0

class Pinecone():
    def __init__(self):
        self.PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
        assert self.PINECONE_API_KEY, "PINECONE_API_KEY environment variable is missing from .env"

        self.PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "")
        assert (
            self.PINECONE_ENVIRONMENT
        ), "PINECONE_ENVIRONMENT environment variable is missing from .env"

        # Table config
        self.YOUR_TABLE_NAME = os.getenv("TABLE_NAME", "")
        assert self.YOUR_TABLE_NAME, "TABLE_NAME environment variable is missing from .env"

        self.NOMIC_API_KEY = os.getenv("NOMIC_API_KEY", "")
        assert self.NOMIC_API_KEY, "NOMIC_API_KEY variable is missing from .env"

        # Create Pinecone index
        table_name = self.YOUR_TABLE_NAME
        dimension = 1536
        metric = "cosine"
        pod_type = "p1"
        if table_name not in pinecone.list_indexes():
            pinecone.create_index(
                table_name, dimension=dimension, metric=metric, pod_type=pod_type
            )

        # Connect to the index
        self.index = pinecone.Index(table_name)

    def get_ada_embedding(self, text):
        text = text.replace("\n", " ")
        return openai.Embedding.create(input=[text], model="text-embedding-ada-002")[
            "data"
        ][0]["embeddings"]

    def save(self, data, insightID, context):
        enriched_metadata = {
            "data": data,
            "context": context,
            "user_id": userID,
        }  # This is where you should enrich the metadata to be saved (if needed)

        data_id = f"u{userID}_i{insightID}"

        # get vector of the actual result extracted from the dictionary
        vector = self.get_ada_embedding(enriched_metadata["data"])

        # Save as in Pinecone
        self.index.upsert(
            [(data_id, vector, enriched_metadata)])  # ,namespace=OBJECTIVE

    def retreive_embeddings(self, query: str, top_results_num: int):
        query_embedding = self.get_ada_embedding(query)
        results = self.index.query(
            query_embedding, top_k=top_results_num, include_metadata=True)  # ,namespace=OBJECTIVE
        print("***** RESULTS *****")
        print(results)
        sorted_results = sorted(
            results.matches, key=lambda x: x.score, reverse=True)
        return [(str(item.metadata["data"])) for item in sorted_results]


