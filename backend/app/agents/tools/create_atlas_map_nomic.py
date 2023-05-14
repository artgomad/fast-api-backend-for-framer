import os
import json
import nomic
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import pinecone
from nomic import atlas
from sklearn.cluster import KMeans
from app.agents.insight_writer import write_insights

userID = 0


def cluster_embeddings(df, n_clusters,  cluster_column: str = "cluster"):
    print("\nKMEAN CLUSTERING STARTS\n")

    # check if all items in the embedding column are ndarrays. Important when clustering for a second time
    if not all(isinstance(item, np.ndarray) for item in df.embeddings):
        # convert string to numpy array
        df["embeddings"] = df.embeddings.apply(eval).apply(
            np.array)

    matrix = np.vstack(df.embeddings.values)
    matrix.shape

    kmeans = KMeans(n_clusters=n_clusters,
                    init="k-means++", random_state=42)
    kmeans.fit(matrix)
    labels = kmeans.labels_

    # Add a a column named "cluster" to the dataframe and assign the cluster labels to it
    df[cluster_column] = labels

    for i in range(n_clusters):
        cluster_i = df[df[cluster_column] == i]
        # print("cluster", i, ":\n", cluster_i)

    df = df.sort_values(by=cluster_column)

    return df, matrix


def name_clusters(n_clusters, df, max_rev_per_cluster, cluster_column: str = "cluster"):
    # Let's show random samples from each cluster.
    # We'll use text-davinci-003 to name the clusters, based on a random sample of 5 reviews from that cluster.

    print("NAMING CLUSTERS")
    cluster_labels = []

    for i in range(n_clusters):
        # check the size of each cluster and set rev_per_cluster to the size of the cluster
        # if it is smaller than the desired sample size
        cluster_size = len(df[df[cluster_column] == i])
        if cluster_size < max_rev_per_cluster:
            rev_per_cluster = cluster_size
        else:
            rev_per_cluster = max_rev_per_cluster

        print(f"Cluster {i} - Ocurrences: {cluster_size}")

        # Get a random sample of reviews from the cluster
        reviews = "\n".join(
            df[df[cluster_column] == i]
            .data
            .sample(rev_per_cluster, random_state=42)
            .values
        )

        # Generate name for the cluster
        llm_chain = write_insights(model='gpt-4', temperature=0.7)
        chain_output = llm_chain.run(
            {'observations': reviews,
             })

        print(chain_output)
        cluster_labels.append(chain_output)

    # Replace cluster numbers with cluster names in df["Cluster"] column of the dataframe
    df[cluster_column] = df[cluster_column].replace(
        list(range(n_clusters)), cluster_labels)

    return df, cluster_labels


class NomicAtlas():
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

        # Connect to the index
        self.index = pinecone.Index(self.YOUR_TABLE_NAME)

    def visualise_database(self, n_clusters: int = 5):
        nomic.login(self.NOMIC_API_KEY)
        index_stats = self.index.describe_index_stats()
        num_embeddings = index_stats['namespaces']['']['vector_count']
        print(num_embeddings)

        # Fetch all data from Pinecone Index
        vectors = self.index.fetch(
            ids=[f"u0_i{i}" for i in range(num_embeddings)])

        ids = []
        embeddings = []
        data_list = []
        context_list = []
        userID_list = []

        # Add the data from the Pinecone index to different arrays
        for id, vector in vectors['vectors'].items():
            ids.append(id)
            embeddings.append(vector['values'])
            data_list.append(vector['metadata'].get('data'))
            context_list.append(vector['metadata'].get('context'))
            userID_list.append(vector['metadata'].get('user_id'))

        # Convert the embeddings array to a numpy array of embeddings
        embeddings = np.array(embeddings)
        # Convert the embeddings array to a string to pass it to the dataframe as a single column
        embeddings_str = [json.dumps(e.tolist()) for e in embeddings]

        df = pd.DataFrame({
            'id': ids,
            'embeddings': embeddings_str,
            'data': data_list,
            'context': context_list,
            'user_id': userID_list
        })

        df, matrix = cluster_embeddings(df, n_clusters=n_clusters)

        df_clustered, cluster_labels = name_clusters(
            n_clusters=n_clusters, df=df, max_rev_per_cluster=10, cluster_column="cluster")

        # Convert the embeddings column to a numpy array
        embeddings_2 = np.array([np.array(xi)
                                for xi in df_clustered['embeddings']])

        # Remove embeddings column to avoid passing it to Atlas
        df_clustered = df_clustered.drop(columns=['embeddings'])
        """
        self.export_excel(n_clusters=n_clusters,
                          n_sub_clusters=n_sub_clusters, df=df, max_rev_per_cluster=10)
        """
        """
        data_to_atlas_1 = [{'id': id, 'data': data, 'context': context, 'userID': user}
                           for id, data, context, user in zip(ids, data_list, context_list, userID_list)]
        """

        data_to_atlas = [{'id': id, 'data': data, 'cluster': cluster, 'context': context}
                         for id, data, cluster, context
                         in zip(df_clustered['id'], df_clustered['data'], df_clustered['cluster'], df_clustered['context'])]

        atlas_project = atlas.map_embeddings(
            embeddings=embeddings_2,
            data=data_to_atlas,  # data_to_atlas and data_to_atlas_2 don't work for some reason
            id_field='id',
            colorable_fields=['cluster'],
            #topic_label_field='cluster',
            name='UsabilityAI-test2',
            reset_project_if_exists=True,
        )

        # Add tags to the map based on the clusters
        # Initialize an empty dictionary
        cluster_ids = {}

        # Loop through the list of dictionaries
        for dictionary in data_to_atlas:
            # If the cluster is not in the dictionary, initialize it with an empty list
            if dictionary['cluster'] not in cluster_ids:
                cluster_ids[dictionary['cluster']] = []
            # Add the id list to the appropriate cluster
            cluster_ids[dictionary['cluster']].append(dictionary['id'])

        # Now cluster_ids is a dictionary where each key is a cluster and each value is a list of ids
        print('cluster_ids')
        print(cluster_ids)

        map = atlas_project.get_map('UsabilityAI-test2')

        # Tag the atlas project with the clusters
        # cluster_ids.items() will give us a tuple of each key-value pair in the cluster_ids dictionary.
        for cluster, ids in cluster_ids.items():
            map.tag(ids, [cluster])

