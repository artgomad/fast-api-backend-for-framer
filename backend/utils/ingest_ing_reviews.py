import csv
import os
import pickle
from typing import List
from xlsxwriter import Workbook
import matplotlib.pyplot as plt
import nltk
import numpy as np
import openai
import pandas as pd
from dotenv import load_dotenv
from googletrans import Translator
from nltk.corpus import names
from openai.embeddings_utils import get_embedding
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

load_dotenv()
nltk.download('names')
nltk.download('averaged_perceptron_tagger')
translator = Translator()
openai.api_key = os.environ.get('OPENAI_API_KEY')


def translate(text):
    translator = Translator()
    try:
        translated = translator.translate(
            text, dest="en", src="nl")
        return translated.text
    except:
        return text


def replace_names(text):
    # Load list of human names
    name_set = set(names.words())
    # Tokenize the text
    complaint_words = nltk.word_tokenize(text)
    # Check for human names and replace them with NAME_i
    replaced_names = [
        f"NAME_{i}" if token in name_set else token for i, token in enumerate(complaint_words)]
    # Join the tokens back into a string
    result = " ".join(replaced_names)

    return result


def translate_doc(file_path, max_rows):
    with open(file_path, newline='', encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile, delimiter='|')

        contents: List[str] = []
        metadata: List[dict] = []

        for i, row in enumerate(reader):
            # Stop after processing X rows
            if max_rows != None:
                if i >= max_rows:
                    break

            print('-------------')
            print('NEW ROW')

            for fields in row:

                if fields != None:
                    row_contents = row[fields].split(";")
                    try:
                        new_row = {
                            new_names[0]: row_contents[0],
                            new_names[1]: row_contents[1],
                            new_names[2]: row_contents[2],
                            new_names[3]: row_contents[3]
                        }

                    except IndexError:
                        new_row = {
                            new_names[0]: '',
                            new_names[1]: '',
                            new_names[2]: '',
                            new_names[3]: ''
                        }

                    for item in new_row:
                        if new_row[item] != '':
                            text_result = translate(new_row[item])

                            if item == 'complaint':
                                text_result = replace_names(
                                    text_result)

                            new_row[item] = text_result

                    # Join new_row['intent'] and new_row['complaint'] into a single string spearated by '\n'
                    if new_row['complaint'] == "":
                        content = f"User intent: {new_row['intent']}\n"
                    else:
                        content = f"User intent: {new_row['intent']}\nComplaint: {new_row['complaint']}\n"

                    print(content)
                    contents.append(content)

                    metadata_dict = {
                        'intent': new_row['intent'],
                        'success': new_row['success'],
                        'satisfaction_score': new_row['satisfaction_score']
                    }
                    metadata.append(metadata_dict)

        return (contents, metadata)


def filterComplaints(df):
    # Filter only the rows where the "content" column doesn't end with "complaint: "
    # df_complaints = df[~df.content.str.endswith("Complaint: ")]

    # Filter only the rows where the "content" column contains "Complaint: "
    df_complaints = df[df.content.str.contains("Complaint: ")]

    return df_complaints


def embed_doc(contents, metadata, embeddings_file_path):
    df = pd.DataFrame({
        'content': contents,
        'intent': [d['intent'] for d in metadata],
        'success': [d['success'] for d in metadata],
        'satisfaction_score': [d['satisfaction_score'] for d in metadata]
    }).pipe(lambda x: x[['intent', 'success', 'satisfaction_score', 'content']])

    df["embedding"] = df.content.apply(
        lambda x: get_embedding(x, engine=embedding_model))

    df_complaints = filterComplaints(df)

    df.to_csv(embeddings_file_path)
    df_complaints.to_csv(
        '../vectorstores/ING_Exit_Survey_complaints_embeddings.csv')


def cluster_embeddings(df, n_clusters,  cluster_column: str = "Cluster"):
    print("\nKMEAN CLUSTERING STARTS\n")

    # check if all items in the embedding column are ndarrays. Important when clustering for a second time
    if not all(isinstance(item, np.ndarray) for item in df.embedding):
        # convert string to numpy array
        df["embedding"] = df.embedding.apply(eval).apply(
            np.array)

    matrix = np.vstack(df.embedding.values)
    matrix.shape

    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42)
    kmeans.fit(matrix)
    labels = kmeans.labels_
    df[cluster_column] = labels

    for i in range(n_clusters):
        cluster_i = df[df[cluster_column] == i]
        #print("Cluster", i, ":\n", cluster_i)

    df = df.sort_values(by=cluster_column)

    return df, matrix


def visualise_clusters(df, matrix):
    # Increase the value of perpelexity when the clusters are too spread out
    tsne = TSNE(n_components=2, perplexity=5, random_state=42,
                init="random", learning_rate=200)
    vis_dims2 = tsne.fit_transform(matrix)

    x = [x for x, y in vis_dims2]
    y = [y for x, y in vis_dims2]

    for category, color in enumerate(["purple", "green", "red", "blue"]):
        xs = np.array(x)[df.Cluster == category]
        ys = np.array(y)[df.Cluster == category]
        plt.scatter(xs, ys, color=color, alpha=0.3)

        avg_x = xs.mean()
        avg_y = ys.mean()

        plt.scatter(avg_x, avg_y, marker="x", color=color, s=100)
    plt.title("Clusters identified visualized in language 2d using t-SNE")
    plt.savefig('../outputs/ING_Exit_Survey_clustered_visualization.png')
    # plt.show()


def name_clusters(n_clusters, df, max_rev_per_cluster, cluster_column: str = "Cluster"):
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

        print(f"Cluster {i} - Ocurrences: {cluster_size} - Theme:", end=" ")

        # Get a random sample of reviews from the cluster
        reviews = "\n".join(
            df[df[cluster_column] == i]
            .content
            .sample(rev_per_cluster, random_state=42)
            .values
        )

        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f'What do the following customer reviews have in common?\n\nCustomer reviews:\n"""\n{reviews}\n"""\n\nTheme:',
            temperature=0,
            max_tokens=64,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        print(response["choices"][0]["text"].replace("\n", ""))
        cluster_labels.append("Cluster " + str(i) + " - " +
                              response["choices"][0]["text"].replace("\n", ""))

    # Create a new DataFrame with the selected columns and cluster labels
    df_clustered = pd.DataFrame({
        'intent': df['intent'],
        'success': df['success'],
        'satisfaction_score': df['satisfaction_score'],
        'complaint': df['content'].str.split("Complaint: ").str[1],
        'embedding': df['embedding'],
        'content': df['content'],
    })

    df[cluster_column] = df[cluster_column].replace(
        list(range(n_clusters)), cluster_labels)

    return df, cluster_labels


def export_excel(n_clusters, n_sub_clusters, df, max_rev_per_cluster):

    df_clustered, cluster_labels = name_clusters(
        n_clusters, df, max_rev_per_cluster, "Cluster")

    writer = pd.ExcelWriter(
        '../outputs/ING_Exit_Survey_clustered.xlsx', engine='xlsxwriter')

    for i, cluster_label in enumerate(cluster_labels):
        cluster_ocurrences = df_clustered[df_clustered['Cluster']
                                          == cluster_label].shape[0]

        tab_name = f"Cluster {i} - {cluster_ocurrences}"

        # Filtering each cluster in the loop
        cluster_i = df_clustered[df_clustered['Cluster'] == cluster_label]

        # SECOND CLUSTERING ROUND
        cluster_i_withSubclusters, matrix = cluster_embeddings(
            cluster_i, n_clusters=n_sub_clusters, cluster_column="Sub_cluster")

        # Naming the subclusters
        cluster_i_withSubclusters, sub_cluster_labels = name_clusters(
            n_clusters=n_sub_clusters, df=cluster_i_withSubclusters, max_rev_per_cluster=max_rev_per_cluster, cluster_column="Sub_cluster")

        # ****************************************************************************************

        # REORGANISING FINAL DATAFRAME
        cluster_i_withSubclusters['complaint'] = cluster_i_withSubclusters['content'].str.split(
            "Complaint: ").str[1]
        cluster_i_withSubclusters = cluster_i_withSubclusters.drop(
            ['embedding', 'Unnamed: 0', 'content'], axis=1)

        final_df = cluster_i_withSubclusters.reindex(
            columns=['Cluster', 'Sub_cluster', 'intent', 'complaint', 'success', 'satisfaction_score'])

        final_df.to_excel(writer, sheet_name=tab_name)

    writer.save()


embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002

csv_column_names = [
    'Wat is het doel van je bezoek vandaag?',
    'Is dit gelukt?',
    'Hoe tevreden ben je over dit bezoek?',
    'Vervelend dat het niet gelukt is, kun je ons vertellen waarom niet?',


]
new_names = [
    'intent',
    'success',
    'satisfaction_score',
    'complaint'
]

csv_file_path = '../data/ING_Exit_Survey.csv'
embeddings_file_path = '../vectorstores/ING_Exit_Survey_embeddings.csv'
embeddings_complaints_file_path = '../vectorstores/ING_Exit_Survey_complaints_embeddings.csv'

if os.path.exists(embeddings_file_path):

    n_clusters = 5
    n_sub_clusters = 3

    with open(embeddings_file_path, "rb") as f:
        print("Loading vectorstore from file")

        df = pd.read_csv(embeddings_complaints_file_path)

        df, matrix = cluster_embeddings(df, n_clusters=n_clusters)

        visualise_clusters(df, matrix)

        export_excel(n_clusters=n_clusters,
                     n_sub_clusters=n_sub_clusters, df=df, max_rev_per_cluster=10)

else:
    # Write max_rows=None to run the whol document
    contents, metadata = translate_doc(csv_file_path, max_rows=None)
    embed_doc(contents, metadata, embeddings_file_path)
