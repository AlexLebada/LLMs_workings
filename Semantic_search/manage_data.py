from API_integration.LLM_API import get_chatgpt_embedding
import plotly.express as px
import pandas as pd
import os
from sklearn.cluster import KMeans
from umap import UMAP
import ast







def load_dataset(url):
    file_exists = os.path.exists('dataset_raw.csv')
    if file_exists == True:
        # csv file needs to have atleast 1 word inside otherwise it gives error
        df = pd.read_csv('dataset_raw.csv', encoding='utf-8')
        if df.empty == True:

            df = pd.read_csv(url)
            df.to_csv('dataset_raw.csv', index=False)
        else:
            print("Load dataset: file not empty")

    else:
        print("Load dataset: file not exist")


#load_dataset(data_URL)
def extract_column_dataset():
    file_exists = os.path.exists('dataset_raw.csv')
    if file_exists == True:
        df = pd.read_csv('dataset_raw.csv', encoding='utf-8')
        if df.empty == True:
            print("Extract: file empty")
        else:
            file_exists_2 = os.path.exists('dataset_review.csv')
            review_df = pd.read_csv('dataset_review.csv', encoding='utf-8')
            if file_exists_2 == True and review_df.empty== True:
                review_df = df[['reviewText']]
                review_df.to_csv('dataset_review.csv', index=False)
            else:
                print("Extract: column extracted exists")
    else:
        print("Extract: file not exist")






# get embeddings for the reviews sampled from above URL and store it into .csv
def embeddings_to_csv():
    review_df = pd.read_csv('dataset_review.csv', encoding='utf-8')
    review_df = review_df.sample(100)
    #review_df["reviewText"] = review_df["reviewText"].str.replace('\n', ' ').str.strip()
    review_df["embedding_name"] = review_df["reviewText"].apply(lambda x: get_chatgpt_embedding(str(x)))
    review_df.to_csv('dataset_review_sampled.csv', index=False)
    print(review_df.head(5))
    print(review_df.shape)



# Clustering & dim. reduction
def vis_analysis():
    review_df = pd.read_csv('dataset_review_sampled.csv', encoding='utf-8')
    # when a list is stored in .csv format, is serialized as string, so needs to be deserialized into python lists:
    review_df["embedding_name"] = review_df["embedding_name"].apply(ast.literal_eval)
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(review_df["embedding_name"].tolist())

    reducer = UMAP()
    embeddings_2d = reducer.fit_transform(review_df["embedding_name"].tolist())

    fig = px.scatter(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], color=kmeans.labels_)
    fig.update_layout(
        xaxis=dict(range=[embeddings_2d[:, 0].min(), embeddings_2d[:, 0].max()]),
        yaxis=dict(range=[embeddings_2d[:, 1].min(), embeddings_2d[:, 1].max()])
    )
    fig.show()