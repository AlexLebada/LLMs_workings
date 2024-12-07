import pandas as pd
from manage_data import load_dataset, extract_column_dataset
from embbedings_to_db import get_similarities, embeddings_to_db,retrieve_text


data_URL = "https://raw.githubusercontent.com/keitazoumana/Experimentation-Data/main/Musical_instruments_reviews.csv"

if 1==0:
    load_dataset(data_URL)
    extract_column_dataset()
    #get embeddings for column dataset
    embeddings_to_db("reviewText", "dataset_review.csv")


# calculate similarities
get_similarities("milk espresso")
# get top n similar texts
results = retrieve_text(8)
print(results)
