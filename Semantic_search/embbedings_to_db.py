from pymongo import MongoClient
from API_integration.LLM_API import get_chatgpt_embedding
from cosine_similarity_1d import calculate_1d_cosine_similarity
import pandas as pd
import ast
from API_integration.MONGO_CLIENT import MONGO_DB_KEY

cluster = MongoClient(MONGO_DB_KEY)
db = cluster["LLM_work"]
collection = db["embeddings"]


def embeddings_to_db(column_name: str, file_name: str):
    review_df = pd.read_csv(file_name,encoding='utf-8')
    review_df.rename(columns={column_name: 'Text'}, inplace=True)
    review_df = review_df.sample(26)
    review_df["embedding_name"] = review_df["Text"].apply(lambda x: get_chatgpt_embedding(str(x)))

    data_to_insert = review_df.to_dict(orient='records')
    collection.insert_many(data_to_insert)


#embeddings_to_db("reviewText", "dataset_review.csv")
#collection.update_many({}, {"$set": {"similarities": 0.0}})



# not used
def embeddings_to_file():
        review_df = pd.read_csv('dataset_review_sampled.csv', encoding='utf-8')
        review_df["embedding_name"] = review_df["embedding_name"].apply(ast.literal_eval)  # string to list
        #review_df["similarity"] = review_df["embedding_name"].apply(lambda x: calculate_1d_cosine_similarity(x, search_text_embedding))
        review_df.to_csv('dataset_review_sampled_2.csv', index=False)


def get_one_similarity(input_text):
    search_text = input_text
    search_text_embedding = get_chatgpt_embedding(search_text)

    random_document = collection.aggregate([{"$sample": {"size": 1}}])
    for doc in random_document:
        db_embedding = doc.get("embedding_name")
        similarity = calculate_1d_cosine_similarity(search_text_embedding, db_embedding)
        print(similarity)
        update_result = collection.update_one(
            {"_id": doc["_id"]},  # identify the doc
            {"$set": {"similarities": similarity}}  # update field
        )
        if update_result.modified_count > 0:
            print(f"Document with _id {doc['_id']} updated successfully.")
        else:
            print(f"No update was made for _id {doc['_id']}.")




def get_similarities(input_text):
    search_text = input_text
    search_text_embedding = get_chatgpt_embedding(search_text)
    random_document = collection.find()
    for doc in random_document:
        db_embedding = doc.get("embedding_name")
        similarity = calculate_1d_cosine_similarity(search_text_embedding, db_embedding)
        print(similarity)
        update_result = collection.update_one(
            {"_id": doc["_id"]},
            {"$set": {"similarities": similarity}}
        )
    random_document = collection.find()

def retrieve_text(n_results):
    similarities_dict = []
    random_document = collection.find()
    for doc in random_document:
        similarity = doc.get("similarities")
        if similarity is not None:
            similarities_dict.append({
                #"_id": doc["_id"],
                "Text": doc["Text"],
                "similarity": similarity
            })
    sorted_similarities = sorted(similarities_dict, key=lambda x:x['similarity'], reverse=True)
    return sorted_similarities[:n_results]




