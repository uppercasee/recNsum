import torch
import pandas as pd
import numpy as np
import pickle

# Load embeddings
itememb = torch.load("./embeddings/item_embeddings.pt")
useremb = torch.load("./embeddings/user_embeddings.pt")
print(itememb.shape, useremb.shape)

# read user2ind
with open("./embeddings/user2ind.pkl", "rb") as f:
    user2ind = pickle.load(f)
    print("user2ind loaded successfully.")

# read item2ind
with open("./embeddings/item2ind.pkl", "rb") as f:
    item2ind = pickle.load(f)
    print("item2ind loaded successfully.")

# load train.csv and raw_behaviour.csv
news_train = pd.read_csv("./datasets/train.csv")
raw_behaviour = pd.read_csv("./datasets/raw_behaviour.csv")
print("final data loaded successfully.")


# # Get the index of the item
# ind = user2ind.get("U13740")
# itememb_cpu = itememb.cpu()
# similarity = torch.nn.functional.cosine_similarity(itememb_cpu[ind], itememb_cpu, dim=-1)
# similarity_np = similarity.detach().numpy()
# most_sim_indices = similarity_np.argsort()[-10:][::-1]
# most_similar_articles = news_train.iloc[most_sim_indices]
# print(most_similar_articles.head())


def get_most_similar_articles(
    item_id, item_embeddings=itememb, news_data=news_train, top_n=5
):
    ind = user2ind.get(item_id)
    if not ind:
        print("user not found")
        ind = np.random.randint(1, 51)
    itememb_cpu = item_embeddings.cpu()
    similarity = torch.nn.functional.cosine_similarity(
        itememb_cpu[ind], itememb_cpu, dim=-1
    )
    similarity_np = similarity.detach().numpy()
    most_sim_indices = similarity_np.argsort()[-top_n:][::-1]
    most_similar_articles = news_data.iloc[most_sim_indices]
    return most_similar_articles


# userid = "U1374000"
# most_similar_articles = get_most_similar_articles(userid)
# most_similar_articles_json = most_similar_articles.to_json(orient="records")
# print(most_similar_articles_json)

# ind = item2ind.get("N55689")
# similarity = torch.nn.functional.cosine_similarity(itememb[ind], itememb, dim=-1)
# most_similar_indices = (similarity.argsort(descending=True).numpy()-1)
# most_sim_filtered = news_train[~news_train.ind.isna()].iloc[most_similar_indices]
# # Add the similarity as a new column in the DataFrame
# most_sim_filtered['similarity'] = similarity[most_similar_indices]
# # Display the DataFrame with similarity as a new column
# print(most_sim_filtered.head(10))

# def recommend_news_for_user(user_id, item_embeddings, user_data, news_data, top_n=10):
#     user_index = user2ind[user_id]
#     similarity_scores = torch.nn.functional.cosine_similarity(item_embeddings, item_embeddings[user_index].unsqueeze(0), dim=1)
#     sorted_indices = similarity_scores.argsort(descending=True)
#     user_interactions = set(user_data[user_data["userId"] == user_id]["impressions"])
#     recommended_news = [user_data.iloc[index.item()] for index in sorted_indices if user_data.iloc[index.item()]["impressions"] not in user_interactions]
#     recommended_news_df = pd.DataFrame(recommended_news)
#     return recommended_news_df.head(top_n)

# user_id = "U13740"  # User ID for which you want to recommend news
# recommended_news = recommend_news_for_user(user_id, useremb, raw_behaviour, news_train, top_n=5)
# print(recommended_news)

# # Define a function to recommend news articles for a given user ID
# def recommend_news_for_user(user_id, item_embeddings, user_data, news_data, top_n=10):
#     # Assuming you have a mapping between user IDs and indices (user2ind)
#     user_index = user2ind[user_id]

#     # Compute similarity scores between the user and all news articles
#     similarity_scores = torch.nn.functional.cosine_similarity(item_embeddings, item_embeddings[user_index].unsqueeze(0), dim=1)

#     # Sort news articles based on similarity scores
#     sorted_indices = similarity_scores.argsort(descending=True)

#     # Filter out news articles that the user has already interacted with
#     user_interactions = set(user_data[user_data["userId"] == user_id]["impressions"])
#     recommended_news = [user_data.iloc[index.item()] for index in sorted_indices if user_data.iloc[index.item()]["impressions"] not in user_interactions]

#     # Convert recommended_news list to DataFrame
#     recommended_news_df = pd.DataFrame(recommended_news)

#     # Return top-N recommended news articles as DataFrame
#     return recommended_news_df.head(top_n)

#     # # Return top-N recommended news articles
#     # return recommended_news[:top_n]

# # Example usage
# user_id = "U13740"  # User ID for which you want to recommend news
# recommended_news = recommend_news_for_user(user_id, useremb, raw_behaviour, news_train, top_n=5)
# print(recommended_news)

# # Define a function to recommend news articles for a given user ID
# def recommend_news_for_user(user_id, item_embeddings, user_data, news_data, top_n=10):
#     # Assuming you have a mapping between user IDs and indices (user2ind)
#     user_index = user2ind[user_id]

#     # Compute similarity scores between the user and all news articles
#     similarity_scores = torch.nn.functional.cosine_similarity(item_embeddings, item_embeddings[user_index].unsqueeze(0), dim=1)

#     # Sort news articles based on similarity scores
#     sorted_indices = similarity_scores.argsort(descending=True)

#     # Filter out news articles that the user has already interacted with
#     user_interactions = set(user_data[user_data["userId"] == user_id]["impressions"])
#     recommended_news = [user_data.iloc[index.item()] for index in sorted_indices if user_data.iloc[index.item()]["impressions"] not in user_interactions]

#     # Convert recommended_news list to DataFrame
#     recommended_news_df = pd.DataFrame(recommended_news)

#     # Return top-N recommended news articles as DataFrame
#     return recommended_news_df.head(top_n)

#     # # Return top-N recommended news articles
#     # return recommended_news[:top_n]

# # Example usage
# user_id = "U13740"  # User ID for which you want to recommend news
# recommended_news = recommend_news_for_user(user_id, useremb, raw_behaviour, news_train, top_n=5)
# print(recommended_news)

# def recommend_news_for_user(user_id, user_embeddings, user_data, news_data, top_n=10):
#     user_index = user2ind[user_id]  # Get the index of the user

#     # Compute similarity scores between the user and all news articles
#     similarity_scores = torch.nn.functional.cosine_similarity(user_embeddings, user_embeddings[user_index].unsqueeze(0), dim=1)

#     # Sort news articles based on similarity scores
#     sorted_indices = similarity_scores.argsort(descending=True)

#     # Filter out news articles that the user has already interacted with
#     user_interactions = set(user_data[user_data["userId"] == user_id]["impressions"])
#     recommended_news = [news_data.iloc[index.item()] for index in sorted_indices if news_data.iloc[index.item()]["news_id"] not in user_interactions]

#     # Convert recommended_news list to DataFrame
#     recommended_news_df = pd.DataFrame(recommended_news)

#     # Return top-N recommended news articles as DataFrame
#     return recommended_news_df.head(top_n)

# # Example usage
# user_id = "U13740"  # User ID for which you want to recommend news
# recommended_news = recommend_news_for_user(user_id, useremb, raw_behaviour, news_train, top_n=5)
# print(recommended_news)
