import re
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def preprocess_tweet(tweet):
    # Remove tweet id and timestamp
    tweet = re.sub(r'^.*?:', '', tweet)
    # Remove words starting with '@' symbol
    tweet = re.sub(r'\B@\w+', '', tweet)
    # Remove hashtag symbols
    tweet = re.sub(r'#', '', tweet)
    # Remove URLs
    tweet = re.sub(r'http\S+|www\S+', '', tweet)
    # Convert to lowercase
    tweet = tweet.lower()
    return tweet

def jaccard_distance(tweet1, tweet2):
    set1 = set(tweet1)
    set2 = set(tweet2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return 1 - (intersection / union)
# Function to perform K-means clustering
def kmeans_clustering(tweets, k):
    tweet_list = [preprocess_tweet(tweet) for tweet in tweets]

    # Create a count vectorizer
    vectorizer = CountVectorizer(binary=True)

    # Create a term-document matrix
    X = vectorizer.fit_transform(tweet_list)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)

    # Assign each tweet to its corresponding cluster
    labels = kmeans.labels_
    clusters = {}
    for i, label in enumerate(labels):
        if label in clusters:
            clusters[label].append(tweets[i])
        else:
            clusters[label] = [tweets[i]]

    return clusters

# Load tweets from file
tweet_file = "https://github.com/HusamAldulaimi/tweets-K-mean-clustering/blob/main/bbchealth.txt"
tweets = []
with open(tweet_file, 'r', encoding='utf-8') as file:
    tweets = file.readlines()

# Preprocess tweets
preprocessed_tweets = [preprocess_tweet(tweet) for tweet in tweets]

# Vectorize preprocessed tweets using TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(preprocessed_tweets)

# Define values of K for clustering
k_values = [2, 3, 4, 5, 6]

# Perform K-means clustering with different values of K
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
    labels = kmeans.labels_
    clusters = kmeans_clustering(tweets, k)

    # Print clusters and their tweets
    print(f"Clustering with K={k}")
    for i in range(k):
        cluster_tweets = [tweets[j] for j in range(len(tweets)) if labels[j] == i]
        print(f"Cluster {i + 1}:")
        for tweet in cluster_tweets:
            print(tweet)
        print()

    # Compute centroid of each cluster
    centroids = kmeans.transform(X).argmin(axis=0)
    centroid_tweets = [tweets[i] for i in centroids]

    # Print centroids
    print("Centroids:")
    for i, centroid in enumerate(centroid_tweets):
        print(f"Cluster {i+1}: {centroid}")
    print()

    # Compute SSE and size of each cluster
    sse = 0
    cluster_sizes = []
    for cluster in clusters.values():
        centroid = np.mean(jaccard_distance(np.array(tweets.index(tweet1,tweet2) for tweet in cluster)), axis=0)

        centroid = np.mean(jaccard_distance(np.array((tweets.index(tweet2) for tweet in cluster))), axis=0)
        sse += sum((jaccard_distance(tweets.index(tweet1))[0] - centroid[0])**2 for tweet in cluster)

        sse += sum((jaccard_distance(tweets.index(tweet2))[0] - centroid[0])**2 for tweet in cluster)
        cluster_sizes.append(len(cluster))

    results.append((k, sse, cluster_sizes))

    # Compute Jaccard Distance between each tweet and its cluster's centroid
    jaccard_distances = []
    for i, tweet in enumerate(preprocessed_tweets):
        distance = jaccard_distance(tweet.split(), centroid_tweets[labels[i]].split())
        jaccard_distances.append(distance)

    # Print Jaccard Distances
    print("Jaccard Distances:")
    for i, distance in enumerate(jaccard_distances):
        print(f"Tweet {i+1}: {distance}")
    print("----------------------------------------")
    # Report the results
for result in results:
    print(f"\nValue of K: {result[0]}")
    print(f"Sum of Squared Error (SSE): {result[1]}")
    print("Size of each Cluster:")
    for i, size in enumerate(result[2]):
        print(f"Cluster {i+1}: {size}")


