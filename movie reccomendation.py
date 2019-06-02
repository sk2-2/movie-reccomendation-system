import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_title_from_index(index):
    return (df[df.index == index]["title"].values[0])

def get_index_from_title(title):
    return (df[df.title == title]["index"].values[0])

df = pd.read_csv("movie_dataset.csv")
#print(df.head())

features = ['keywords','cast','genres','director']

#To fill the NA's in the keyword
for feature in features:
	df[feature] = df[feature].fillna('')

def combine_features(row):
    return row['keywords'] +" "+row['cast']+" "+row["genres"]+" "+row["director"]

df["combined_features"] = df.apply(combine_features,axis=1 )

cv=CountVectorizer()

count_matrix = cv.fit_transform(df["combined_features"])

cosine_sim = cosine_similarity(count_matrix)

movie_name = "avatar"

movie_index = get_index_from_title(movie_name)

similar_movies = list(enumerate(cosine_sim[movie_index]))

sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse = True)

i = 0
print("Since you liked ",end =":")
print(movie_name,"You might also like:")
for element in sorted_similar_movies:
    
    i=i+1
    print(i,end =" : ")
    print(get_title_from_index(element[0]))
    if(i>50):
        break
