import json, re, numpy as np, matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

JSONL_PATH = "gmu_cs_courses_with_prereqs.jsonl"

#load jsonl data into a list
data = []
with open(JSONL_PATH, 'r', encoding='utf-8') as file:
    for line in file:
        l = json.loads(line)
        data.append(l)
#print(data[:2])



#creating individual lists in regards to their field
course_ids = []
course_names = []
course_summaries = []
course_information = []


lemm = WordNetLemmatizer()
stop = set(stopwords.words("english"))
for course in data:
    course_id = course.get("course_id")
    course_name = course.get("course_name")
    course_summary = course.get("course_summary")
    full_text = f"{course_name} {course_summary}"


    #NLTK preprocessing  
    text = full_text.lower() #lower text 
    text = re.sub(r"[^a-z\s]", " ", text) #remove anything that is not a alphabet character 
    tokenized_text = text.split() #tokenizes the text ['x','y','z',...]
    
    filtered_tokens = [] #filter these words even further
    for word in tokenized_text:
        if (word not in stop and len(word) > 2):
            lemma = lemm.lemmatize(word) #'learning' -> 'learn'
            filtered_tokens.append(lemma)

    clean_text = " ".join(filtered_tokens) #['x','y','z',...] -> ['x y z ...']

    course_ids.append(course_id)
    course_names.append(course_name)
    course_summaries.append(course_summary)
    course_information.append(clean_text)

print(course_information[:2])

vectorizer = TfidfVectorizer()
matrix_X = vectorizer.fit_transform(course_information)
print(matrix_X)

#k-means
k = 8
km = KMeans(n_clusters=k, random_state=42, n_init="auto")
labels = km.fit_predict(matrix_X)
print(labels)

#top terms in per cluster 
feature_names = np.array(vectorizer.get_feature_names_out())
for cluster in range(k):
    idx = np.where(labels == cluster)[0]
    if len(idx) == 0:
        continue
    cluster_vector = matrix_X[idx]
    centroid_vector = np.asarray(cluster_vector.mean(axis=0))
    centroid_vector = np.ravel(centroid_vector)
    top_10 = np.argsort(-centroid_vector)[:10]
    terms = feature_names[top_10]
    print(f"Cluster {cluster}: {', '.join(terms)}")

#tsne
tsne = TSNE(n_components=2, random_state=42, perplexity=15, init="pca")
tsne = tsne.fit_transform(matrix_X.toarray())

#plotting
plt.figure(figsize=(10,7))
for c in range(k):
    mask = (labels == c)
    if mask.any():
        plt.scatter(tsne[mask,0], tsne[mask,1], label=f"Cluster {c}", s=40)


#labels
for i, (x, y) in enumerate(tsne):
    if i % max(1, len(tsne)//40) == 0:
        plt.text(x, y, course_ids[i], fontsize=8)

plt.title("GMU CS Courses — KMeans(k=15) + t-SNE")
plt.legend(markerscale=1.2, fontsize=6)
plt.tight_layout()
plt.show()
