import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- NLP prep with NLTK ---
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK assets
try:
    _ = stopwords.words("english")
except LookupError:
    nltk.download("stopwords")
try:
    _ = nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")
try:
    _ = nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# --- ML / Viz ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.metrics.pairwise import cosine_distances

# ===============================
# Config
# ===============================
JSONL_PATH = "gmu_cs_courses_with_prereqs.jsonl"
USE_CLUSTERER = "kmeans"
N_CLUSTERS = 15
TSNE_PERPLEXITY = 15
TSNE_SEED = 42

# Load data
rows = []
with open(JSONL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))
df = pd.DataFrame(rows)
#print(df)

# build the text field we’ll embed: title + summary
df["text"] = (df["course_name"].fillna("") + " " + df["course_summary"].fillna("")).str.strip()
#print(df['text'])

# Clean text (NLTK)
lemmatizer = WordNetLemmatizer()
stop = set(stopwords.words("english"))

def normalize(text: str) -> str:
    # lowercase
    text = text.lower()
    # keep letters only
    text = re.sub(r"[^a-z\s]", " ", text)
    # tokenize
    tokens = text.split()
    # drop stopwords + short tokens, lemmatize
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop and len(t) > 2]
    return " ".join(tokens)

df["clean"] = df["text"].apply(normalize)

# ===============================
# TF-IDF features
# ===============================
vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_df=0.9)
X = vectorizer.fit_transform(df["clean"])

# ===============================
# Clustering
# ===============================
if USE_CLUSTERER == "kmeans":
    km = KMeans(n_clusters=N_CLUSTERS, random_state=TSNE_SEED, n_init="auto")
    labels = km.fit_predict(X)
    centers = km.cluster_centers_
    clusterer_name = f"KMeans(k={N_CLUSTERS})"
else:
    # Heuristic bandwidth from pairwise distances on a small sample
    sample = min(200, X.shape[0])
    sample_idx = np.random.RandomState(TSNE_SEED).choice(X.shape[0], sample, replace=False)
    # MeanShift expects dense; use cosine distances to estimate a scale
    # Convert a small sample to dense carefully (TF-IDF is okay for small sample)
    X_sample_dense = X[sample_idx].toarray()
    bw = estimate_bandwidth(X_sample_dense, quantile=0.2, n_samples=sample)
    ms = MeanShift(bandwidth=bw if bw > 0 else None, bin_seeding=True)
    labels = ms.fit_predict(X_sample_dense)
    # Need labels for full set; map via nearest sample center:
    # Build cluster representatives as sample medoids
    unique_labs = np.unique(labels)
    reps = []
    for lab in unique_labs:
        idx = sample_idx[labels == lab]
        reps.append(idx[0])
    reps = np.array(reps)
    # assign full set by nearest rep (cosine)
    dists = cosine_distances(X, X[reps])
    labels_full = dists.argmin(axis=1)
    labels = labels_full
    clusterer_name = f"MeanShift(~{len(np.unique(labels))} clusters)"

df["cluster"] = labels

# ===============================
# Print top terms per cluster
# ===============================
def top_terms_for_cluster(cluster_id, topn=10):
    # average tf-idf vector for cluster
    idx = np.where(labels == cluster_id)[0]
    if len(idx) == 0:
        return []
    centroid = np.asarray(X[idx].mean(axis=0)).ravel()
    top_idx = np.argsort(-centroid)[:topn]
    terms = np.array(vectorizer.get_feature_names_out())[top_idx]
    return terms.tolist()

print(f"\n== {clusterer_name} ==")
for c in sorted(np.unique(labels)):
    terms = top_terms_for_cluster(c, topn=10)
    print(f"Cluster {c:02d}: {', '.join(terms)}")

# ===============================
# 2D t-SNE for visualization
# ===============================
# t-SNE on dense is typical; for small datasets this is fine
X_dense = X.toarray()
tsne = TSNE(n_components=2, random_state=TSNE_SEED, perplexity=TSNE_PERPLEXITY, init="pca")
Z = tsne.fit_transform(X_dense)

# ===============================
# Plot (matplotlib only; no custom colors)
# ===============================
plt.figure(figsize=(10, 7))
# draw points, one cluster at a time (no explicit colors set)
for c in sorted(np.unique(labels)):
    mask = labels == c
    plt.scatter(Z[mask, 0], Z[mask, 1], label=f"Cluster {c}", s=40)

# annotate a few points with course_id for reference (avoid clutter)
for i, (x, y) in enumerate(Z):
    if i % max(1, len(Z)//40) == 0:  # label ~40 points
        plt.text(x, y, df.iloc[i]["course_id"], fontsize=8)

plt.title(f"GMU CS Courses — {clusterer_name} + t-SNE")
plt.legend(markerscale=1.2, fontsize=8)
plt.tight_layout()
plt.show()
