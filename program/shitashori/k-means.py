#k-meansによるクラスタリング処理のプログラム（主観的記述の固有度で使用）
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import nltk
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# NLTKのデータをダウンロード（トークン化に必要）
nltk.download('punkt')

# CSVファイルの読み込み
df = pd.read_csv('data/related_ or_unrelated/nuclear_related.txt')

# データの前処理
processed_docs = [word_tokenize(doc.lower()) for doc in df['text']]  

# Word2Vecモデルのトレーニング
model = Word2Vec(sentences=processed_docs, vector_size=100, window=5, min_count=1, workers=4)

# 記事をベクトル化
def document_vector(word2vec_model, doc):
    return np.mean([word2vec_model.wv[word] for word in doc if word in word2vec_model.wv], axis=0)

doc_vectors = [document_vector(model, doc) for doc in processed_docs]

'''
# K-meansクラスタリングとSSEの計算
k = 4  # クラスタの数
kmeans_model = KMeans(n_clusters=k, random_state=42)
kmeans_model.fit(doc_vectors)

# クラスタリングの結果
labels = kmeans_model.labels_

# SSEの出力
sse = kmeans_model.inertia_
print(f"SSE: {sse}")

# クラスタリングの評価
silhouette_avg = silhouette_score(doc_vectors, labels)
print(f"Silhouette Score: {silhouette_avg}")

# 記事IDをDataFrameに追加（1からの通し番号）
df['article_id'] = range(1, len(df) + 1)

# 記事IDとクラスタラベルを含むDataFrameの作成
output_df = pd.DataFrame({'article_id': df['article_id'], 'cluster_label': labels})

# CSVファイルとして書き出し
output_df.to_csv('clustered_articles.csv', index=False)
'''
'''
k=4
kmeans = KMeans(n_clusters=k, random_state=42)
cluster_labels = kmeans.fit_predict(doc_vectors)

# PCAによる次元削減
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(doc_vectors)

# 散布図のプロット
plt.figure(figsize=(10, 6))
for i in range(k):
    plt.scatter(reduced_data[cluster_labels == i, 0], reduced_data[cluster_labels == i, 1], label=f'Cluster {i}')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D Scatter Plot of Clusters')
plt.legend()
plt.show()
'''

range_n_clusters = range(2, 11)

sse = []
silhouette_avg = []

for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(doc_vectors)
    
    # SSE
    sse.append(kmeans.inertia_)

    # シルエットスコア
    silhouette_avg.append(silhouette_score(doc_vectors, cluster_labels))

# SSEのプロット
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range_n_clusters, sse, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('SSE')
plt.title('Elbow Method For Optimal k')

# シルエットスコアのプロット
plt.subplot(1, 2, 2)
plt.plot(range_n_clusters, silhouette_avg, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score For Each k')

#plt.tight_layout()
#plt.show()
print(sse)
print(silhouette_avg)