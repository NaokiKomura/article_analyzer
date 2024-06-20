#主観度算出のプログラム
import pandas as pd

# CSVファイルを読み込む
file_path = 'data/labeled_sentences.csv'  # CSVファイルのパスを適宜変更してください
data = pd.read_csv(file_path)

# 主観度の計算
# 主観度 = ある発信者の記事群内の意見文の数 / ある発信者の記事群の文の総数
subjectivity_by_sender = data.groupby('発信者ID').apply(
    lambda x: x['分類ラベル'].sum() / len(x)
).reset_index(name='主観度')

# 計算結果を表示
print(subjectivity_by_sender)
