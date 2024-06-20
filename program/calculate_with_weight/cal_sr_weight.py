#主観度算出プログラム（重み付けあり）
import pandas as pd

# CSVファイルを読み込む
file_path = 'data/labeled_sentences.csv'  # CSVファイルのパスを適宜変更してください
data = pd.read_csv(file_path)

# 加重主観度の計算
# 加重主観度 = ある発信者の記事群内の意見文（ラベル1）の重みの合計 / ある発信者の記事群の文の重みの総数
weighted_subjectivity_by_sender = data.groupby('発信者ID').apply(
    lambda x: (x['分類ラベル'] * x['重み']).sum() / x['重み'].sum()
).reset_index(name='加重主観度')

# 計算結果を表示
print(weighted_subjectivity_by_sender)