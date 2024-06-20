#実験用データセット作成のためのプログラム
import pandas as pd
import re

# ファイルの読み込み
file_path = 'data/nuclear_related.txt'
df = pd.read_csv(file_path, names=['記事タイトル', '記事本文', '発信新聞社'])

# 発信者IDのマッピング
sender_id_map = {'読売新聞社': '0', '毎日新聞社': '1', '朝日新聞社': '2'}

# 記事IDと発信者IDの割り当て
df['記事ID'] = df.index + 1
df['発信者ID'] = df['発信新聞社'].map(sender_id_map)

# 文単位での分割
def split_into_sentences(text):
    # 「」で囲まれた部分を保護
    protected_parts = re.findall(r'「.*?」', text)
    for i, part in enumerate(protected_parts):
        text = text.replace(part, f'PROTECTED{i}', 1)

    # 。で分割
    sentences = re.split(r'(?<=[^「」。])。', text)

    # 保護された部分を元に戻す
    for i, part in enumerate(protected_parts):
        sentences = [sentence.replace(f'PROTECTED{i}', part) for sentence in sentences]

    return [sentence for sentence in sentences if sentence]

# 各文に記事IDと発信者IDを割り当て
rows = []
for _, row in df.iterrows():
    sentences = split_into_sentences(row['記事本文'])
    for sentence in sentences:
        rows.append([sentence, row['記事ID'], row['発信者ID']])

# 新しいDataFrameを作成
df_sentences = pd.DataFrame(rows, columns=['文', '記事ID', '発信者ID'])

print(df_sentences.head())  # 確認のため最初の数行を表示