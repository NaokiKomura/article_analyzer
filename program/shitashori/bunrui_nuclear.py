#3.11原発事故に関係ありなしに分類するプログラム（ルールベース）
import pandas as pd

# データセットをロード
yomiuri = pd.read_csv('yomiuri_data/data/yomiuri_shasetu.txt', header=None, names=['title', 'content', 'source'])
mai = pd.read_csv('mainichi_data/data/mai_shasetu.txt', header=None, names=['title', 'content', 'source'])
asahi = pd.read_csv('asahi_data/data/asahi_shasetu.txt', header=None, names=['title', 'content', 'source'])

# キーワードの定義
keywords = ["原子力発電所", "放射能", "核燃料", "福島第一", "原発", "放射線", "放射線量", "核事故", "放射性物質"]

def classify_nuclear_related(row):
    count = sum([row['content'].count(keyword) for keyword in keywords])
    return 1 if count >= 2 else 0

# キーワードを用いて記事を分類
yomiuri['is_nuclear_related'] = yomiuri.apply(classify_nuclear_related, axis=1)
mai['is_nuclear_related'] = mai.apply(classify_nuclear_related, axis=1)
asahi['is_nuclear_related'] = asahi.apply(classify_nuclear_related, axis=1)

# 記事数の集計
yomiuri_count = yomiuri['is_nuclear_related'].sum()
mai_count = mai['is_nuclear_related'].sum()
asahi_count = asahi['is_nuclear_related'].sum()

# データセットの結合
all_articles = pd.concat([yomiuri, mai, asahi])

# ラベルが1のものと0のもので分類
nuclear_related_articles = all_articles[all_articles['is_nuclear_related'] == 1]
non_nuclear_related_articles = all_articles[all_articles['is_nuclear_related'] == 0]

#ファイルに保存
nuclear_related_file = "nuclear_related.txt"
non_nuclear_related_file = "nuclear_unrelated.txt"

nuclear_related_articles[['title', 'content', 'source']].to_csv(nuclear_related_file, index=False, header=None, sep=',')
non_nuclear_related_articles[['title', 'content', 'source']].to_csv(non_nuclear_related_file, index=False, header=None, sep=',')

nuclear_related_file, non_nuclear_related_file
