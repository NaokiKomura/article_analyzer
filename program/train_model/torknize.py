import pandas as pd
from pyknp import Juman

def tokenize_with_juman(text):
    juman = Juman(timeout=3600)  # タイムアウトを60秒に設定
    result = juman.analysis(text)
    tokens = [mrph.midasi for mrph in result.mrph_list()]
    return ' '.join(tokens)

def main():
    # CSVファイルを読み込む
    df = pd.read_csv('data/classification_data_5.csv')

    # 各テキストデータをJuman++でトークナイズ
    df['text'] = df['text'].apply(tokenize_with_juman)

    # トークナイズされたデータを新しいCSVファイルとして保存
    df.to_csv('data/tokenized_classification_data_5.csv', index=False)

if __name__ == "__main__":
    main()
