
import pandas as pd
from asari.api import Sonar
from tqdm import tqdm
tqdm.pandas()

# CSVファイルを読み込む
df = pd.read_csv('data/classified/classified_v4.csv')

# '分類ラベル'が1の行を削除
df = df[df['分類ラベル'] != 1]

# 重み付けのルールを定義
keywords = {
    1.0: ["絶対", "100%", "必ず", "絶対に", "間違いなく", "確実に"],
    0.75: ["だろう", "であろう", "ろう", "でしょう", "違いない", "間違いない", "疑いない", "はずだ", "はず", "可能性が高い", "おそれが強い", "疑いが強い", "確信する", "信じる", "予測する", "考える", "予想する", "かもしれない", "おそらく", "多分", "きっと"],
    0.5: ["かもしれない", "かも", "かもわからない", "可能性がある", "おそれがある", "疑いがある", "可能性", "おそれ", "疑い", "のではないか", "のではないだろうか", "思う", "疑う", "ありうる", "保証はない", "確信はない", "確証はない"],
    0.25: ["可能性は低い", "おそれは低い", "可能性はあまりない", "まい", "思わない", "思えない", "考えない", "考えられない", "信じない", "信じられない"],
    0.0: ["可能性はない", "おそれはない", "疑いはない", "あり得ない"]
}

# 文に対して重み付けを適用
def apply_weight(row):
    if row['分類ラベル'] == 0:  # 主観的記述の場合
        text = row['文']
        for weight, key_list in keywords.items():
            if any(key in text for key in key_list):
                return weight
        return 1.0  # キーワードがない場合

df['重み'] = df.progress_apply(apply_weight, axis=1)

# 変更をCSVに保存
df.to_csv('data/classified_v7.csv', index=False)
