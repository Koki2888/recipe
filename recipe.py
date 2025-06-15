import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

# データ読み込み関数
@st.cache_data
def load_data():
    recipe_data = pd.read_csv("co類似度中間発表s1.csv")  # レシピデータ
    music_data = pd.read_csv("co類似度中間発表s2.csv")  # 音楽データ
    return recipe_data, music_data

# 類似度計算関数
def recommend(recipe_index, weights):
    recipe_vector = recipe_data.iloc[recipe_index, 2:5].values  # 特徴量 (感情値, 彩度, 明度)
    music_features = music_data.iloc[:, 1:4].values  # 音楽の特徴量
    music_names = music_data.iloc[:, 0].values  # 音楽名
    music_files = music_data.iloc[:, 4].values  # MP3ファイルパス

    # ウェイト適用して類似度計算
    weighted_recipe = recipe_vector * weights
    weighted_music = music_features * weights
    similarities = cosine_similarity([weighted_recipe], weighted_music)[0]
    sorted_indices = np.argsort(similarities)[::-1]  # 高い順に並び替え

    # 上位5件を推薦
    recommendations = [
        {
            "name": music_names[i],
            "similarity": similarities[i],
            "file": os.path.join("music", os.path.basename(music_files[i]))
        }
        for i in sorted_indices[:5]
    ]
    return recommendations

# Streamlitアプリの構築
st.title("レシピに合う音楽推薦システム")

# データ読み込み
recipe_data, music_data = load_data()

# UI - レシピ選択（ドロップダウン形式）
st.write("### 料理を選んでください:")
recipe_options = recipe_data.iloc[:, 0].tolist()  # 料理名のリスト
selected_recipe = st.selectbox("レシピを選択:", recipe_options)

# 選択されたレシピのインデックスを取得
recipe_index = recipe_data.loc[recipe_data.iloc[:, 0] == selected_recipe].index[0]

# 選択されたレシピの画像を表示
recipe_image_path = "images/" + os.path.basename(recipe_data.iloc[recipe_index, 1])
st.image(recipe_image_path, caption=selected_recipe, width=300)

# UI - 特徴量のウェイト調整
st.write("### 特徴量の重要度を調整:")
sentiment_weight = st.slider("感情値の重み", 0.0, 2.0, 1.0, 0.1)
saturation_weight = st.slider("彩度の重み", 0.0, 2.0, 1.0, 0.1)
brightness_weight = st.slider("明度の重み", 0.0, 2.0, 1.0, 0.1)
weights = np.array([sentiment_weight, saturation_weight, brightness_weight])

# 推薦ボタン
if st.button("🎵 音楽を推薦"):
    recommendations = recommend(recipe_index, weights)

    st.write("### 推薦された音楽:")
    for rec in recommendations:
        st.write(f"🎶 {rec['name']} (Similarity: {rec['similarity']:.2f})")
        if os.path.exists(rec["file"]):
            st.audio(rec["file"], format="audio/mp3")
        else:
            st.warning(f"⚠️ 音楽ファイルが見つかりません: {rec['file']}")
