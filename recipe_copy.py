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
    sorted_indices = np.argsort(similarities)[::-1]  # 高い順

    # 上位5件
    recommendations = [
        {"name": music_names[i], "similarity": similarities[i], "file": os.path.join("music", os.path.basename(music_files[i]))}
        for i in sorted_indices[:5]
    ]
    return recommendations

# Streamlitアプリの構築
st.title("レシピに合う音楽推薦システム 🎵🍽️")

# データ読み込み
recipe_data, music_data = load_data()

# UI - レシピ選択（画像付き）
st.write("### 料理を選んでください:")
recipe_options = recipe_data.iloc[:, 0].tolist()  # 料理名のリスト
recipe_image_paths = ["images/" + os.path.basename(img) for img in recipe_data.iloc[:, 1]]

# 画像を横に並べる
cols = st.columns(5)  # 5列で表示
for i, col in enumerate(cols[:len(recipe_options)]):
    with col:
        st.image(recipe_image_paths[i], caption=recipe_options[i], width=150)

selected_recipe = st.radio("レシピを選択:", recipe_options, index=0)  # デフォルト値を設定
recipe_index = recipe_data.loc[recipe_data.iloc[:, 0] == selected_recipe].index[0]  # インデックス検索

# UI - ウェイト調整
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
        if os.path.exists(rec["file"]):  # ファイル存在チェック
            st.audio(rec["file"], format="audio/mp3")
        else:
            st.warning(f"⚠️ 音楽ファイルが見つかりません: {rec['file']}")
