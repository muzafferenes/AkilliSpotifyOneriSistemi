"""
Spotify Hit Prediction & Recommendation System
Streamlit Web Uygulaması
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics.pairwise import cosine_similarity

# Sayfa ayarları
st.set_page_config(
    page_title="Spotify ML Projesi",
    page_icon="♪",
    layout="wide"
)

# CSS - Sadece kartlar için
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    /* Başlıktan link ikonunu kaldır */
    h1 a {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# VERİ YÜKLEME VE MODEL EĞİTİMİ
# ============================================

@st.cache_data
def load_and_prepare_data():
    """Veriyi yükle ve hazırla"""
    df = pd.read_csv('model_icin_hazir_veri.csv', decimal=',', thousands='.')
    df['release_date'] = pd.to_datetime(df['release_date'], format='%d-%m-%Y', errors='coerce')
    df.dropna(inplace=True)
    
    # Özellik mühendisliği
    current_year = datetime.now().year
    df['song_age'] = current_year - df['release_date'].dt.year
    
    return df

@st.cache_resource
def train_models(df):
    """Modelleri eğit"""
    df_model = df.copy()
    
    # Gereksiz sütunları çıkar
    df_model = df_model.drop(columns=['name', 'artist', 'album', 'release_date', 'duration(min)'])
    
    # Feature engineering
    df_model['artist_pop_x_energy'] = df_model['artist_popularity'] * df_model['energy']
    df_model['energy_x_valence'] = df_model['energy'] * df_model['valence']
    df_model['dance_+_energy'] = df_model['danceability'] + df_model['energy']
    df_model['pop_per_age'] = df_model['artist_popularity'] / (df_model['song_age'] + 1)
    df_model['artist_flw_x_dance'] = df_model['artist_followers'] * df_model['danceability']
    df_model['followers_log'] = np.log1p(df_model['artist_followers'])
    
    # Popülerlik filtresi
    df_model = df_model[(df_model['song_popularity'] >= 20) & (df_model['song_popularity'] <= 70)]
    
    X = df_model.drop('song_popularity', axis=1)
    y = df_model['song_popularity']
    
    # One-hot encoding
    X = pd.get_dummies(X, columns=['genre'], drop_first=True)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Modelleri eğit
    models = {}
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    models['Random Forest'] = rf
    
    # XGBoost
    xgb = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
    xgb.fit(X_train_scaled, y_train)
    models['XGBoost'] = xgb
    
    # CatBoost
    cb = CatBoostRegressor(iterations=100, depth=6, learning_rate=0.1, random_state=42, verbose=False)
    cb.fit(X_train_scaled, y_train)
    models['CatBoost'] = cb
    
    return models, scaler, X.columns

# ============================================
# ÖNERİ SİSTEMİ
# ============================================

def get_recommendations(df, selected_indices, n_recommendations=10):
    """Content-based öneri sistemi"""
    audio_features = ['danceability', 'energy', 'valence', 'tempo']
    
    # Feature matrix
    feature_matrix = df[audio_features].values
    
    # Seçilen şarkıların ortalama profili
    user_profile = feature_matrix[selected_indices].mean(axis=0).reshape(1, -1)
    
    # Benzerlik hesapla
    similarities = cosine_similarity(user_profile, feature_matrix)[0]
    
    # Seçilen şarkıları çıkar
    mask = np.ones(len(similarities), dtype=bool)
    mask[selected_indices] = False
    similarities_filtered = similarities.copy()
    similarities_filtered[~mask] = -1
    
    # En benzer şarkıları al
    top_indices = similarities_filtered.argsort()[-n_recommendations:][::-1]
    
    recommendations = df.iloc[top_indices].copy()
    recommendations['similarity_score'] = similarities[top_indices]
    
    return recommendations

# ============================================
# ANA UYGULAMA
# ============================================

# Ana başlık
st.title("Spotify Hit Tahmin ve Öneri Sistemi")
st.markdown("Machine Learning ile Şarkı Popülerlik Tahmini ve Kişiselleştirilmiş Öneri")
st.markdown("---")

# Veriyi yükle
with st.spinner('Veri yükleniyor...'):
    df = load_and_prepare_data()

# Modelleri eğit
with st.spinner('Modeller eğitiliyor...'):
    models, scaler, feature_names = train_models(df)

st.success(f'{len(df)} şarkı yüklendi ve modeller eğitildi!')

# ============================================
# TABS
# ============================================

tab1, tab2 = st.tabs(["Hit Tahmin", "Şarkı Öneri"])

# ============================================
# TAB 1: HIT TAHMİN
# ============================================

with tab1:
    st.header("Hit Tahmin Sistemi")
    st.markdown("Şarkı özelliklerini girerek popülerlik tahmini yapın")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Audio Features")
        danceability = st.slider("Danceability", 0.0, 1.0, 0.7, 0.01)
        energy = st.slider("Energy", 0.0, 1.0, 0.7, 0.01)
        valence = st.slider("Valence (Pozitiflik)", 0.0, 1.0, 0.5, 0.01)
        tempo = st.slider("Tempo (BPM)", 60, 200, 120, 1)
    
    with col2:
        st.subheader("Sanatçı & Şarkı Bilgileri")
        artist_pop = st.slider("Sanatçı Popülerliği", 0, 100, 60, 1)
        artist_followers = st.number_input("Sanatçı Takipçi Sayısı", 0, 10000000, 1000000, 10000)
        song_age = st.slider("Şarkı Yaşı (yıl)", 0, 50, 5, 1)
        
        genre = st.selectbox("Genre", ['pop', 'trap', 'alt-rock', 'arabesk', 'drill', 'huzunlu-pop', 
                                       'indie-pop', 'klasik-rock', 'pop-rock', 'funk-rock', 'old-rap', 'alt-rap', 'other'])
    
    if st.button("Tahmin Yap", type="primary"):
        # Feature engineering
        input_data = {
            'artist_followers': artist_followers,
            'artist_popularity': artist_pop,
            'danceability': danceability,
            'energy': energy,
            'valence': valence,
            'tempo': tempo,
            'song_age': song_age,
            'artist_pop_x_energy': artist_pop * energy,
            'energy_x_valence': energy * valence,
            'dance_+_energy': danceability + energy,
            'pop_per_age': artist_pop / (song_age + 1),
            'artist_flw_x_dance': artist_followers * danceability,
            'followers_log': np.log1p(artist_followers)
        }
        
        # Genre one-hot encoding
        for col in feature_names:
            if col.startswith('genre_'):
                genre_name = col.replace('genre_', '')
                input_data[col] = 1 if genre == genre_name else 0
        
        # DataFrame oluştur
        input_df = pd.DataFrame([input_data])
        
        # Eksik kolonları ekle
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Sıralama
        input_df = input_df[feature_names]
        
        # Scale et
        input_scaled = scaler.transform(input_df)
        
        # Tahmin yap
        st.markdown("---")
        st.subheader("Tahmin Sonuçları")
        
        cols = st.columns(3)
        for idx, (name, model) in enumerate(models.items()):
            pred = model.predict(input_scaled)[0]
            
            with cols[idx]:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{name}</h3>
                    <h1>{pred:.1f}</h1>
                    <p>Popülerlik Skoru</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Ortalama
        avg_pred = np.mean([model.predict(input_scaled)[0] for model in models.values()])
        st.markdown(f"### Ortalama Tahmin: **{avg_pred:.1f}**")
        
        if avg_pred >= 60:
            st.success("Bu şarkı HIT olabilir!")
        elif avg_pred >= 40:
            st.info("Orta düzey popülerlik bekleniyor")
        else:
            st.warning("Düşük popülerlik bekleniyor")

# ============================================
# TAB 2: ŞARKı ÖNERİ
# ============================================

with tab2:
    st.header("Şarkı Öneri Sistemi")
    st.markdown("Beğendiğiniz şarkılara benzer şarkılar keşfedin")
    
    # Şarkı seçimi
    st.subheader("Şarkı Seçin (3-5 tane)")
    
    # Arama kutusu
    search = st.text_input("Şarkı Ara", placeholder="Şarkı adı veya sanatçı...")
    
    if search:
        filtered_df = df[df['name'].str.contains(search, case=False, na=False) | 
                        df['artist'].str.contains(search, case=False, na=False)]
    else:
        filtered_df = df.head(50)
    
    # Şarkı listesi
    song_options = [f"{row['name']} - {row['artist']}" for idx, row in filtered_df.iterrows()]
    selected_songs = st.multiselect("Şarkılarınızı seçin:", song_options, max_selections=5)
    
    if len(selected_songs) >= 3:
        if st.button("Benzer Şarkıları Bul", type="primary"):
            # Seçilen şarkıların index'lerini bul
            selected_indices = []
            for song in selected_songs:
                song_name = song.split(' - ')[0]
                idx = df[df['name'] == song_name].index[0]
                selected_indices.append(idx)
            
            # Önerileri al
            recommendations = get_recommendations(df, selected_indices, n_recommendations=10)
            
            # Kullanıcı profili
            st.markdown("---")
            st.subheader("Müzik Profiliniz")
            
            user_profile = df.loc[selected_indices, ['danceability', 'energy', 'valence', 'tempo']].mean()
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Danceability", f"{user_profile['danceability']:.2f}")
            col2.metric("Energy", f"{user_profile['energy']:.2f}")
            col3.metric("Valence", f"{user_profile['valence']:.2f}")
            col4.metric("Tempo", f"{user_profile['tempo']:.0f} BPM")
            
            # Radar chart
            categories = ['Danceability', 'Energy', 'Valence']
            values = [user_profile['danceability'], user_profile['energy'], user_profile['valence']]
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Profiliniz'
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=False,
                title="Müzik Zevki Profili"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Öneriler
            st.markdown("---")
            st.subheader("Size Özel Şarkı Önerileri")
            
            for idx, row in recommendations.iterrows():
                col1, col2, col3 = st.columns([3, 2, 1])
                
                with col1:
                    st.markdown(f"**{row['name']}**")
                    st.caption(f"{row['artist']}")
                
                with col2:
                    st.caption(f"Genre: {row['genre']}")
                    st.caption(f"Popülerlik: {row['song_popularity']}")
                
                with col3:
                    similarity_pct = row['similarity_score'] * 100
                    st.metric("Benzerlik", f"{similarity_pct:.0f}%")
                
                st.divider()
    
    elif selected_songs:
        st.info(f"En az 3 şarkı seçmelisiniz (Seçili: {len(selected_songs)})")
    else:
        st.info("Yukarıdan şarkı arayın ve seçin")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.9rem;'>
    <p>Spotify ML Projesi | Streamlit & Python</p>
</div>
""", unsafe_allow_html=True)