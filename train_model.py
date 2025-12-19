"""
ADIM 3: Model Eğitimi ve Karşılaştırma
Random Forest, XGBoost, CatBoost modellerini eğitir
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt

# ============================================
# VERİ YÜKLEME VE TEMİZLEME
# ============================================

print("📂 Veri yükleniyor...")
df = pd.read_csv('model_icin_hazir_veri.csv', decimal=',', thousands='.')

# Tarihi parse et
df['release_date'] = pd.to_datetime(df['release_date'], format='%d-%m-%Y', errors='coerce')

# Eksik tarihleri temizle
df.dropna(inplace=True)
print(f"✅ {len(df)} şarkı yüklendi (temizleme sonrası)")

# ============================================
# ÖZELLİK MÜHENDİSLİĞİ (Feature Engineering)
# ============================================

print("\n🔧 Özellik mühendisliği yapılıyor...")

# Şarkının yaşı
current_year = datetime.now().year
df['song_age'] = current_year - df['release_date'].dt.year

# Gereksiz sütunları çıkar
df = df.drop(columns=['name', 'artist', 'album', 'release_date', 'duration(min)'])

# Etkileşim özellikleri
df['artist_pop_x_energy'] = df['artist_popularity'] * df['energy']
df['energy_x_valence'] = df['energy'] * df['valence']
df['dance_+_energy'] = df['danceability'] + df['energy']
df['pop_per_age'] = df['artist_popularity'] / (df['song_age'] + 1)
df['artist_flw_x_dance'] = df['artist_followers'] * df['danceability']
df['followers_log'] = np.log1p(df['artist_followers'])

print(f"  ✅ {6} yeni özellik eklendi")

# ============================================
# POPÜLERLİK FİLTRESİ
# ============================================

print("\n🎯 Popülerlik aralığı filtreleniyor (20-70)...")
initial_len = len(df)
df = df[(df['song_popularity'] >= 20) & (df['song_popularity'] <= 70)]
print(f"  {len(df)} şarkı kaldı ({initial_len - len(df)} filtrelendi)")

# ============================================
# VERİ HAZIRLIĞI
# ============================================

X = df.drop('song_popularity', axis=1)
y = df['song_popularity']

# One-Hot Encoding (genre için)
X = pd.get_dummies(X, columns=['genre'], drop_first=True)

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n📊 Train set: {len(X_train)} şarkı")
print(f"📊 Test set: {len(X_test)} şarkı")
print(f"📊 Toplam özellik: {X_train.shape[1]}")

# ============================================
# MODEL EĞİTİMİ VE KARŞILAŞTIRMA
# ============================================

def evaluate_model(model, X_test, y_test, model_name):
    """Modeli değerlendir"""
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    print(f"\n📊 {model_name} Sonuçları:")
    print(f"  MAE (Ortalama Hata): {mae:.2f}")
    print(f"  R² (Açıklanan Varyans): {r2:.2f}")
    print(f"  MAPE (Yüzdesel Sapma): {mape:.2f}%")
    
    return {'model': model_name, 'MAE': mae, 'R2': r2, 'MAPE': mape}

# ============================================
# MODEL 1: RANDOM FOREST
# ============================================

print("\n" + "="*60)
print("🌲 RANDOM FOREST EĞİTİLİYOR")
print("="*60)

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_scaled, y_train)
rf_results = evaluate_model(rf_model, X_test_scaled, y_test, "Random Forest")

# ============================================
# MODEL 2: XGBOOST
# ============================================

print("\n" + "="*60)
print("🚀 XGBOOST EĞİTİLİYOR")
print("="*60)

xgb_model = XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train_scaled, y_train)
xgb_results = evaluate_model(xgb_model, X_test_scaled, y_test, "XGBoost")

# ============================================
# MODEL 3: CATBOOST
# ============================================

print("\n" + "="*60)
print("🐱 CATBOOST EĞİTİLİYOR")
print("="*60)

cb_model = CatBoostRegressor(
    iterations=100,
    depth=6,
    learning_rate=0.1,
    random_state=42,
    verbose=False
)

cb_model.fit(X_train_scaled, y_train)
cb_results = evaluate_model(cb_model, X_test_scaled, y_test, "CatBoost")

# ============================================
# KARŞILAŞTIRMA TABLOSU
# ============================================

print("\n" + "="*60)
print("📊 MODEL KARŞILAŞTIRMASI")
print("="*60)

comparison_df = pd.DataFrame([rf_results, xgb_results, cb_results])
print(comparison_df.to_string(index=False))

# En iyi model
best_model = comparison_df.loc[comparison_df['MAE'].idxmin(), 'model']
print(f"\n🏆 En iyi model: {best_model}")

# ============================================
# FEATURE IMPORTANCE (Random Forest)
# ============================================

print("\n" + "="*60)
print("🎯 EN ÖNEMLİ ÖZELLİKLER (Random Forest)")
print("="*60)

importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1][:10]
features = X.columns[indices]

importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances[indices]
}).sort_values('Importance', ascending=False)

print(importance_df.to_string(index=False))

# Görselleştir
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.gca().invert_yaxis()
plt.title("En Önemli 10 Özellik (Random Forest)")
plt.xlabel("Önem Derecesi")
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("\n📊 Grafik kaydedildi: feature_importance.png")

print("\n✅ ADIM 3 TAMAMLANDI!")