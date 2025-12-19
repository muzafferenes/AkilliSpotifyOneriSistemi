
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*60)
print("MODEL PERFORMANS ANALIZI")
print("="*60)


print("\n[1/6] Veri yukleniyor...")
df = pd.read_csv('model_icin_hazir_veri.csv', decimal=',', thousands='.')

df['release_date'] = pd.to_datetime(df['release_date'], format='%d-%m-%Y', errors='coerce')
df.dropna(inplace=True)

print(f"  {len(df)} sarki yuklendi")

current_year = datetime.now().year
df['song_age'] = current_year - df['release_date'].dt.year

df_model = df.drop(columns=['name', 'artist', 'album', 'release_date', 'duration(min)'])

df_model['artist_pop_x_energy'] = df_model['artist_popularity'] * df_model['energy']
df_model['energy_x_valence'] = df_model['energy'] * df_model['valence']
df_model['dance_+_energy'] = df_model['danceability'] + df_model['energy']
df_model['pop_per_age'] = df_model['artist_popularity'] / (df_model['song_age'] + 1)
df_model['artist_flw_x_dance'] = df_model['artist_followers'] * df_model['danceability']
df_model['followers_log'] = np.log1p(df_model['artist_followers'])

df_model = df_model[(df_model['song_popularity'] >= 20) & (df_model['song_popularity'] <= 70)]

X = df_model.drop('song_popularity', axis=1)
y = df_model['song_popularity']

X = pd.get_dummies(X, columns=['genre'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"  Train set: {len(X_train)} sarki")
print(f"  Test set: {len(X_test)} sarki")
print(f"  Ozellik sayisi: {X_train.shape[1]}")


print("\n[2/6] Modeller egitiliyor...")

models = {}
predictions = {}

print("  Random Forest...")
rf = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=5, 
                           min_samples_leaf=2, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)
models['Random Forest'] = rf
predictions['Random Forest'] = rf.predict(X_test_scaled)

print("  XGBoost...")
xgb_model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, 
                         subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
xgb_model.fit(X_train_scaled, y_train)
models['XGBoost'] = xgb_model
predictions['XGBoost'] = xgb_model.predict(X_test_scaled)

print("  CatBoost...")
cb = CatBoostRegressor(iterations=100, depth=6, learning_rate=0.1, random_state=42, verbose=False)
cb.fit(X_train_scaled, y_train)
models['CatBoost'] = cb
predictions['CatBoost'] = cb.predict(X_test_scaled)

print("  Tamamlandi!")



print("\n[3/6] Performans metrikleri hesaplaniyor...")

results = []
for name, y_pred in predictions.items():
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    results.append({
        'Model': name,
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'R2': round(r2, 3),
        'MAPE': round(mape, 2)
    })
    
    print(f"\n  {name}:")
    print(f"    MAE:  {mae:.2f}")
    print(f"    RMSE: {rmse:.2f}")
    print(f"    R2:   {r2:.3f}")
    print(f"    MAPE: {mape:.2f}%")

results_df = pd.DataFrame(results)


print("\n[4/6] Model karsilastirma tablosu olusturuluyor...")

fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('tight')
ax.axis('off')

table = ax.table(cellText=results_df.values,
                colLabels=results_df.columns,
                cellLoc='center',
                loc='center',
                colColours=['#4CAF50']*len(results_df.columns))

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

for i in range(len(results_df.columns)):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

best_mae_idx = results_df['MAE'].idxmin() + 1
best_r2_idx = results_df['R2'].idxmax() + 1

table[(best_mae_idx, 1)].set_facecolor('#90EE90')
table[(best_r2_idx, 3)].set_facecolor('#90EE90')

plt.title('Model Karsilastirma Tablosu', fontsize=14, fontweight='bold', pad=20)
plt.savefig('1_model_comparison_table.png', dpi=300, bbox_inches='tight', facecolor='white')
print("  Kaydedildi: 1_model_comparison_table.png")



print("\n[5/6] Metrik karsilastirma grafikleri olusturuluyor...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

colors_mae = ['#FF6B6B', '#4ECDC4', '#45B7D1']
bars1 = ax1.barh(results_df['Model'], results_df['MAE'], color=colors_mae)
ax1.set_xlabel('Mean Absolute Error (MAE)', fontsize=11, fontweight='bold')
ax1.set_title('MAE Karsilastirmasi\n(Dusuk = Iyi)', fontsize=12, fontweight='bold')
ax1.invert_yaxis()

for i, bar in enumerate(bars1):
    width = bar.get_width()
    ax1.text(width + 0.2, bar.get_y() + bar.get_height()/2, 
             f'{width:.2f}', ha='left', va='center', fontweight='bold')

colors_r2 = ['#95E1D3', '#F38181', '#AA96DA']
bars2 = ax2.barh(results_df['Model'], results_df['R2'], color=colors_r2)
ax2.set_xlabel('R² Score', fontsize=11, fontweight='bold')
ax2.set_title('R² Karsilastirmasi\n(Yuksek = Iyi)', fontsize=12, fontweight='bold')
ax2.invert_yaxis()

for i, bar in enumerate(bars2):
    width = bar.get_width()
    ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
             f'{width:.3f}', ha='left', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('2_metrics_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
print("  Kaydedildi: 2_metrics_comparison.png")



print("\n[6/6] Feature importance grafigi olusturuluyor...")

importances = rf.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values('Importance', ascending=False).head(15)

fig, ax = plt.subplots(figsize=(10, 8))
colors_importance = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_importance_df)))

bars = ax.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color=colors_importance)
ax.set_xlabel('Onem Derecesi', fontsize=11, fontweight='bold')
ax.set_title('En Onemli 15 Ozellik (Random Forest)', fontsize=13, fontweight='bold', pad=15)
ax.invert_yaxis()

for i, bar in enumerate(bars):
    width = bar.get_width()
    ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
            f'{width:.3f}', ha='left', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('3_feature_importance.png', dpi=300, bbox_inches='tight', facecolor='white')
print("  Kaydedildi: 3_feature_importance.png")



print("\n[BONUS] Tahmin vs Gercek grafigi olusturuluyor...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (name, y_pred) in enumerate(predictions.items()):
    ax = axes[idx]
    
    ax.scatter(y_test, y_pred, alpha=0.5, s=20, edgecolors='k', linewidth=0.5)
    
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax.set_xlabel('Gercek Deger', fontweight='bold')
    ax.set_ylabel('Tahmin', fontweight='bold')
    ax.set_title(f'{name}\nR² = {r2_score(y_test, y_pred):.3f}', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('4_prediction_scatter.png', dpi=300, bbox_inches='tight', facecolor='white')
print("  Kaydedildi: 4_prediction_scatter.png")



print("\n[BONUS] Hata dagilimi grafigi olusturuluyor...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (name, y_pred) in enumerate(predictions.items()):
    ax = axes[idx]
    
    errors = y_test - y_pred
    
    ax.hist(errors, bins=30, alpha=0.7, color=colors_mae[idx], edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Sifir Hata')
    
    ax.set_xlabel('Hata (Gercek - Tahmin)', fontweight='bold')
    ax.set_ylabel('Frekans', fontweight='bold')
    ax.set_title(f'{name}\nOrtalama Hata: {errors.mean():.2f}', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('5_error_distribution.png', dpi=300, bbox_inches='tight', facecolor='white')
print("  Kaydedildi: 5_error_distribution.png")


print("\n" + "="*60)
print("ANALIZ TAMAMLANDI!")
print("="*60)

best_model = results_df.loc[results_df['MAE'].idxmin(), 'Model']
print(f"\nEn iyi model: {best_model}")

print("\nOlusturulan dosyalar:")
print("  1. 1_model_comparison_table.png")
print("  2. 2_metrics_comparison.png")
print("  3. 3_feature_importance.png")
print("  4. 4_prediction_scatter.png")
print("  5. 5_error_distribution.png")

print("\n" + "="*60)