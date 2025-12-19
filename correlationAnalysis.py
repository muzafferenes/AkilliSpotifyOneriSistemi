
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("coolwarm")

print("="*60)
print("KORELASYON ANALIZI VE CROSS-VALIDATION")
print("="*60)



print("\n[1/4] Veri yukleniyor...")
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


print("\n[2/4] Korelasyon matrisleri olusturuluyor...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

audio_features = ['danceability', 'energy', 'valence', 'tempo', 
                  'artist_popularity', 'song_popularity']
corr_audio = df_model[audio_features].corr()

mask1 = np.triu(np.ones_like(corr_audio, dtype=bool))
sns.heatmap(corr_audio, mask=mask1, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
            ax=ax1, vmin=-1, vmax=1)
ax1.set_title('Audio Features Korelasyon Matrisi', fontsize=13, fontweight='bold', pad=15)

all_numeric = df_model.select_dtypes(include=[np.number])
corr_with_pop = all_numeric.corr()['song_popularity'].sort_values(ascending=False)
corr_with_pop = corr_with_pop.drop('song_popularity')

top_corr = corr_with_pop.head(15)
colors = ['green' if x > 0 else 'red' for x in top_corr.values]

ax2.barh(range(len(top_corr)), top_corr.values, color=colors, alpha=0.7)
ax2.set_yticks(range(len(top_corr)))
ax2.set_yticklabels(top_corr.index, fontsize=9)
ax2.set_xlabel('Korelasyon Katsayisi', fontweight='bold')
ax2.set_title('Populerlik ile En Yuksek Korelasyonlar (Top 15)', 
              fontsize=13, fontweight='bold', pad=15)
ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax2.grid(axis='x', alpha=0.3)

for i, v in enumerate(top_corr.values):
    ax2.text(v + 0.01 if v > 0 else v - 0.01, i, f'{v:.3f}', 
             va='center', ha='left' if v > 0 else 'right', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig('6_correlation_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
print("  Kaydedildi: 6_correlation_analysis.png")



print("\n[3/4] Detayli korelasyon matrisi olusturuluyor...")

numeric_features = ['artist_followers', 'artist_popularity', 'song_popularity',
                    'danceability', 'energy', 'valence', 'tempo', 'song_age',
                    'artist_pop_x_energy', 'energy_x_valence', 'dance_+_energy',
                    'pop_per_age', 'followers_log']

corr_full = df_model[numeric_features].corr()

fig, ax = plt.subplots(figsize=(12, 10))

mask = np.triu(np.ones_like(corr_full, dtype=bool))
sns.heatmap(corr_full, mask=mask, annot=True, fmt='.2f', cmap='RdYlBu_r',
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
            ax=ax, vmin=-1, vmax=1)

ax.set_title('Tum Ozellikler Korelasyon Matrisi', fontsize=14, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig('7_full_correlation_matrix.png', dpi=300, bbox_inches='tight', facecolor='white')
print("  Kaydedildi: 7_full_correlation_matrix.png")



print("\n[4/4] Cross-validation yapiliyor...")


X = df_model.drop('song_popularity', axis=1)
y = df_model['song_popularity']


X = pd.get_dummies(X, columns=['genre'], drop_first=True)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


models_cv = {
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'XGBoost': XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1),
    'CatBoost': CatBoostRegressor(iterations=100, depth=6, learning_rate=0.1, random_state=42, verbose=False)
}

cv_results = []

for name, model in models_cv.items():
    print(f"\n  {name}...")
    
    cv_scores_mae = -cross_val_score(model, X_scaled, y, cv=5, 
                                      scoring='neg_mean_absolute_error', n_jobs=-1)
    cv_scores_r2 = cross_val_score(model, X_scaled, y, cv=5, 
                                   scoring='r2', n_jobs=-1)
    
    cv_results.append({
        'Model': name,
        'CV MAE Mean': cv_scores_mae.mean(),
        'CV MAE Std': cv_scores_mae.std(),
        'CV R2 Mean': cv_scores_r2.mean(),
        'CV R2 Std': cv_scores_r2.std()
    })
    
    print(f"    MAE: {cv_scores_mae.mean():.2f} (+/- {cv_scores_mae.std():.2f})")
    print(f"    R2:  {cv_scores_r2.mean():.3f} (+/- {cv_scores_r2.std():.3f})")

cv_df = pd.DataFrame(cv_results)


print("\n[BONUS] Cross-validation sonuc tablosu olusturuluyor...")

fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('tight')
ax.axis('off')

table_data = []
for _, row in cv_df.iterrows():
    table_data.append([
        row['Model'],
        f"{row['CV MAE Mean']:.2f}",
        f"{row['CV MAE Std']:.2f}",
        f"{row['CV R2 Mean']:.3f}",
        f"{row['CV R2 Std']:.3f}"
    ])

table = ax.table(cellText=table_data,
                colLabels=['Model', 'MAE (Mean)', 'MAE (Std)', 'R² (Mean)', 'R² (Std)'],
                cellLoc='center',
                loc='center',
                colColours=['#4CAF50']*5)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

for i in range(5):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

best_mae_idx = cv_df['CV MAE Mean'].idxmin() + 1
best_r2_idx = cv_df['CV R2 Mean'].idxmax() + 1

table[(best_mae_idx, 1)].set_facecolor('#90EE90')
table[(best_r2_idx, 3)].set_facecolor('#90EE90')

plt.title('5-Fold Cross-Validation Sonuclari', fontsize=14, fontweight='bold', pad=20)
plt.savefig('8_cross_validation_results.png', dpi=300, bbox_inches='tight', facecolor='white')
print("  Kaydedildi: 8_cross_validation_results.png")


print("\n[BONUS] Cross-validation grafikleri olusturuluyor...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))


x_pos = np.arange(len(cv_df))
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

ax1.bar(x_pos, cv_df['CV MAE Mean'], yerr=cv_df['CV MAE Std'], 
        color=colors, alpha=0.7, capsize=5, error_kw={'linewidth': 2})
ax1.set_xticks(x_pos)
ax1.set_xticklabels(cv_df['Model'])
ax1.set_ylabel('MAE (Mean +/- Std)', fontweight='bold')
ax1.set_title('Cross-Validation MAE Karsilastirmasi', fontweight='bold', fontsize=12)
ax1.grid(axis='y', alpha=0.3)


for i, (mean, std) in enumerate(zip(cv_df['CV MAE Mean'], cv_df['CV MAE Std'])):
    ax1.text(i, mean + std + 0.3, f'{mean:.2f}', ha='center', fontweight='bold')


ax2.bar(x_pos, cv_df['CV R2 Mean'], yerr=cv_df['CV R2 Std'], 
        color=colors, alpha=0.7, capsize=5, error_kw={'linewidth': 2})
ax2.set_xticks(x_pos)
ax2.set_xticklabels(cv_df['Model'])
ax2.set_ylabel('R² Score (Mean +/- Std)', fontweight='bold')
ax2.set_title('Cross-Validation R² Karsilastirmasi', fontweight='bold', fontsize=12)
ax2.grid(axis='y', alpha=0.3)


for i, (mean, std) in enumerate(zip(cv_df['CV R2 Mean'], cv_df['CV R2 Std'])):
    ax2.text(i, mean + std + 0.02, f'{mean:.3f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('9_cross_validation_charts.png', dpi=300, bbox_inches='tight', facecolor='white')
print("  Kaydedildi: 9_cross_validation_charts.png")


print("\n" + "="*60)
print("ANALIZ TAMAMLANDI!")
print("="*60)

print("\nOlusturulan dosyalar:")
print("  6. 6_correlation_analysis.png (Audio features + Top korelasyonlar)")
print("  7. 7_full_correlation_matrix.png (Tum ozellikler)")
print("  8. 8_cross_validation_results.png (CV sonuc tablosu)")
print("  9. 9_cross_validation_charts.png (CV bar chart'lar)")

print("\nKorelasyon Bulgulari:")
print(f"  En yuksek pozitif korelasyon: {corr_with_pop.iloc[0]:.3f} ({corr_with_pop.index[0]})")
print(f"  En yuksek negatif korelasyon: {corr_with_pop.iloc[-1]:.3f} ({corr_with_pop.index[-1]})")

print("\nCross-Validation Sonuclari:")
best_model_cv = cv_df.loc[cv_df['CV MAE Mean'].idxmin(), 'Model']
print(f"  En iyi model (CV MAE): {best_model_cv}")
print(f"  Ortalama MAE: {cv_df['CV MAE Mean'].min():.2f} (+/- {cv_df.loc[cv_df['CV MAE Mean'].idxmin(), 'CV MAE Std']:.2f})")
print(f"  Ortalama R2: {cv_df.loc[cv_df['CV MAE Mean'].idxmin(), 'CV R2 Mean']:.3f}")

print("\n" + "="*60)