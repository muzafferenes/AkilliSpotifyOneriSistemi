

import pandas as pd
import json
import random



def ms_to_min_sec(ms): 
    """Milisaniyeyi dakika:saniye formatına çevir"""
    seconds = ms // 1000
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    return f"{minutes}:{remaining_seconds:02d}"

def complete_date(tarih_str): 
    """Eksik tarihleri tamamla (sadece yıl varsa 01-01 ekle)"""
    if not isinstance(tarih_str, str): 
        return None
    if len(tarih_str) == 4 and tarih_str.isdigit():
        return f"{tarih_str}-01-01"
    return tarih_str

def normalize_turkish(name):
    """Türkçe karakterleri normalize et"""
    if not isinstance(name, str):
        return ""
    
    name = name.strip().lower()
    name = name.replace('i̇', 'i')
    name = name.replace("ı", "i")
    name = name.replace("ç", "c")
    name = name.replace("ş", "s")
    name = name.replace("ğ", "g")
    name = name.replace("ü", "u")
    name = name.replace("ö", "o")
    return name



genre_profiles = {
    "trap": {"danceability": 0.71, "energy": 0.60, "valence": 0.45, "tempo": 142},
    "arabesk": {"danceability": 0.58, "energy": 0.65, "valence": 0.38, "tempo": 105},
    "drill": {"danceability": 0.83, "energy": 0.66, "valence": 0.35, "tempo": 144},
    "pop": {"danceability": 0.74, "energy": 0.72, "valence": 0.68, "tempo": 120},
    "huzunlu-pop": {"danceability": 0.55, "energy": 0.52, "valence": 0.29, "tempo": 115},
    "indie-pop": {"danceability": 0.63, "energy": 0.68, "valence": 0.55, "tempo": 124},
    "klasik-rock": {"danceability": 0.48, "energy": 0.78, "valence": 0.59, "tempo": 128},
    "pop-rock": {"danceability": 0.59, "energy": 0.81, "valence": 0.54, "tempo": 135},
    "alt-rock": {"danceability": 0.52, "energy": 0.84, "valence": 0.47, "tempo": 132},
    "funk-rock": {"danceability": 0.65, "energy": 0.80, "valence": 0.69, "tempo": 118},
    "old-rap": {"danceability": 0.81, "energy": 0.69, "valence": 0.58, "tempo": 95},
    "alt-rap": {"danceability": 0.75, "energy": 0.63, "valence": 0.41, "tempo": 125},
    "other": {"danceability": 0.62, "energy": 0.67, "valence": 0.51, "tempo": 122}
}

def generate_features_from_profile(row): 
    """Genre profiline göre audio features üret"""
    genre = row["genre"]
    base_profile = genre_profiles.get(genre, genre_profiles["other"])
    
    new_danceability = max(0.0, min(1.0, random.uniform(
        base_profile["danceability"] - 0.05, 
        base_profile["danceability"] + 0.05
    )))
    new_energy = max(0.0, min(1.0, random.uniform(
        base_profile["energy"] - 0.07, 
        base_profile["energy"] + 0.07
    )))
    new_valence = max(0.0, min(1.0, random.uniform(
        base_profile["valence"] - 0.06, 
        base_profile["valence"] + 0.06
    )))
    new_tempo = random.uniform(
        base_profile["tempo"] - 8, 
        base_profile["tempo"] + 8
    )
    
    return pd.Series({
        "danceability": round(new_danceability, 3), 
        "energy": round(new_energy, 3), 
        "valence": round(new_valence, 3), 
        "tempo": round(new_tempo, 0)
    })


try:
    random.seed(42)  
    

    df = pd.read_csv('spotify_temel_liste.csv')
    print(f"[OK] {len(df)} sarki yuklendi")
    

    json_dosya_yolu = "sanatciler_ve_turleri.json"
    with open(json_dosya_yolu, 'r', encoding='utf-8') as f:
        artist_genres = json.load(f)
    

    artist_genres_normalized = {normalize_turkish(k): v for k, v in artist_genres.items()}
    

    df["genre"] = df["artist"].apply(
        lambda x: artist_genres_normalized.get(normalize_turkish(x), "other")
    )
    
    print(f"\nGenre dagilimi:")
    print(df["genre"].value_counts())
    
   
    print("\nAudio features uretiliyor...")
    audio_features = df.apply(generate_features_from_profile, axis=1)
    
  
    df_final = pd.concat([df, audio_features], axis=1)
    
  
    df_final['release_date'] = df_final['release_date'].apply(complete_date)
    df_final['release_date_dt'] = pd.to_datetime(df_final['release_date'], errors='coerce')
    df_final['release_date'] = df_final['release_date_dt'].dt.strftime('%d-%m-%Y')

    df_final['artist_followers'] = df_final['artist_followers'].apply(
        lambda x: f"{int(x):,}".replace(',', '.')
    )
    
    df_final['duration(min)'] = df_final['duration_ms'].apply(ms_to_min_sec)
    
    df_final = df_final.drop(columns=['duration_ms', 'artist_genres', 'release_date_dt'], errors='ignore')
    
    final_dosya_adi = "model_icin_hazir_veri.csv"
    df_final.to_csv(final_dosya_adi, index=False, encoding='utf-8', decimal=',')
    
    print(f"\n[OK] ADIM 2 BASARILI! '{final_dosya_adi}' dosyasi olusturuldu")
    print(f"[INFO] {len(df_final)} sarki, {len(df_final.columns)} sutun")
    
    print("\nIlk 5 satir:")
    print(df_final.head())

except FileNotFoundError as e:
    print(f"[ERROR] Dosya bulunamadi: {e}")
except Exception as e:
    print(f"[ERROR] ADIM 2'DE HATA! {e}")