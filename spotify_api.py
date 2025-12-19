import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from time import sleep

CLIENT_ID = "555741dcca4b449481dca7255f05d965"
CLIENT_SECRET = "ce41f9f7a32b48f79a1ea4d38f29775a"

print("Spotify API'ye bağlanıyor...")
auth_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(auth_manager=auth_manager)

# Mevcut veriyi yükle
print("Mevcut veri yükleniyor...")
df = pd.read_csv('spotify_data.csv')
print(f"Toplam {len(df)} şarkı bulundu.")

# Track ID'leri çekmek için playlist'ten tekrar çek
playlist_id = "2XvPIcOrbpieXZD4RmCujC"
print(f"Playlist'ten track ID'leri çekiliyor...")

results = sp.playlist_items(playlist_id, market="TR", limit=100)
all_items = results['items']
while results['next']:
    results = sp.next(results)
    all_items.extend(results['items'])
    sleep(0.5)

print(f"Toplam {len(all_items)} track bulundu.")

# Track ID'leri topla
track_ids = []
track_names = []
for item in all_items:
    track = item.get('track')
    if track and track.get('id'):
        track_ids.append(track['id'])
        track_names.append(track.get('name'))

print(f"\n{len(track_ids)} şarkı için audio features çekiliyor...")

# Audio features çek (100'lük gruplar halinde)
audio_features_list = []
for i in range(0, len(track_ids), 100):
    batch = track_ids[i:i+100]
    try:
        print(f"  {i+1}-{min(i+100, len(track_ids))} arası çekiliyor...")
        features = sp.audio_features(batch)
        audio_features_list.extend(features)
        sleep(0.5)  # Rate limit için
    except Exception as e:
        print(f"  HATA: {e}")
        # Hata olursa None ekle
        audio_features_list.extend([None] * len(batch))

print(f"\nAudio features başarıyla çekildi!")

# Audio features'ları dataframe'e dönüştür
audio_data = []
for idx, af in enumerate(audio_features_list):
    if af:
        audio_data.append({
            'danceability': af.get('danceability', 0),
            'energy': af.get('energy', 0),
            'loudness': af.get('loudness', 0),
            'speechiness': af.get('speechiness', 0),
            'acousticness': af.get('acousticness', 0),
            'instrumentalness': af.get('instrumentalness', 0),
            'liveness': af.get('liveness', 0),
            'valence': af.get('valence', 0),
            'tempo': af.get('tempo', 0),
            'key': af.get('key', 0),
            'mode': af.get('mode', 0)
        })
    else:
        # Eğer audio feature yoksa 0'larla doldur
        audio_data.append({
            'danceability': 0, 'energy': 0, 'loudness': 0,
            'speechiness': 0, 'acousticness': 0, 'instrumentalness': 0,
            'liveness': 0, 'valence': 0, 'tempo': 0, 'key': 0, 'mode': 0
        })

df_audio = pd.DataFrame(audio_data)

# Boyut kontrolü
if len(df) != len(df_audio):
    print(f"\nUYARI: Boyut uyuşmazlığı! df={len(df)}, audio={len(df_audio)}")
    # Küçük olanı al
    min_len = min(len(df), len(df_audio))
    df = df.iloc[:min_len]
    df_audio = df_audio.iloc[:min_len]
    print(f"Her iki dataframe de {min_len} satıra kesildi.")

# Birleştir
print("\nDataframe'ler birleştiriliyor...")
df_final = pd.concat([df.reset_index(drop=True), df_audio.reset_index(drop=True)], axis=1)

# Kaydet
output_file = 'spotify_full_data.csv'
df_final.to_csv(output_file, index=False, encoding='utf-8')

print(f"\n✅ BAŞARILI!")
print(f"📊 Toplam şarkı: {len(df_final)}")
print(f"📊 Toplam özellik: {len(df_final.columns)}")
print(f"💾 Dosya: {output_file}")
print(f"\n📈 Sütunlar:")
print(df_final.columns.tolist())
print(f"\n🎵 İlk 5 satır:")
print(df_final.head())