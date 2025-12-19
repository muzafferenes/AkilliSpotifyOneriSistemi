import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

CLIENT_ID = "555741dcca4b449481dca7255f05d965"
CLIENT_SECRET = "ce41f9f7a32b48f79a1ea4d38f29775a"

try:
    auth_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
    sp = spotipy.Spotify(auth_manager=auth_manager)
    playlist_id = "2XvPIcOrbpieXZD4RmCujC"
    
    results = sp.playlist_items(playlist_id, market="TR", limit=100)
    all_items = results['items']
    while results['next']:
        results = sp.next(results)
        all_items.extend(results['items'])
    print(f"Çalma listesinden toplam {len(all_items)} şarkı meta verisi çekildi.")

    # Benzersiz sanatçı ID'lerini toplama
    print("Benzersiz sanatçı ID'leri toplanıyor...")
    artist_ids = []
    for item in all_items:
        track = item.get('track')
        if track and track.get('artists'):
            artist = track['artists'][0]
            artist_id = artist.get('id')
            if artist_id:
                artist_ids.append(artist_id)
    
    unique_artist_ids = list(set(artist_ids))
    
    # Sanatçı verilerini 50'lik gruplar halinde topluca çekme
    print(f"{len(unique_artist_ids)} benzersiz sanatçı için veriler toplu halde çekiliyor...")
    artist_info_map = {}
    for i in range(0, len(unique_artist_ids), 50):
        batch = unique_artist_ids[i:i+50]
        batch_results = sp.artists(batch)
        
        if batch_results and 'artists' in batch_results:
            for artist in batch_results['artists']:
                if artist:
                    artist_info_map[artist['id']] = artist
    
    data = []
    print("Şarkı verileri işleniyor...")
    for item in all_items:
        track = item.get('track')
        if track and track.get('artists'):
            artists = track.get('artists')
            artist_name = artists[0].get('name')
            artist_id = artists[0].get('id')
            if not artist_id: continue

            artist_info = artist_info_map.get(artist_id)

            if artist_info:
                followers = artist_info.get('followers', {}).get('total', 0)
                genres = ", ".join(artist_info.get('genres', []))
                artist_pop = artist_info.get('popularity', 0)
            else:
                followers, genres, artist_pop = 0, "", 0
            
            data.append({
                'name': track.get('name'), 'artist': artist_name,
                'artist_followers': followers, 'artist_genres': genres,
                'artist_popularity': artist_pop, 'song_popularity': track.get('popularity', 0),
                'album': track.get('album', {}).get('name'),
                'release_date': track.get('album', {}).get('release_date'),
                'duration_ms': track.get('duration_ms', 0)
            })

    df_temel = pd.DataFrame(data)
    dosya_adi = 'spotify_temel_liste.csv'
    df_temel.to_csv(dosya_adi, index=False, encoding='utf-8')
    print(f"ADIM 1 BAŞARILI! '{dosya_adi}' dosyası oluşturuldu.")

except Exception as e:
    print(f"ADIM 1'DE HATA! Veri çekilemedi. Hata mesajı: {e}")