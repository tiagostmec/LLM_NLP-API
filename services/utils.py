import requests
import time
import pandas as pd

def get_anime_data(anime_id):
    url = f"https://api.jikan.moe/v4/anime/{anime_id}"
    response = requests.get(url)
    time.sleep(0.35)  
    if response.status_code == 200:
        data = response.json().get('data', {})
        
        anime_info = {
            "title": data.get("title", "N/A").strip(),
            "synopsis": data.get("synopsis", "N/A").strip(),
            "genre": ", ".join(sorted([genre["name"].strip() for genre in data.get("genres", [])])),
            "rating": data.get("rating", "N/A").strip(),
            "score": float(data.get("score", "0")) if data.get("score") else 0,
            "episodes": data.get("episodes", 0),
            "airing_status": data.get("status", "N/A").strip(),
            "type": data.get("type", "N/A").strip(),
            "members": int(data.get("members", 0))
        }
        
        train_data = [
            {
                "prompt": f"Tell me the plot of {anime_info['title']}",
                "response": anime_info['synopsis']
            },
            {
                "prompt": f"Tell me the genre of {anime_info['title']}",
                "response": anime_info['genre']
            },
            {
                "prompt": f"Tell me the rating of {anime_info['title']}",
                "response": anime_info['rating']
            },
            {
                "prompt": f"Tell me about {anime_info['title']}",
                "response": f"{anime_info['title']} is a {anime_info['type']} anime with {anime_info['episodes']} episodes. It has a rating of {anime_info['rating']} and a score of {anime_info['score']}. The anime is currently {anime_info['airing_status']}."
            }
        ]    
        return train_data
    else:
        print(f"Erro ao obter dados para anime_id {anime_id}: {response.status_code}")
        return None


def get_anime_ids_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    return df['anime_id'].tolist()  