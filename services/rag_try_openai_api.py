import requests
import pandas as pd
import unicodedata
import openai

openai.api_key = 'af'
def get_anime_info(anime_id):
    url = f'https://api.jikan.moe/v4/anime/{anime_id}'
    response = requests.get(url)
    data = response.json()

    if response.status_code == 200:
        anime_info = {
            'title': data['data']['title'],
            'synopsis': data['data']['synopsis'],
            'genres': [genre['name'] for genre in data['data']['genres']],
            'episodes': data['data']['episodes'],
            'score': data['data']['score'],
            'aired': data['data']['aired']['string']
        }
        return anime_info
    else:
        return None

def normalize_string(s):
    return unicodedata.normalize('NFD', s).encode('ascii', 'ignore').decode('ascii')

def get_anime_id_from_csv(anime_name, csv_path="C:\\Users\\th_sm\\Desktop\\LLM_NLP API\\data\\anime.csv"):
    df = pd.read_csv(csv_path)
    
    anime_name_normalized = normalize_string(anime_name.lower())
    
    df['normalized_name'] = df['name'].apply(lambda x: normalize_string(x.lower()))
    
    anime = df[df['normalized_name'].str.contains(anime_name_normalized, na=False)]
    
    if not anime.empty:
        return anime.iloc[0]['anime_id']
    else:
        return None
    
def answer_anime_question_gpt3(anime_name):
    anime_id = get_anime_id_from_csv(anime_name)
    
    if anime_id is None:
        return f"Desculpe, não encontrei o anime '{anime_name}' no banco de dados."
    
    anime_info = get_anime_info(anime_id)
    
    if not anime_info:
        return "Desculpe, não consegui encontrar informações sobre esse anime."
    
    prompt = f"""
        Você é um assistente especializado em animes. Um usuário pediu informações sobre o anime "{anime_info['title']}". Aqui estão os detalhes:

        - Título: {anime_info['title']}
        - Gêneros: {', '.join(anime_info['genres'])}
        - Episódios: {anime_info['episodes']}
        - Nota: {anime_info['score']}
        - Sinopse: {anime_info['synopsis']}
        - Exibido em: {anime_info['aired']}

        Com base nesses detalhes, escreva uma descrição detalhada, envolvente e fácil de entender sobre o anime. Não adicione informações irrelevantes ou inventadas.
        """
    response = openai.completions.create(
        model="o1-mini", 
        prompt=prompt,
        max_tokens=300,
        temperature=0.7,
    )
    
    return response['choices'][0]['text'].strip()

anime_name = "Kimi no Na wa"
response = answer_anime_question_gpt3(anime_name)
print(response)
