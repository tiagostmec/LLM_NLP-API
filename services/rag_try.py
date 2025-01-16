import requests
import pandas as pd
import unicodedata
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPTNeoForCausalLM, GPT2Tokenizer as GPT2TokenizerNeo, \
    GPTNeoForCausalLM as GPTNeo, BartForConditionalGeneration, BartTokenizer, T5ForConditionalGeneration, T5Tokenizer

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


def generate_anime_response(anime_info):
    model_choice = "bart"  # Altere entre "gpt2", "gpt_neo", "gpt_j", "bart", "t5"
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

    prompt = f"""
    Write a brief and engaging description of this anime, focusing on the main plot and characters.
    Anime: {anime_info['title']}
    Genre(s): {', '.join(anime_info['genres'])}
    Score: {anime_info['score']}
    Synopsis: {anime_info['synopsis']}
    Episodes: {anime_info['episodes']}

    """

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    if model_choice in ["gpt2", "gpt_neo", "gpt_j"]:
        outputs = model.generate(
            inputs['input_ids'], 
            max_length=512,  
            num_return_sequences=1, 
            do_sample=True, 
            no_repeat_ngram_size=2, 
            top_p=0.9, 
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=inputs.get('attention_mask')
        )
    elif model_choice in ["bart", "t5"]:
        outputs = model.generate(
            inputs['input_ids'], 
            max_length=256,  
            num_return_sequences=1, 
            do_sample=True,  
            no_repeat_ngram_size=2, 
            top_p=1.2, 
            temperature=0.9,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=inputs.get('attention_mask')
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if model_choice in ["bart", "t5"]:
            response = response.replace(prompt, "").strip()
    if model_choice in ["gpt2", "gpt_neo"]:
            response = response[len(prompt):].strip()

    return response

async def answer_anime_question_rag(anime_name):
    anime_id = get_anime_id_from_csv(anime_name)
    
    if anime_id is None:
        return f"Sorry, I couldn't find the anime '{anime_name}' in the database."
    
    anime_info = get_anime_info(anime_id)
    
    if anime_info:
        response = generate_anime_response(anime_info)
        return response
    else:
        try:
            raise ValueError("Sorry, I couldn't find information about this anime. Error Code: 9999")
        except ValueError as e:
            return e


