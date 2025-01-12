import pandas as pd
import requests
import time
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split

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


def fine_tune(csv_file, model_type, output_dir="./lora_gpt2_trained"):
    
    model_name = model_type
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    tokenizer.pad_token = tokenizer.eos_token 

    model = GPT2LMHeadModel.from_pretrained(model_name)

    anime_ids = get_anime_ids_from_csv(csv_file)

    train_data = []
    max_animes = 500

    for i, anime_id in enumerate(anime_ids):
        anime_data = get_anime_data(anime_id)
        if anime_data:
            train_data.extend(anime_data)
        
        if i + 1 >= max_animes:
            print(f"Atingido o número máximo de {max_animes} animes.")
            break

    def preprocess_data(data):
        prompts = [item['prompt'] for item in data]
        responses = [item['response'] for item in data]
        inputs = [f"{prompt}\n{response}" for prompt, response in zip(prompts, responses)]

        encodings = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": encodings["input_ids"],
        }


    processed_data = preprocess_data(train_data)
    train_dataset = Dataset.from_dict(processed_data)

    lora_config = LoraConfig(
    r=8,  
    lora_alpha=8,  
    lora_dropout=0.1,  
    bias="all"  
    )


    model = get_peft_model(model, lora_config)


    training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="steps", 
    eval_steps=15,  
    save_total_limit=2,  
    per_device_train_batch_size=8,  
    gradient_accumulation_steps=8, 
    learning_rate=5e-5,  
    num_train_epochs=10,
    weight_decay=0.01,  
    logging_dir="./logs",
    logging_steps=10,
)


    train_data, eval_data = train_test_split(train_data, test_size=0.15, random_state=42)


    processed_eval_data = preprocess_data(eval_data)
    eval_dataset = Dataset.from_dict(processed_eval_data)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  
)


    trainer.train()


    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Modelo treinado e salvo em {output_dir}")



# csv_file = 'C:\\Users\\th_sm\\Desktop\\LLM_NLP API\\data\\anime.csv'  
# train_lora_gpt2(csv_file)
