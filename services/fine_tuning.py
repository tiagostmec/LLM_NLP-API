import pandas as pd
import requests
import time
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split

# Função para obter os dados do anime da API
def get_anime_data(anime_id):
    url = f"https://api.jikan.moe/v4/anime/{anime_id}"
    response = requests.get(url)
    time.sleep(0.35)  # Aguardar para evitar limite de requisições
    if response.status_code == 200:
        data = response.json().get('data', {})
        
        # Ajustar os dados para o formato desejado
        # Ensure normalized and cleaned data
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
        # Estrutura o par de prompt/resposta para treinamento
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

# Função para ler o CSV e obter todos os anime_ids
def get_anime_ids_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    return df['anime_id'].tolist()  # Retorna uma lista com todos os anime_id

# Função de treinamento LoRA para GPT-2
def train_lora_gpt2(csv_file, output_dir="./lora_gpt2_trained", num_train_epochs=5, per_device_train_batch_size=8):
    # Carregar o modelo GPT-2 e o tokenizador
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Definir o pad_token como o eos_token
    tokenizer.pad_token = tokenizer.eos_token  # Usa o token de fim de sequência (eos_token) como o token de preenchimento

    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Ler os anime_ids do CSV
    anime_ids = get_anime_ids_from_csv(csv_file)

    # Preencher o train_data com informações de todos os animes
    train_data = []
    max_animes = 500 # Número máximo de animes que você deseja processar

    for i, anime_id in enumerate(anime_ids):
        anime_data = get_anime_data(anime_id)
        if anime_data:
            train_data.extend(anime_data)  # Adiciona os dados de cada anime ao train_data
        
        # Verificar se atingiu o número máximo de animes
        if i + 1 >= max_animes:
            print(f"Atingido o número máximo de {max_animes} animes.")
            break

    # Processar os dados de treinamento
    # Process the dataset into a dictionary compatible with Dataset.from_dict
    def preprocess_data(data):
        prompts = [item['prompt'] for item in data]
        responses = [item['response'] for item in data]
        inputs = [f"{prompt}\n{response}" for prompt, response in zip(prompts, responses)]

        # Tokenize inputs and create attention masks
        encodings = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": encodings["input_ids"],  # Language modeling uses input_ids as labels
        }


# Create the dataset from the processed data
    processed_data = preprocess_data(train_data)
    train_dataset = Dataset.from_dict(processed_data)

    # Configuração do LoRA
    lora_config = LoraConfig(
    r=16,  # Increase rank for better adaptation
    lora_alpha=64,  # Higher scaling factor
    lora_dropout=0.05,  # Lower dropout for stable training
    bias="all"  # Allow biases to adapt
    )


    # Aplicando LoRA ao modelo
    model = get_peft_model(model, lora_config)

    # Definir argumentos de treinamento
    training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="steps",  # Validate periodically
    eval_steps=500,  # Frequency of validation
    save_steps=1000,  # Frequency of saving checkpoints
    save_total_limit=2,  # Keep only the best checkpoints
    per_device_train_batch_size=8,  # Adjust to your hardware
    gradient_accumulation_steps=8,  # Simulate larger batches
    learning_rate=5e-5,  # Adjust learning rate
    num_train_epochs=num_train_epochs,
    weight_decay=0.01,  # Regularization
    logging_dir="./logs",
    logging_steps=100,
)

  # Dividir os dados em treinamento e avaliação
    train_data, eval_data = train_test_split(train_data, test_size=0.1, random_state=42)

    # Processar os dados de avaliação
    processed_eval_data = preprocess_data(eval_data)
    eval_dataset = Dataset.from_dict(processed_eval_data)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # Adicionar o conjunto de avaliação
)

    # Treinar o modelo
    trainer.train()

    # Salvar o modelo treinado
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Modelo treinado e salvo em {output_dir}")


# Exemplo de uso da função de treinamento
csv_file = 'C:\\Users\\th_sm\\Desktop\\LLM_NLP API\\data\\anime.csv'  # Caminho para o seu arquivo CSV
train_lora_gpt2(csv_file)
