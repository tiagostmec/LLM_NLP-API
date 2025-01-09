import requests
import pandas as pd
from datasets import Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Função para obter dados de um anime usando a API Jikan (MyAnimeList)
def get_anime_data(anime_id):
    url = f"https://api.jikan.moe/v4/anime/{anime_id}"  # Atualizando a URL para a nova versão da API
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()['data']  # Ajustando para pegar o conteúdo 'data' do JSON

        return {
            "anime_id": data["mal_id"],
            "name": data["title"],  # Nome do anime
            "title_english": data["title_english"],  # Nome em inglês
            "title_japanese": data["title_japanese"],  # Nome em japonês
            "type": data["type"],  # Tipo de anime (ex. Movie, TV)
            "episodes": data["episodes"],  # Número de episódios
            "rating": data["rating"],  # Classificação
            "members": data["members"],  # Número de membros
            "synopsis": data["synopsis"],  # Sinopse
            "genres": ", ".join([genre["name"] for genre in data["genres"]]),  # Gêneros
            "image_url": data["images"]["jpg"]["image_url"],  # URL da imagem
            "trailer_url": data["trailer"]["url"] if "trailer" in data else None,  # URL do trailer, se disponível
            "score": data["score"],  # Nota média
            "popularity": data["popularity"],  # Popularidade
            "favorites": data["favorites"]  # Número de favoritos
        }
    else:
        return None

# Função assíncrona para fine-tuning
async def fine_tuning(model_type: str, csv_file: str, output_dir: str):
    # Carregar o CSV com informações dos animes
    df = pd.read_csv(csv_file)

    # Lista para armazenar os dados dos animes
    anime_data = []

    # Obter dados de animes pela API com base no anime_id do CSV
    i = 0
    for anime_id in df['anime_id']:
        i = i+1
        data = get_anime_data(anime_id)
        if data:
            anime_data.append(data)
        if i == 10:
            break
        

    # Criar DataFrame com os dados dos animes
    df_animes = pd.DataFrame(anime_data)

    # Preparar o dataset para o fine-tuning
    def create_input_text(row):
        return f"""
        Anime: {row['name']}.\n
        Nome em inglês: {row['title_english']}.\n
        Nome em japonês: {row['title_japanese']}.\n
        Tipo: {row['type']}.\n
        Episódios: {row['episodes']}.\n
        Avaliação: {row['rating']}.\n
        Membros: {row['members']}.\n
        Sinopse: {row['synopsis']}.\n
        Gêneros: {row['genres']}.\n
        URL da imagem: {row['image_url']}.\n
        URL do trailer: {row['trailer_url'] if row['trailer_url'] else 'Não disponível'}.\n
        Nota média: {row['score']}.\n
        Popularidade: {row['popularity']}.\n
        Número de favoritos: {row['favorites']}.
        """

    # Criar texto de entrada com base nas informações do anime
    df_animes['input_text'] = df_animes.apply(create_input_text, axis=1)

    # Converter o DataFrame para um Dataset do Hugging Face
    dataset = Dataset.from_pandas(df_animes[['input_text']])

    # Carregar o modelo e o tokenizer do GPT-2
    model_name = model_type
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Garantir que o pad token seja o mesmo que o eos token

    # Mover o modelo para o dispositivo (GPU ou CPU)
    model.to(device)

    # Tokenizar os dados
    def tokenize_function(examples):
        encodings = tokenizer(
            examples['input_text'], 
            padding="max_length",  # Garantir que o texto seja padronizado para o comprimento máximo
            truncation=True, 
            max_length=512,  # Garantir que o texto não ultrapasse 512 tokens
            return_tensors='pt'
        )
        encodings['labels'] = encodings.input_ids.detach().clone()
        return encodings

    # Aplica a tokenização ao dataset
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Definir argumentos de treinamento
    training_args = TrainingArguments(
        output_dir=output_dir,  
        evaluation_strategy="no",  
        learning_rate=5e-5,  
        per_device_train_batch_size=4,  
        num_train_epochs=3,  
        weight_decay=0.01,  
        logging_dir='./logs',  
        save_steps=500,  
        save_total_limit=2,
        no_cuda=False if torch.cuda.is_available() else True
    )

    # Criar o Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        tokenizer=tokenizer,
    )

    # Treinar o modelo
    trainer.train()

    # Salvar o modelo fine-tuned
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
