import requests
import pandas as pd
from datasets import Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Função para obter dados de um anime usando a API Jikan (MyAnimeList)
def get_anime_data(anime_id):
    url = f"https://api.jikan.moe/v4/anime/{anime_id}"  # Atualizando a URL para a nova versão da API
    response = requests.get(url)
    time.sleep(0.35)
    if response.status_code == 200:
        data = response.json().get('data', {})  # Garantir que o conteúdo 'data' existe

        return {
            "anime_id": data.get("mal_id"),
            "name": data.get("title", "N/A"),  # Nome do anime
            "title_english": data.get("title_english", "N/A"),  # Nome em inglês
            "title_japanese": data.get("title_japanese", "N/A"),  # Nome em japonês
            "type": data.get("type", "N/A"),  # Tipo de anime (ex. Movie, TV)
            "episodes": data.get("episodes", "N/A"),  # Número de episódios
            "rating": data.get("rating", "N/A"),  # Classificação
            "members": data.get("members", 0),  # Número de membros
            "synopsis": data.get("synopsis", "Sinopse não disponível."),  # Sinopse
            "genres": ", ".join([genre["name"] for genre in data.get("genres", [])]),  # Gêneros
            "image_url": data.get("images", {}).get("jpg", {}).get("image_url", "N/A"),  # URL da imagem
            "trailer_url": data.get("trailer", {}).get("url", "N/A"),  # URL do trailer, se disponível
            "score": data.get("score", 0),  # Nota média
            "popularity": data.get("popularity", 0),  # Popularidade
            "favorites": data.get("favorites", 0)  # Número de favoritos
        }
    else:
        print(f"Erro ao obter dados para anime_id {anime_id}: {response.status_code}")
        return None

# Função para criar entrada textual para treinamento
def create_input_text(row):
    # Remover quebras de linha extras e formatar o texto para ter uma aparência mais limpa
    input_text = f"""
    Anime: {row['name']}.
    Nome em inglês: {row['title_english']}.
    Nome em japonês: {row['title_japanese']}.
    Tipo: {row['type']}.
    Episódios: {row['episodes']}.
    Avaliação: {row['rating']}.
    Membros: {row['members']}.
    Sinopse: {row['synopsis']}.
    Gêneros: {row['genres']}.
    URL da imagem: {row['image_url']}.
    URL do trailer: {row['trailer_url']}.
    Nota média: {row['score']}.
    Popularidade: {row['popularity']}.
    Número de favoritos: {row['favorites']}.
    """
    
    # Remover quebras de linha (\n) indesejadas
    input_text = input_text.replace("\n", " ").strip()
    
    return input_text

async def fine_tuning(model_type: str, csv_file: str, output_dir: str):
    # Carregar o CSV com informações dos animes
    df = pd.read_csv(csv_file)

    # Lista para armazenar os dados dos animes
    anime_data = []

    # Obter dados de animes pela API com base no anime_id do CSV
    i = 0
    total_animes = len(df['anime_id'])  # Total de animes no DataFrame
    for anime_id in df['anime_id']:
        i += 1
        data = get_anime_data(anime_id)
        if data:
            anime_data.append(data)
        
        # Exibe a porcentagem de progresso a cada iteração
        if i % 100 == 0:  # Exibe a porcentagem a cada 100 iterações (ajuste conforme necessário)
            percentage = (i / total_animes) * 100
            print(f"Progresso: {percentage:.2f}% ({i}/{total_animes})")
            
        # Caso queira limitar o número de animes, adicione a condição de parada
        # if i == 150:
        #     break

    # Criando o dataset de treinamento
    dataset = pd.DataFrame(anime_data)
    dataset['input_text'] = dataset.apply(create_input_text, axis=1)

    # Inicializando o Tokenizer e Modelo GPT-2
    tokenizer = GPT2Tokenizer.from_pretrained(model_type)
    tokenizer.pad_token = tokenizer.eos_token  # Definindo o token de padding como eos
    model = GPT2LMHeadModel.from_pretrained(model_type).to(device)  # Carregando o modelo para o dispositivo
    # Verificar as primeiras linhas do dataset
    print(dataset.head())  # Exibe as primeiras 5 linhas do DataFrame

    # Tokenizando os dados
    def tokenize_function(examples):
        encodings = tokenizer(
            examples['input_text'], 
            padding="max_length",  # Garantir que o texto seja padronizado para o comprimento máximo
            truncation=True, 
            max_length=512,  # Garantir que o texto não ultrapasse 512 tokens
            return_tensors='pt'
        )
        encodings['labels'] = encodings.input_ids.detach().clone()

        # Adicionar a máscara de atenção explicitamente
        attention_mask = (encodings['input_ids'] != tokenizer.pad_token_id).long()
        encodings['attention_mask'] = attention_mask
        
        return encodings


    train_dataset = Dataset.from_pandas(dataset[['input_text']])
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    # Exemplo para verificar os dados tokenizados
    print(train_dataset[0])  # Verifique o primeiro exemplo após a tokenização

    # Definindo os argumentos de treinamento
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="no",
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        num_train_epochs=4,
        weight_decay=0.01,
        save_total_limit=1,
        logging_dir='./logs', 
        logging_steps=25,
    )

    # Inicializando o Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # Treinando o modelo
    trainer.train()

    # Salvar o modelo fine-tuned
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Função para gerar uma resposta do modelo
    def generate_anime_info(anime_name):
        input_text = f"Me fale sobre {anime_name}."
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)  # Passar os inputs para o dispositivo
        # Adicionar a máscara de atenção
        attention_mask = (inputs['input_ids'] != tokenizer.pad_token_id).long().to(device)
        outputs = model.generate(inputs['input_ids'], attention_mask=attention_mask, max_length=200, num_return_sequences=1, no_repeat_ngram_size=2)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response


    # Exemplo de uso
    anime_name = "Kimi no Na wa"
    response = generate_anime_info(anime_name)
    print(response)
