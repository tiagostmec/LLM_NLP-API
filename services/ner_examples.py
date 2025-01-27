import pandas as pd
import requests
import spacy
from transformers import BertTokenizer, BertForTokenClassification, pipeline
import torch
import warnings
from sklearn.metrics import precision_recall_fscore_support
import time

# Ignorar todos os warnings de bibliotecas externas
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


class NERProcessor:
    def __init__(self, model_name="dbmdz/bert-large-cased-finetuned-conll03-english", spacy_model="en_core_web_sm"):
        # Carregar o modelo spaCy e BERT
        self.nlp_spacy = spacy.load(spacy_model)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForTokenClassification.from_pretrained(model_name)
        self.device = 0 if torch.cuda.is_available() else -1  # Se GPU disponível, usar CUDA
        self.model = self.model.to(self.device)
        self.nlp_ner = pipeline("ner", model=self.model, tokenizer=self.tokenizer, device=self.device, aggregation_strategy="simple")

    def preprocess_text(self, text):
        """Função de pré-processamento do texto"""
        # Converter para minúsculas, remover URLs, emojis, hashtags, menções e caracteres especiais
        text = text.lower()  # Normalizar para minúsculas
        text = text.replace("\n", " ").replace("\r", " ")  # Remover quebras de linha
        text = ''.join(e for e in text if e.isalnum() or e.isspace())  # Remover caracteres especiais
        return text

    def ner_spacy(self, texts):
        """Aplicar NER usando spaCy"""
        docs = self.nlp_spacy.pipe(texts, batch_size=50)
        entities = []
        for doc in docs:
            entities.append([{"entity": ent.text, "label": ent.label_} for ent in doc.ents])
        return entities

    def ner_bert(self, texts):
        """Aplicar NER usando BERT"""
        return self.nlp_ner(texts)

    def evaluate_ner(self, y_true, y_pred):
        """Avaliar a performance do modelo com precisão, recall e F1-score"""
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="micro")
        return precision, recall, f1


class AnimeNER:
    def __init__(self, csv_path, max_animes=500):
        self.csv_path = csv_path
        self.max_animes = max_animes
        self.ner_processor = NERProcessor()  # Inicializar o processador de NER
    
    def get_anime_data(self, anime_id):
        """Função para fazer a requisição para a API do Jikan"""
        url = f"https://api.jikan.moe/v4/anime/{anime_id}"
        time.sleep(0.35)
        try:
            response = requests.get(url)
            response.raise_for_status()  # Levanta uma exceção para respostas de erro HTTP
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Erro ao acessar a API para anime_id {anime_id}: {e}")
            return None  # Retorna None em caso de erro de requisição
    
    def process_animes(self):
        """Função principal para processar os animes e fazer NER"""
        df = pd.read_csv(self.csv_path)
        df_sample = df.head(self.max_animes)

        results_spacy = []
        results_bert = []
        texts_spacy = []
        texts_bert = []
        
        # Loop sobre os IDs de anime no CSV
        for anime_id in df_sample['anime_id']:
            anime_data = self.get_anime_data(anime_id)
            if anime_data is None:  # Caso não tenha conseguido recuperar os dados
                continue  # Pular para o próximo anime

            try:
                # Extrair apenas a sinopse
                synopsis = anime_data['data']['synopsis']

                # Pré-processar apenas a sinopse
                preprocessed_synopsis = self.ner_processor.preprocess_text(synopsis)  # Pré-processamento

                # Adicionar a sinopse pré-processada ao lote para spaCy e BERT
                texts_spacy.append(preprocessed_synopsis)
                texts_bert.append(preprocessed_synopsis)

                # Armazenar apenas a sinopse para o CSV
                results_spacy.append({
                    'anime_id': anime_id,
                    'synopsis': synopsis,
                })
                results_bert.append({
                    'anime_id': anime_id,
                    'synopsis': synopsis,
                })
            except KeyError:
                print(f"Erro ao processar o anime_id {anime_id}, dados incompletos.")
        
        # Processar os textos em lote
        spacy_entities = self.ner_processor.ner_spacy(texts_spacy)
        bert_entities = self.ner_processor.ner_bert(texts_bert)

        # Adicionar as entidades nos resultados
        for idx in range(len(results_spacy)):
            results_spacy[idx]['ner_spacy'] = spacy_entities[idx]
            results_bert[idx]['ner_bert'] = bert_entities[idx]

        # Retornar os resultados
        return pd.DataFrame(results_spacy), pd.DataFrame(results_bert)


# Caminho para o arquivo CSV com os IDs dos animes
csv_path = 'C:\\Users\\th_sm\\Desktop\\LLM_NLP API\\data\\anime.csv'  # Substitua pelo caminho correto

# Processar os animes e obter os resultados
anime_ner = AnimeNER(csv_path, max_animes=50)
df_spacy, df_bert = anime_ner.process_animes()

# Exibir resultados
print(df_spacy.head())

# Salvar os resultados em arquivos CSV
df_spacy.to_csv('C:\\Users\\th_sm\\Desktop\\LLM_NLP API\\data\\resultados_spacy.csv', index=False)
df_bert.to_csv('C:\\Users\\th_sm\\Desktop\\LLM_NLP API\\data\\resultados_bert.csv', index=False)

print("Resultados salvos em 'resultados_spacy.csv' e 'resultados_bert.csv'.")

# Avaliação do modelo (exemplo com rótulos fictícios)
y_true = ['anime', 'synopsis', 'genres']  # Exemplo de rótulos reais
y_pred_spacy = ['anime', 'synopsis', 'genres']  # Exemplo de predições do modelo
y_pred_bert = ['anime', 'synopsis', 'genres']

precision_spacy, recall_spacy, f1_spacy = anime_ner.ner_processor.evaluate_ner(y_true, y_pred_spacy)
precision_bert, recall_bert, f1_bert = anime_ner.ner_processor.evaluate_ner(y_true, y_pred_bert)

print(f"Evaluating spaCy NER - Precision: {precision_spacy}, Recall: {recall_spacy}, F1: {f1_spacy}")
print(f"Evaluating BERT NER - Precision: {precision_bert}, Recall: {recall_bert}, F1: {f1_bert}")
