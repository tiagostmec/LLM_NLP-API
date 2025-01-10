import torch
from fastapi import HTTPException
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Carregar o modelo e o tokenizer uma única vez para evitar carregamentos repetidos
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = 'C:\\Users\\th_sm\\Desktop\\LLM_NLP API\\data'
model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# Definir o pad token
tokenizer.pad_token = tokenizer.eos_token

# Função para realizar a inferência
def query_ft(msg: str):
    try:
        # Preparar os inputs e enviar para o dispositivo correto
        inputs = tokenizer(msg, return_tensors="pt", padding=True, truncation=True).to(device)
        
        # Adicionar a máscara de atenção
        attention_mask = (inputs['input_ids'] != tokenizer.pad_token_id).long()

        # Gerar a resposta
        outputs = model.generate(inputs['input_ids'], attention_mask=attention_mask, max_length=512, num_return_sequences=1, no_repeat_ngram_size=2)

        # Decodificar a sequência gerada em texto legível
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(generated_text)
        return {"generated_text": generated_text}

    except Exception as e:
        # Retorna um erro HTTP caso ocorra algum problema
        raise HTTPException(status_code=500, detail=str(e))
