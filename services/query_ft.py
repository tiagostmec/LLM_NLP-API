import torch
from fastapi import HTTPException
from transformers import GPT2LMHeadModel, GPT2Tokenizer



# Função para realizar a inferência
def query_ft(msg: str):
    # Carregar o modelo e o tokenizer uma única vez para evitar carregamentos repetidos
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = 'C:\\Users\\th_sm\\Desktop\\LLM_NLP API\\data\\checkpoint-9'
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)

    # Definir o pad token
    tokenizer.pad_token = tokenizer.eos_token
    try:
        # Tokenizar o input do usuário
        inputs = tokenizer(msg, return_tensors="pt", padding=True, truncation=True, max_length=512)

        # Gerar a máscara de atenção
        attention_mask = inputs["attention_mask"]

        # Gerar texto a partir do modelo
        outputs = model.generate(
            inputs["input_ids"],  # Entrada tokenizada
            attention_mask=attention_mask,  # Máscara de atenção
            max_length=2000,  # Tamanho máximo da sequência gerada
            num_return_sequences=1,  # Número de sequências geradas
            do_sample=True,  # Habilitar amostragem
            temperature=1.2,  # Controla a aleatoriedade
            top_p=0.95,  # Amostragem de núcleo (nucleus sampling)
            top_k=25,  # Amostragem de top-k
            pad_token_id=tokenizer.pad_token_id  # Usar pad token como eos token
        )

        # Decodificar a sequência gerada em texto legível
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {"generated_text": generated_text}

    except Exception as e:
        # Retorna um erro HTTP caso ocorra algum problema
        raise HTTPException(status_code=500, detail=str(e))