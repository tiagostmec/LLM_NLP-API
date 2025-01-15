import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split
from services.utils import get_anime_data, get_anime_ids_from_csv


def fine_tuning(csv_file, model_type, output_dir="./lora_gpt2_trained"):
    
    model_name = model_type
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    tokenizer.pad_token = tokenizer.eos_token 

    model = GPT2LMHeadModel.from_pretrained(model_name)

    anime_ids = get_anime_ids_from_csv(csv_file)

    train_data = []
    max_animes = 12000

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
