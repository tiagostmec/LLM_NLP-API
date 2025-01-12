import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


model_path = 'C:\\Users\\th_sm\\Desktop\\LLM_NLP API\\lora_gpt2_trained'
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)


tokenizer.pad_token = tokenizer.eos_token


def ask_question(question):
   
    input_text = (
        f"You are an anime expert. Answer the following question based on the database:\n"
        f"Question: {question}\nAnswer:"
    )
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=512,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
        )


    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer.replace(input_text, "").strip()

# anime_name = 'Death Note'
# response = ask_question(f"Tell me the plot of {anime_name}")
# print(f"Final response: {response}")
