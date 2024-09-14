from transformers import BartTokenizer, BartForConditionalGeneration

tokenizer = BartTokenizer.from_pretrained('final-fine-tuned-model')
model = BartForConditionalGeneration.from_pretrained('final-fine-tuned-model')


def generate_response(input_text):
    inputs = tokenizer(input_text, return_tensors='pt')

    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        num_beams=4,
        early_stopping=True,
        max_new_tokens=100,
        no_repeat_ngram_size=3,
        forced_bos_token_id=0,
        forced_eos_token_id=2
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


sample_query = input("Enter your query: ")

output = generate_response(sample_query)

print("Output:", output)
