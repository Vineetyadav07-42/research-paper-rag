from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)


MODEL_NAME = "google/flan-t5-base"


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME
)


def generate_answer(query, retrieved_chunks):

    context = "\n\n".join(
        result["text"]
        for result in retrieved_chunks
    )

    prompt = f"""
Answer the question using only the context below.

If the answer cannot be found in the context,
say "The answer is not available in the provided paper."

Context:
{context}

Question:
{query}

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=200
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return answer