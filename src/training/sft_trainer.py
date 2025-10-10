from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
import os

def load_data(data_path):
    # Assumes sft.jsonl is in data/ and formatted as {"prompt": ..., "response": ...}
    dataset = load_dataset("json", data_files=data_path, split="train")
    dataset = dataset.map(lambda x: {"text": x["prompt"] + x["response"]})
    return dataset

def tokenize_data(dataset, tokenizer):
    return dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length"), batched=True)

def main():
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    data_path = os.path.join("data", "sft.jsonl")
    dataset = load_data(data_path)
    tokenized_dataset = tokenize_data(dataset, tokenizer)

    training_args = TrainingArguments(
        output_dir="./checkpoints/sft_gpt2",
        per_device_train_batch_size=4,
        num_train_epochs=3,
        logging_dir="./logs",
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        evaluation_strategy="no",
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

if __name__ == "__main__":
    main()