def main():
    
    from utils import jpath, ls
    from datasets import load_dataset
    from tqdm import tqdm

    # all: 362251   train: 361526   test: 725
    # Load the dataset
    data_root = '/data2/longshen/Datasets/LAMD_v4/LAMD/REMI_ORGANIZED'
    all_remi_fps = []
    sub_dirs = ls(data_root)
    for sub_dir in tqdm(sub_dirs):
        sub_fp = jpath(data_root, sub_dir)
        remi_fns = ls(sub_fp)
        all_remi_fps.extend([jpath(sub_fp, fn) for fn in remi_fns])
    dataset = load_dataset("text", data_files={"train": all_remi_fps})

    # Split the dataset
    import datasets
    dataset_splitted = dataset['train'].train_test_split(test_size=0.002)


    # Tokenize the dataset
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("/home/longshen/work/MuseCoco/musecoco/dataset_preparation/test_tokenizer2")
    context_length = 2048 #2048
    outputs = tokenizer(
        dataset["train"][:2]["text"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )

    def tokenize(element):
        outputs = tokenizer(
            element["text"],
            truncation=True,
            max_length=context_length,
            padding="max_length",
            return_overflowing_tokens=True,
            return_length=True,
        )
        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            if length == context_length:
                input_batch.append(input_ids)
        return {"input_ids": input_batch}

    tokenized_datasets = dataset_splitted.map(
        tokenize, batched=True, remove_columns=dataset_splitted["train"].column_names
    )

    from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig, AutoModelForCausalLM
    import torch

    config = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=len(tokenizer),
        n_ctx=context_length,
        n_positions=context_length,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        n_embd=768,
        n_head=16,
        n_layer=12, #24
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    # model = GPT2LMHeadModel(config).half()
    model = AutoModelForCausalLM.from_config(config)
    model.save_pretrained("test_model")
    model = AutoModelForCausalLM.from_pretrained("test_model", torch_dtype=torch.bfloat16)
    # model = GPT2LMHeadModel(config)
    model_size = sum(t.numel() for t in model.parameters())
    print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")

    from transformers import DataCollatorForLanguageModeling

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    a = 2

    from transformers import Trainer, TrainingArguments

    args = TrainingArguments(
        output_dir="m2m_pt",
        per_device_train_batch_size=3,
        per_device_eval_batch_size=3,
        evaluation_strategy="steps",
        eval_steps=2_000, # about 1/16 epoch
        logging_steps=5,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        weight_decay=0.1,
        warmup_steps=1_000,
        lr_scheduler_type="cosine",
        learning_rate=5e-4,
        save_steps=2_000,
        push_to_hub=True,
        bf16=True,
        # fp16=True,
        seed=42,
    )


    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
    )

    trainer.train()

    trainer.push_to_hub()


if __name__ == '__main__':
    main()