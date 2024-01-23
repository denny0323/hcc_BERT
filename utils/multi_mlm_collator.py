import random


def MultiTokenMaskingDataCollator(examples):

    # Define the outputs
    examples["input_ids"] = []
    examples["token_type_ids"] = []
    examples["attention_mask"] = []
    examples["labels"] = []

    examples["tokenized_sequences"] = [
        tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize(seq, max_length=MAX_LENGTH, truncation=True)
        ) for seq in examples["evnt"]
    ]

    min_num_tokens = min_num_tokens - tokenizer.num_special_tokens_to_add(pair=False) ## [CLS], [SEP]
    max_num_tokens = MAX_LENGTH - tokenizer.num_special_tokens_to_add(pair=False)

    if random.random() < SHORT_SEQ_PROB:
        target_seq_length = random.randint(2, min_num_tokens)

    for seq_idx, seq in enumerate(examples["tokenizerd_sequences"]):
        curr_len = len(seq)
