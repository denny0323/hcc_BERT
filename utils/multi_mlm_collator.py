import random
import numpy as np

'''
    실제 data_collator함수라기보다는 tokenized_fn에 가까움
    dataset에 mapping하여 사용함  
'''
def MultiTokenMaskingDataCollator(examples, tokenizer,
                                  min_num_tokens=3,
                                  SHORT_SEQ_PROB=0.1, MAX_LENGTH=256):

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

        if curr_len > max_num_tokens:
            curr_len = max_num_tokens
            seq = seq[:max_num_tokens]

        if curr_len > min_num_tokens:
            '''
            (참고)  Binomial(n, p, size)
                n : 각 sample당 가질 수 있는 최대 정수
                p : probability of success
                size : sample 크기
                
                즉, Binomial(1, 0.15, 100)
                = 샘플당 1일 확률이 0.15인 sample 100개를 draw
                (output) = 1값을 갖는 index가 masking확보 
            '''
            # 문장의 시작 Token(=CLS)와 끝 또는 문장 구분 Token(=SEP)는 마스킹하지 않음
            mask = np.random.binomial(1, p=MLM_PROB, size=(curr_len,))

            # multiple masking을 위해 masking된 index가 2개 이상일때까지 draw
            while len(np.where(mask)[0]) < 2:
                mask = np.random.binomial(1, p=MLM_PROB, size=(curr_len,))

            curr_labels = [-100] * max_num_tokens
            for mask_idx in np.where(mask)[0]:
                # 마스캉 본래의 위치를 저장
                # examples["tokenized_sentences"] 앞에 CLS가 추라고 붙기 때문에
                # examples["labels"]와는 1 index차이가 남
                curr_labels[mask_idx] = seq[mask_idx]
                seq[mask_idx] = tokenizer.mask_token_id

            # Define the outputs
            input_ids = tokenizer.build_inputs_with_special_tokens(seq)
            curr_labels.append(-100)
            curr_labels += [-100]

            token_type_ids = tokenizer.create_token_type_ids_from_sequences(seq)
            padded = tokenizer.pad({
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
            }, padding="max_length", truncation=True)

            examples["input_ids"].append(padded["input_ids"])
            examples["token_type_ids"].append(padded["token_type_ids"])
            examples["attention_mask"].append(padded["attention_mask"])
            examples["labels"].append(curr_labels)


    del examples["tokenizerd_sequences"]
    del examples["evnt"]
    del examples["__index_level_0__"]

    return examples


