
OOV_token = "[OOV]"

def create_token_to_id_table(word_lists):
    word_to_ix = {}
    for words in word_lists:
        for word in words:
            if (word not in word_to_ix) and (word != OOV_token):
                word_to_ix[word] = len(word_to_ix)
    
    word_to_ix[OOV_token] = len(word_to_ix)
    
    return word_to_ix