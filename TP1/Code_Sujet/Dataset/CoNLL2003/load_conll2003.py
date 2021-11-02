"""
Loading CoNLL 2003 dataset for NER
"""

def format_tag(tag):
    struct = {
        "O\n": "O",
        "B-ORG\n": "ORG",
        "B-MISC\n": "MISC",
        "I-ORG\n": "ORG",
        "I-MISC\n": "MISC",
        "B-PER\n": "PER",
        "I-PER\n": "PER",
        "B-LOC\n": "LOC",
        "I-LOC\n": "LOC"
    }
    
    return struct.get(tag, tag)
    

def load_conll2003(path = "./"):
    train_set = load_conll2003_data(path + "train.txt") + load_conll2003_data(path + "valid.txt")
    test_set = load_conll2003_data(path + "test.txt")
    
    return train_set, test_set

def load_conll2003_data(path = "./"):
    tagged_sentences = []
    
    sent = []
    with open(path) as file:
        for x in file:
            if x != "\n":
                elements = x.split(" ")
                tag =  format_tag(elements[3])
                word = elements[0]
                sent.append((tag, word))
                
            else:
                tagged_sentences.append(sent)
                sent = []
    
    return tagged_sentences
