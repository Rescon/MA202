"""
Loading CoNLL 2000 dataset for Chunking
"""


def format_tag(tag):
    struct = {
        "B-NP\n": "NP",
        "B-PP\n": "PP",
        "O\n": "O",
        "I-NP\n": "NP",
        "I-PP\n": "PP",
        "B-VP\n": "VP",
        "I-VP\n": "VP",
        "B-SBAR\n": "SBAR",
        "B-ADJP\n": "ADJP",
        "I-ADJP\n": "ADJP",
        "B-ADVP\n": "ADVP",
        "I-ADVP\n": "ADVP",
        "B-INTJ\n": "INTJ",
        "I-PRT\n": "PRT",
        "I-SBAR\n": "SBAR",
        "I-UCP\n": "UCP",
        "B-PRT\n": "PRT",
        "B-LST\n": "LST",
        "I-CONJP\n": "CONJP",
        "B-CONJP\n": "CONJP",
        "I-INTJ\n": "INTJ",
        "B-UCP\n": "UCP",
        "I-LST\n": "LST"
    }
    
    return struct.get(tag, tag)


def load_conll2000_data(path = "./"):
    tagged_sentences = []
    
    sent = []
    with open(path, "r") as file:
        for x in file:
            if x != "\n":
                elements = x.split(" ")
                tag = format_tag(elements[2])
                word = elements[0]
                sent.append((tag, word))
                
            else:
                tagged_sentences.append(sent)
                sent = []
    
    return tagged_sentences 

        
def load_conll2000(path = "./"):
    train_set = load_conll2000_data(path + "train.txt")
    test_set = load_conll2000_data(path + "test.txt")
    return train_set, test_set