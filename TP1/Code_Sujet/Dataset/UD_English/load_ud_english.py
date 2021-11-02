
def load_ud_english_data(path = "./"):
    
    with open(path) as file:
        
        sentences = []
        
        sent = []
        for line in file:
            
            if line == "\n":
                sentences.append(sent)
                sent = []
                
            else:
                elements = line.split("\t")
                word = elements[1]
                tag = elements[3]
                
                sent.append((tag, word))
                
        return sentences
            
    
def load_ud_english(path = "./"):
    train_set = load_ud_english_data(path + "en-ud-train.conllu") + load_ud_english_data(path + "en-ud-dev.conllu")
    test_set = load_ud_english_data(path + "en-ud-test.conllu")
    
    return train_set, test_set
            