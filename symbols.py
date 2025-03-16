def load_dict(path):
    f = open(path, 'r')
    return f


pinyin = load_dict('data/pinyin.txt')
pinyin2id = {j.strip(): i for i, j in enumerate(pinyin)}
id2pinyin = {j: i for i, j in pinyin2id.items()}