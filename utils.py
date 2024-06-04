import json

def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}

class NameParser:
    def __init__(self):
        self.glossary = json.load(open("data/glossary.json"))
        self.food = set(self.glossary['food'])
        self.action = set(self.glossary['action'])
        self.flavour = set(self.glossary['flavour'])

    def __call__(self, name):
        word_list = []
        type_list = []
        id_list = []

        s = 0
        while s < len(name):
            new_s = s
            for e in range(len(name), s, -1):
                w = name[s:e]
                if w in self.action:
                    word_list.append(w)
                    type_list.append(0)
                    id_list.append(self.glossary['action'][w])
                    new_s = e
                    break
                elif w in self.flavour:
                    word_list.append(w)
                    type_list.append(1)
                    id_list.append(self.glossary['flavour'][w])
                    new_s = e
                    break
                elif w in self.food:
                    word_list.append(w)
                    type_list.append(2)
                    id_list.append(self.glossary['food'][w])
                    new_s = e
                    break
                else:
                    continue
            
            if new_s == s:
                break
            else:
                s = new_s

        if s == len(name):
            return word_list, type_list, id_list
        else:
            return [], [], []

