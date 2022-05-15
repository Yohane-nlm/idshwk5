from sklearn.ensemble import RandomForestClassifier
import numpy as np

domainlist = []
class Domain:
    def __init__(self, _name, _label, _length, _num, _entropy):
        self.name = _name
        self.label = _label
        self.num = _num
        self.length = _length
        self.entropy = _entropy

    
    def returnData(self):
        return [self.length, self.num, self.entropy]
    
    def returnLabel(self):
        if self.label == "dga":
            return 1
        else:
            return 0

def countNum(str):
    count = 0
    for c in str:
        if(c.isdigit()):
            count += 1
    return count

def calEntropy(str):
    h = 0.0
    sum = 0
    letter = [0] * (26)
    str = str.lower()
    for i in range(len(str)):
        if str[i].isalpha():
            letter[ord(str[i]) - ord('a')] += 1
            sum += 1
    for i in range(26):
        p = 1.0 * letter[i] / sum
        if p > 0:
            h += -p * np.log2(p)
    return h



def initData(filename, domainList):
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line == "":
                continue
            tokens = line.split(",")
            name = tokens[0]
            label = tokens[1] if len(tokens) > 1 else "notdga"
            num = countNum(name)
            length = len(name)
            entropy = calEntropy(name)
            domainList.append(Domain(name, label, length, num, entropy))

def main():
    domainList = []
    initData("train.txt", domainList)
    featureMatrix = []
    labelList = []
    for domain in domainList:
        featureMatrix.append(domain.returnData())
        labelList.append(domain.returnLabel())
    
    clf = RandomForestClassifier(random_state = 0)
    clf.fit(featureMatrix, labelList)

    domainList4test = []
    initData("test.txt", domainList4test)
    with open("result.txt","w") as f:
        for domain in domainList4test:
            feature = domain.returnData()
            label = clf.predict([feature])[0]
            if label == 1:
                f.write(domain.name + "," + "dga" + "\n")
            else:
                f.write(domain.name + "," + "notdga" + "\n")

if __name__ == "__main__":
    main()




