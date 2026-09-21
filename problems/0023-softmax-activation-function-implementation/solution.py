import math

def softmax(scores: list[float]) -> list[float]:
    # Your code here
    max_score = max(scores)
    sum=0
    L = []
    for i in range(len(scores)):
        sum+=math.exp(scores[i] - max_score)
    
    for i in range(len(scores)):
        L.append(math.exp(scores[i] - max_score)/sum)

    return L