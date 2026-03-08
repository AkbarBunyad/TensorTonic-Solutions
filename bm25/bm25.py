import numpy as np
from collections import Counter
import math

def bm25_score(query_tokens, docs, k1=1.2, b=0.75):

    l_query = len(query_tokens)
    l_docs = len(docs)

    score = np.zeros(l_docs)

    for i in range(l_docs):
        eps = 0
        for j in range(l_query):

            df = sum(query_tokens[j] in doc for doc in docs)

            idf = math.log((len(docs) - df + 0.5) /
                           (df + 0.5) + 1)

            tf = Counter(docs[i])[query_tokens[j]]

            score_intermediate = idf * (tf * (k1 + 1)) / (
                tf + k1 * (1 - b + b * len(docs[i]) /
                np.mean([len(doc) for doc in docs]))
            )

            eps += score_intermediate

        score[i] = eps

    return score