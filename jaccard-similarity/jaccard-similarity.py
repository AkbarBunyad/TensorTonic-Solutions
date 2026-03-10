def jaccard_similarity(set_a, set_b):
    """
    Compute the Jaccard similarity between two item sets.
    """
    # Write code here
    total = set(set_a + set_b)
    intersect = set(set_a).intersection(set_b)
    if len(total) == 0:
        return 0
    return len(intersect) / len(total)