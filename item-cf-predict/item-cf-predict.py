def item_cf_predict(user_ratings, item_similarities, target):
    """
    Predict the rating using item-based collaborative filtering.
    """
    # Write code here
    user, item = map(lambda x: x[:target] + x[target+1:], 
                    (user_ratings, item_similarities))

    num = sum(r * s for r, s in zip(user, item) if s>0)

    den = sum(s for r, s in zip(user, item) if (r!= 0 and s>=0))

    if den == 0:
        return 0
        
    return num/den