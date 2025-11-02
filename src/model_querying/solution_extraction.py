import math

def get_yes_no(text):
    text = text.lower()
    y = "true" in text
    n = "false" in text
    if y == n:
        return None
    return y

def get_yes_no_diff_logprobs(logprobs_dict):
    eps = 1e-5
    prob_sums = {False: eps, True: eps}
    for token, logprob in logprobs_dict.items():
        o = get_yes_no(token)
        if o is None:
            continue
        prob_sums[o] += math.exp(logprob)
    
    if prob_sums[False] == eps and prob_sums[True] == eps:
        return 0
    else:
        return math.log(prob_sums[True]) - math.log(prob_sums[False])

