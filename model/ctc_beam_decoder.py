import numpy as np

def logsumexp(a, b):
    if a is None and b is None:
        return None
    elif a is None:
        return b
    elif b is None:
        return a
    mx = max(a, b)
    mn = min(a, b)
    return np.log(1.0 + np.exp(mn-mx)) + mx

class GammaEntry(object):
    def __init__(self):
        self.blank = None
        self.other = None

class BeamEntry(object):
    def __init__(self):
        self.path = None
        self.gamma = None
        self.P_path_full = None
        self.P_path_partial = None

def prefix_serach_decoding(logprob, T, threshold):
    ranges = prefix_beam_search_split(logprob, T, threshold)
    path = []
    for s, e in ranges:
        curr_logprob = logprob[s:e+1]
        curr_T = e - s + 1
        curr_path = prefix_beam_search(curr_logprob, curr_T)
        path.extend(curr_path)
    return path

def prefix_beam_search_split(logprob, T, threshold):
    start = 0
    ranges = []
    for t in range(1, T):
        if logprob[t, 0] >= threshold:
            ranges.append((start, t))
            start = t+1
    if start < T:
        ranges.append((start, T-1))
    return ranges

#page 64 Prefix Search Decoding Algorithm
#Supervised Sequence Labelling with Recurrent Neural Networks (https://www.cs.toronto.edu/~graves/preprint.pdf
def prefix_beam_search(logprob, T):
    ZERO_LOG_PROB=None
    num_tags = logprob.shape[0] - 1
    beam = []
    gamma = []
    gamma_entry = GammaEntry()
    gamma_entry.blank = logprob[0, 0]
    gamma_entry.other = ZERO_LOG_PROB
    gamma.append(gamma_entry)

    for t in range(1, T):
        gamma_entry = GammaEntry()
        gamma_entry.blank = logsumexp(logprob[t, 0], gamma[-1].blank)
        gamma_entry.other = ZERO_LOG_PROB
        gamma.append(gamma_entry)

    beam_entry = BeamEntry()
    beam_entry.gamma = gamma
    beam_entry.path = []
    beam_entry.P_path_full = np.exp(gamma[T - 1].blank)
    beam_entry.P_path_partial = 1 - beam_entry.P_path_full
    beam.append(beam_entry)

    l_star, P_l_star = beam_entry.path, beam_entry.P_path_full
    while len(beam) > 0:
        beam.sort(key=lambda x: x.P_path_partial)
        p_star = beam[-1]
        beam.pop()

        prob_remaining = p_star.P_path_partial
        if prob_remaining <= P_l_star:  # done
            break

        for k in range(1, num_tags + 1):
            path = p_star.path + [k]

            gamma = []
            gamma_entry = GammaEntry()
            gamma_entry.other = logprob[0, k] if p_star.path == [] else ZERO_LOG_PROB
            gamma_entry.blank = ZERO_LOG_PROB
            gamma.append(gamma_entry)

            prefix_prob = gamma_entry.other
            for t in range(1, T):
                new_label_prob = p_star.gamma[t - 1].blank
                if len(p_star.path) == 0 or p_star.path[-1] != k:
                    new_label_prob = logsumexp(new_label_prob, p_star.gamma[t - 1].other)
                gamma_entry = GammaEntry()
                gamma_entry.other = logprob[t, k] + logsumexp(new_label_prob, gamma[t - 1].other)
                gamma_entry.blank = logprob[t, 0] + logsumexp(gamma[t - 1].blank, gamma[t - 1].other)
                gamma.append(gamma_entry)
                prefix_prob = logsumexp(prefix_prob, (logprob[t, k] + new_label_prob))

            P_path_full = logsumexp(gamma[T - 1].blank, gamma[T - 1].other)
            P_path_partial = prefix_prob - P_path_full

            prob_remaining = prob_remaining - P_path_partial
            if P_path_full > P_l_star:
                l_star = path
                P_l_star = P_path_full

            if P_path_partial > P_l_star:
                beam_entry = BeamEntry()
                beam_entry.path = path
                beam_entry.P_path_full = P_path_full
                beam_entry.P_path_partial = P_path_partial
                beam.append(beam_entry)

            if prob_remaining <= P_l_star:
                break
    return l_star

if __name__ == '__main__':
    prob = np.array([[2, 0, 0, 0, 0], [1, 3, 0, 0, 0 ], [ 1, 4, 1, 0, 0 ], [ 1, 1, 5, 6, 0 ],
    [ 1, 1, 1, 1, 1 ], [ 1, 1, 7, 1, 1 ], [ 9, 1, 1, 1, 1 ]])
    from scipy.special import softmax
    logprob = np.log(softmax(prob, axis = 1))
    alphabet = ['B','a', 'b', 'c', 'd' ]
    print([alphabet[p] for p in prefix_beam_search(logprob, 7)])
    #dacba
    #max path
    print([alphabet[k] for k in np.argmax(logprob, axis=1) if k != 0])
