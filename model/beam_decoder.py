import numpy as np

def exp_sum(a, b):
    return np.log(np.exp(a) + np.exp(b))
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

def prefix_beam_search(logprob, T, num_tags):
    beam = []
    gamma = []
    gamma_entry = GammaEntry()
    gamma_entry.blank = logprob[0, 0]
    gamma_entry.other = 0
    gamma.append(gamma_entry)

    for t in range(1, T):
        gamma_entry = GammaEntry()
        gamma_entry.blank = logprob[t, 0] + gamma[-1].blank
        gamma_entry.other = 0
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
            gamma_entry.other = logprob[0, k] if p_star.path == [] else 0
            gamma_entry.blank = 0
            gamma.append(gamma_entry)

            prefix_prob = gamma_entry.other
            for t in range(1, T):
                new_label_prob = p_star.gamma[t - 1].blank
                if len(p_star.path) == 0 or p_star.path[-1] != k:
                    new_label_prob = exp_sum(new_label_prob, p_star.gamma[t - 1].other)
                gamma_entry = GammaEntry()
                gamma_entry.other = logprob[t, k] + exp_sum(new_label_prob, gamma[t - 1].other)
                gamma_entry.blank = logprob[t, 0] + exp_sum(gamma[t - 1].blank, gamma[t - 1].other)
                gamma.append(gamma_entry)
                prefix_prob = exp_sum(prefix_prob, (logprob[t, k] + new_label_prob))

            P_path_full = exp_sum(gamma[T - 1].blank, gamma[T - 1].other)
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
    logprob = np.zeros((2, 3))
    logprob[0,:] = np.log([0.6, 0.39, 0.01])
    logprob[1, :] = np.log([0.6, 0.39, 0.01])
    print(prefix_beam_search(logprob, 2, 2))
    #max path
    print([k for k in np.argmax(logprob, axis=1) if k != 0])