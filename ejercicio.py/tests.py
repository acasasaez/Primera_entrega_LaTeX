import numpy as np

from mmh import* 


states = ('Healthy', 'Fever')

observations = ('normal', 'cold', 'dizzy')

start_probability = {'Healthy': 0.6, 'Fever': 0.4}

transition_probability = {
    'Healthy': {'Healthy': 0.7, 'Fever': 0.3},
    'Fever': {'Healthy': 0.4, 'Fever': 0.6},
}

emission_probability = {
    'Healthy': {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
    'Fever': {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6},
}


def generate_index_map(lables):
    index_label = {}
    label_index = {}
    i = 0
    for l in lables:
        index_label[i] = l
        label_index[l] = i
        i += 1
    return label_index, index_label


states_label_index, states_index_label = generate_index_map(states)
observations_label_index, observations_index_label = generate_index_map(observations)

#print (states_label_index)
#print (states_index_label)
#print (observations_label_index)
#print (observations_index_label)

def convert_observations_to_index(observations, label_index):
    list = []
    for o in observations:
        list.append(label_index[o])
    return list

def convert_map_to_vector(map, label_index):
    v = np.empty(len(map), dtype=float)
    for e in map:
        v[label_index[e]] = map[e]
    return v

def convert_map_to_matrix(map, label_index1, label_index2):
    m = np.empty((len(label_index1), len(label_index2)), dtype=float)
    for line in map:
        for col in map[line]:
            m[label_index1[line]][label_index2[col]] = map[line][col]
    return m

A = convert_map_to_matrix(transition_probability, states_label_index, states_label_index)
#print (A)
B = convert_map_to_matrix(emission_probability, states_label_index, observations_label_index)
#print (B)
observations_index = convert_observations_to_index(observations, observations_label_index)
pi = convert_map_to_vector(start_probability, states_label_index)
#print (pi)

h = HMM(A, B, pi)
V, p = h.viterbi(observations_index)
print (" " * 7, " ".join(("%10s" % observations_index_label[i]) for i in observations_index))
for s in range(0, 2):
    print ("%7s: " % states_index_label[s] + " ".join("%10s" % ("%f" % v) for v in V[s]))
print ('\n The most possible states and probability are:')
p, ss = h.state_path(observations_index)
for s in ss:
    print (states_index_label[s],)
print(p) 