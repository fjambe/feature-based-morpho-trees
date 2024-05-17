from collections import Counter

def compute_frequency(words):
    morphs = []
    frequency = {}
    for segmentation in words.values():
        for s in segmentation:
            morphs.append(s)

    total = len(morphs)
    occurrences = Counter(morphs)
    for mor, num in occurrences.items():
        frequency[mor] = round(num / total * 100, 2)
    return frequency

def predict_root(word, frequency):
    freqs = {}
    for seg in word:
        freqs[seg] = frequency.get(seg, 0)
    min_freq = min(freqs, key=freqs.get)
    rootnode = [s for s in word if s == min_freq][0]
    return rootnode


def read_file(filename):
    words = {}
    with open(filename, 'r', encoding='utf8') as input:
        content = input.readlines()
        for line in content:
            line = line.split('\t')[1].rstrip().replace(' @@', '@@').split(' ')
            for word in line:
                if '@@' in word: # i.e., if the word is segmented
                    words[word.replace('@@', '').lower()] = word.lower().split('@@')
        return words
