import networkx as nx
import nltk
from nltk import word_tokenize

## An implementation of the TextRank algorithm, as described in 'TextRank: Bringing Order into Texts'
## https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf

COOCCURENCE_WINDOW = 3
SYNTACTIC_FILTER = ["NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS"]

ITERATIONS = 30
THRESHOLD = 0.0001

class TextNode:
    def __init__(self, token, pos, occurences):
        self.token = token
        self.pos = pos
        self.occurences = occurences

    def precedesBy(self, other, k):
        i = 0
        j = 0
        while (i < len(self.occurences) and j < len(other.occurences)):
            if other.occurences[j] - self.occurences[i] == k:
                return True
            elif self.occurences[i] > other.occurences[j]:
                j = j + 1
            else:
                i = i + 1
        return False
    
    def __eq__(self, other):
        return self.token == other.token

    def __hash__(self):
        return hash(self.token)

    def __str__(self):
        return self.token

def tokenize(filename):
    return word_tokenize(open(filename, 'r').read())

def pre_process(filename):
    return nltk.pos_tag(tokenize(filename))
            
def build_graph(tokens):
    g = nx.Graph()
    occurence_map = {}
    for i, t in enumerate(tokens):
        if t[0] in occurence_map:
            occurence_map[t[0]].append(i)
        else:
            occurence_map[t[0]] = [i]
        node = TextNode(t[0], t[1], occurence_map[t[0]])
        if node.pos not in SYNTACTIC_FILTER:
            continue

        g.add_node(node)
        j = i - 1
        while j >= 0 and j >= (i - COOCCURENCE_WINDOW):
            prev_node = TextNode(tokens[j][0], tokens[j][1], occurence_map[tokens[j][0]])
            if prev_node.pos in SYNTACTIC_FILTER:
                g.add_edge(node, prev_node)
            j = j - 1
    return g

def top_k_nodes(nodes, k):
    return list(next(zip(*(sorted(nodes.items(), key=lambda x: x[1])[-k:]))))

def post_process(nodes):
    key_phrases = []
    visited = set()
    for n in nodes:
        if n in visited: continue
        visited.add(n)
        p = grow_phrase(n, nodes, visited)
        key_phrases.append(p)
    return key_phrases

def connected(n, phrase, precedes=True):
    for i, node in enumerate(phrase):
        if precedes and not n.precedesBy(node, i + 1): return False
        elif not precedes and not node.precedesBy(n, len(phrase) - i): return False
    return True

def grow_phrase(n, nodes, visited):
    phrase = [n]
    for m in nodes:
        updated = False
        for k in nodes:
            if connected(k, phrase, precedes=False):
                phrase.append(k)
                visited.add(k)
                updated = True
            elif connected(k, phrase):
                phrase = [k] + phrase
                visited.add(k)
                updated = True
        if not updated: break
    return ' '.join(str(n) for n in phrase)

def get_k_keywords(filename, k):
    tokens = pre_process(filename)
    graph = build_graph(tokens)
    start_values = {t:1.0 for t in graph.nodes()}
    ranked_nodes = nx.pagerank(graph, max_iter=ITERATIONS, tol=THRESHOLD, nstart=start_values)
    return post_process(top_k_nodes(ranked_nodes, k))
