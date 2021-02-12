import csv
import sys
import nltk
import string
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from collections import Counter
from collections import OrderedDict

def write_output_file(links):
    '''
    Writes the links to an output file.
    Takes as input a list of lists containing a high requirement as the first element and 
    the linked low requirements as the rest of the elements.
    '''
    with open('output/links.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=",", quotechar="\"", quoting=csv.QUOTE_MINIMAL)

        fieldnames = ["id", "links"]

        writer.writerow(fieldnames)
        for row in links:
            writer.writerow(row)

def parse_input_file(filename):
    '''
    Parses a two-column CSV file and returns a dict between requirement ID and its full text. 
    '''
    try:
        inputfile = open(filename, "r")
    except ValueError as e:
        print("Invalid input file name")
        exit(2)
    
    inputlines = inputfile.readlines()

    lines = {} # stores req in the form "id : text"
    
    for line in inputlines[1:]:
        split = line.split(',', 1)
        lines[split[0]] = split[1]
    
    return lines

def tokenize(inputtext, dupes = False):
    '''
    Returns the lowercase words of the input text as list (case-insensitive).
    '''
    tokenlist = inputtext.lower().translate(str.maketrans('', '', string.punctuation)).split()
    if dupes == True:
        # Include duplicates
        return tokenlist
    return list(set(tokenlist))

def remove_stopwords(words):
    '''
    Takes as input a list of words and returns the words list sans stop-words.
    '''
    stop_words = set(stopwords.words('english'))
    filtered = [w for w in words if not w in stop_words]
    return filtered

def stem_tokens(words, stemmer = 'snowball'):
    '''
    Takes as input a list of words and returns the words stems of the input list.
    '''
    if stemmer == 'snowball':
        sno = nltk.stem.SnowballStemmer('english')
        words = [sno.stem(word) for word in words]
    elif stemmer == 'lemma':
        lemma = nltk.wordnet.WordNetLemmatizer()
        words = [lemma.lemmatize(word) for word in words]
    elif stemmer == 'porter':
        po = nltk.stemmer.PorterStemmer()
        words = [po.stem(word) for word in words]
    return words

def preprocess(req_dict):
    '''
    Takes as input a requirements dict and replaces the texts with their tokenized and stemmed counterparts.
    Returns the newly-formed dict.
    '''
    for req_id in req_dict.keys():
        keywords = req_dict[req_id]
        keywords = tokenize(keywords) # tokenize the text strings
        keywords = remove_stopwords(keywords) # remove stop-words
        keywords = stem_tokens(keywords) # stem words
        req_dict[req_id] = keywords
    return req_dict

def req_words(req_dict):
    '''
    Takes as input a requirements dict and returns the set of all words used in it
    '''
    return set([item for sublist in list(req_dict.values()) for item in sublist])

def count_freq(master_vocab, dict):
    '''
    Takes as input a master vocabulary and a requirements dict.
    Returns a dataframe containing word frequency vectors of size len(master_vocab) for each requirement
    and a numpy array of size len(master_vocab) representing the number of requirements containing each word
    '''
    vectors = pd.DataFrame(columns = master_vocab)
    vocab_d = np.array([0] * len(master_vocab))
    for req, words in dict.items():
        vector = [value - 1 for _, value in (Counter(master_vocab) + Counter(words)).items()]
        vocab_d = np.add(vocab_d, np.array([(1 if x != 0 else 0) for x in vector]))
        vectors.loc[req] = vector
    return vectors, vocab_d

def vectorize(high_dict, low_dict):
    '''
    Takes as input a high level and a low level requirements dict.
    Returns two dataframes (high and low) representing the vectors associated with each requirement 
    in each dict.
    '''
    master_vocab = list(req_words(high_dict) | req_words(low_dict))
    
    vectors_high, vocab_d_high = count_freq(master_vocab, high_dict)
    vectors_low, vocab_d_low = count_freq(master_vocab, low_dict)

    # idf = log2(vocab_n/vocab_d), where vocab_n is len(master_vocab) and d is the number of 
    # requirements containing the ith word
    vocab_n = np.array([len(master_vocab)] * len(master_vocab))
    vocab_d = np.add(vocab_d_high, vocab_d_low)
    n_div_d = np.divide(vocab_n, vocab_d, out=np.zeros_like(vocab_n, dtype = 'float'), where = vocab_d != 0)
    df = np.log2(n_div_d, out=np.zeros_like(n_div_d, dtype = 'float'), where = n_div_d != 0)

    vectors_high = np.multiply(vectors_high, df)
    vectors_low = np.multiply(vectors_low, df)

    return (vectors_high, vectors_low)

def sim_matrix(vectors_high, vectors_low):
    '''
    Takes as input two dataframes representing the vectors associated with each high req and 
    with each low req respectively.
    Returns a similarity matrix of size H x L, where H is the number of high reqs and L is 
    the number of low reqs.
    '''
    matrix = pd.DataFrame(columns = vectors_low.index)
    for index, _ in vectors_high.iterrows():
        matrix.loc[index] =  np.array([0] * len(vectors_low.index))
        for column in matrix:
            vrh = vectors_high.loc[index]
            vrl = vectors_low.loc[column]
            cos_sim = (vrh @ vrl.T) / (np.linalg.norm(vrh)*np.linalg.norm(vrl))
            matrix.at[index, column] = cos_sim

    return matrix

def trace_link(sim_matrix, eval):
    '''
    Returns (something) containing trace links according to similarity matrix and evaluation function.
    '''
    res = []
    for index, _ in sim_matrix.iterrows():
        link = sim_matrix.loc[index][eval(sim_matrix.loc[index])]
        res.append([link.name, ','.join(str(req) for req in link.index.values)])

    return res

def trace_link_print(sim_matrix, eval):
    '''
    Prints trace links according to similarity matrix and evaluation function.
    '''
    for index, _ in sim_matrix.iterrows():
        link = sim_matrix.loc[index][eval(sim_matrix.loc[index])]
        formatted_link = link.name + ": {" + ', '.join(str(req) for req in link.index.values) + "}"
        print(formatted_link)

def eval_func(type):
    '''
    Returns a lambda function depending on the input match type.
    '''
    if type == 0:
        # no filtering
        return lambda x: x > 0
    elif type == 1:
        # all l such that sim(h, l) > 0.25
        return lambda x: x >= 0.25
    elif type == 2:
        # all l' such that, for l with highest similarity score, sim(h, l') >= 0.67 * sim(h, l)
        return lambda x: x >= 0.67 * x.max()
    else:
        raise ValueError('Match type not recognized.')

if __name__ == "__main__":
    '''
    Entry point for the script.
    '''
    if len(sys.argv) < 2:
        print("Please provide an argument to indicate which matcher should be used")
        exit(1)

    match_type = 0

    try:
        match_type = int(sys.argv[1])
    except ValueError as e:
        print("Match type provided is not a valid number")
        exit(1)

    print(f"Running with matchtype {match_type}!")

    # Read input low-level requirements and count them (ignore header line).
    # with open("/input/low.csv", "r") as inputfile:
    #     print(f"There are {len(inputfile.readlines()) - 1} low-level requirements")

    high_dict = parse_input_file("dataset-1/high.csv")    
    high_dict = preprocess(high_dict)
    low_dict = parse_input_file("dataset-1/low.csv")
    low_dict = preprocess(low_dict)

    vectors_high, vectors_low = vectorize(high_dict, low_dict);
    sim = sim_matrix(vectors_high, vectors_low)

    # Pass lambda function to evaluate
    links = trace_link(sim, eval_func(match_type))
    trace_link_print(sim, eval_func(match_type))

    write_output_file(links)