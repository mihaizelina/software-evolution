import csv
import sys
import nltk
import string
from nltk.corpus import stopwords

def write_output_file():
    '''
    Writes a dummy output file using the python csv writer, update this 
    to accept as parameter the found trace links. 
    '''
    with open('/output/links.csv', 'w') as csvfile:

        writer = csv.writer(csvfile, delimiter=",", quotechar="\"", quoting=csv.QUOTE_MINIMAL)

        fieldnames = ["id", "links"]

        writer.writerow(fieldnames)

        writer.writerow(["UC1", "L1, L34, L5"])
        writer.writerow(["UC2", "L5, L4"])

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

if __name__ == "__main__":
    '''
    Entry point for the script
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

    print(f"Hello world, running with matchtype {match_type}!")

    # Read input low-level requirements and count them (ignore header line).
    # with open("/input/low.csv", "r") as inputfile:
    #     print(f"There are {len(inputfile.readlines()) - 1} low-level requirements")

    low_dict = parse_input_file("dataset-1/low.csv")
    low_dict = preprocess(low_dict)
    print(low_dict)

    '''
    This is where you should implement the trace level logic as discussed in the 
    assignment on Canvas. Please ensure that you take care to deliver clean,
    modular, and well-commented code.
    '''

    # write_output_file()