import gzip
import os
import re
import sys
import math
import time
from nltk.stem.snowball import SnowballStemmer
import collections

# Trabalho realizado por
# Pedro Marques, 92926
# Inês leite, 92928

block_size = 3000000  # size of index to be written in blocks

snow_stemmer = None
stop_words = []  # List of stopwords to be readed from stopwords.txt

id_stream = []  # Stream which will contain all review ids
index = {}  # Final index
blockNumber = 0  # Number o fblock being written
doc_size = dict()

total_terms = 0
pointer = 1  # pointer to last added id
token_transforms = {}  # all previously calculated transformations
id_count = 0  # counter of all recorded review_ids
settings = dict()
term_idf_dic = dict()


indexWritten = 0
doc_size['avg'] = 0
term_list = []

results_dir = "index_results/"
search_dir = "search_results/"
top_n = 100

def main():
    global snow_stemmer, stop_words, index, top_n, compression, blockNumber, settings, term_list
    args = sys.argv[1:]
    min_length = 1  # minimum word length is 1 which equals to disabling the min length filter
    indexTime = 0
    start_time = time.time()
    if len(args) <= 1 or '-f' not in args:
        raise Exception(
            "Arguments needed: -f filename\n > Optional Arguments:  -m mininum length of token (inclusive), "
            "-s use stop words, "
            "-ps use stemmer, -c use compression.\n\t Example: indexer.py -f file.txt -i -m 2 -k1 1.2 -b 0.75")
    if "-top" in args:
        top_n = int(args[args.index('-top') + 1])

    isExist = os.path.exists(search_dir)
    if not isExist:
        os.mkdir(search_dir)
    else:
        [os.remove(os.path.join(search_dir, f)) for f in os.listdir(search_dir)]

    if '-i' in args:
        isExist = os.path.exists(results_dir)
        if not isExist:
            os.mkdir(results_dir)
        else:
            [os.remove(os.path.join(results_dir, f)) for f in os.listdir(results_dir)]


        filename = args[args.index('-f') + 1]

        if "-m" in args:  # -m =  minimum length of words
            min_length = int(args[args.index('-m') + 1])
        settings = {'f':filename,"len": min_length, 'w': 'lnc.ltc', 'top': None, 'k1': 1.2, 'b': 0.75}
        if "-w" in args:
            settings['w'] = args[args.index('-w') + 1].lower()
        if "-b" in args:
            settings['b'] = float(args[args.index('-b') + 1])
        if "-k1" in args:
            settings['k1'] = float(args[args.index('-k1') + 1])



        snow_stemmer = SnowballStemmer(language='english')


        stop_words = [x.replace('\n', '') for x in open('stopwords.txt', 'r').readlines()]


        save_settings()
        row_count = 0
        for text, _id in open_file(filename, min_length):
            # read file line by line and operate over each line and respective review_id
            st = time.time()
            index = spimi(text, _id)  # indexing algorithm
            indexTime += time.time() - st
            row_count += 1
            if row_count % 10000 == 0:  # unnecessary, just terminal data
                print('Read ' + str(row_count) + ' rows in ' + (str(time.time() - start_time)) + "s")

        if settings['w'].split('.')[0]=="bm25":
            for key in doc_size:
                doc_size['avg'] += doc_size[key]
            doc_size['avg'] = doc_size['avg'] / (len(doc_size) - 1)

        write_file(f'BLOCK{blockNumber}.txt')  # write final block (accounts for leftover data notyet recorded)
        blockNumber += 1  # number of block

        merge_blocks()  # merge all block files
    else:
        settings = load_settings()
        if not settings:
            raise Exception('No previous settings recorded, first do indexing')

    #Reload id_stream from id_stream.txt file and documents size

    load_id_stream()
    load_doc_size()
    load_idf()
    list_of_ids = id_stream.split("#")
    #Start search function
    f = open('queries.txt')
    for query in f.readlines():
        results, query = load_search(query)
        final_results = []
        index.clear()
        #If compression has been used, replace pointers with actual document ids

        for id, score in results[:top_n]:
            final_results.extend([(id_stream[id:id_stream.find('#', id)], score)])
        ids_data = get_data(final_results, settings['f'],list_of_ids, settings["len"])
       
        write_results(final_results, query)

def get_data(ids, filename, list_of_ids, min_length):
    global id_stream
    with gzip.open(filename, 'rb') as f:
        # ReviewId, ReviewHeadline + ReviewBody
        ids_indexes = sorted([(x[0],list_of_ids.index(x[0])+1) for x in ids], key = lambda item: item[1])
        data = {k[0]:[] for k in ids_indexes}

        line_index = 0
        for rev in ids_indexes:
            while line_index<rev[1]-1:
                f.readline()
                line_index+=1
            line_index+=1
            row = f.readline().decode(encoding='utf-8')
            row = row.split('\t')
            text = f'{row[-3]} {row[-2]}'.lower()  # Remove all non-alphabetic chars
            data[rev[0]] = text
    return data


def save_settings():
    #Save indexation settings and parameters to file
    global settings
    _f = open(f'{results_dir}settings.txt', 'a+')
    for s in [x for x in settings if x !='top']:
        _f.write(f'{s}:{settings[s]}\n')


def load_settings():
    #Load indexation settings from file
    global snow_stemmer, stop_words
    _s = dict()
    f = open(f'{results_dir}settings.txt', 'r')
    data = f.readlines()
    for l in data:
        s, v = l.replace('\n','').split(':')
        _s[s] = v
        if s in ['b', 'k1']:
            _s[s] = float(v)
        elif s == 'len':
            if v!='None':
                _s[s] = int(v)
            else:
                _s[s] = None


    snow_stemmer = SnowballStemmer(language='english')

    stop_words = [x.replace('\n', '') for x in open('stopwords.txt', 'r').readlines()]

    return _s


def load_id_stream():
    global id_stream
    id_stream = open(f'{results_dir}id_stream.txt', 'r').read()



def load_search(query = None):
    global settings, index, id_count, doc_size,top_n
    #Ask for a query to search
    if not query:
        query = input('Query: ')
    query_weights = dict()
    print(query)
    bm25_query_weights = dict()
    #Normalize query as used in indexation
    query_terms = re.sub("[^a-z ]+", '', query.lower()).split(" ")  # Remove all non-alphabetic chars
    tokens = [x for x in query_terms if len(x) >= settings['len'] and x not in stop_words]
    transformed_tokens = [apply_transformation(x) for x in tokens]

    counter = collections.Counter(transformed_tokens)

    combinations = settings['w'].split('.')
    if type(top_n) == float:
        top_n = int(top_n)
    if combinations[0] == 'bm25':
        for term in transformed_tokens:

            index = load_chunk_from_index(term)

            #Calculate final scores
            for _id in index[term]:
                if _id in bm25_query_weights:
                    bm25_query_weights[_id] += index[term][_id]
                else:
                    bm25_query_weights[_id] = index[term][_id]

        return sorted(bm25_query_weights.items(), key=lambda item: item[1], reverse=True)[:top_n], query

    else:
        index = dict()
        for term in counter:
            _temp = load_chunk_from_index(term)

            tf = termFrequency(counter[term], mode=combinations[0])

            if term in _temp:
                index[term] = _temp[term]

                w = round(tf * term_idf_dic[term], 2)

                if int(w) == w:
                    w = int(w)
                query_weights[term] = w

            else:
                query_weights[term] = 0

        total_doc_size = round(math.sqrt(sum([x ** 2 for x in query_weights.values()])), 2)
        #Calculate normalized weights for terms in query
        for term in [x for x in query_weights if x in index]:

            if combinations[1][2] == 'c':
                query_weights[term] = query_weights[term] / total_doc_size


        #Normalize weights with doc lengths
        normalized = normalize_weights([x for x in counter if x in index], settings['w'][2])

        return compute_scores(query_weights, normalized), query


def compute_scores(query_weights, index_weights):
    #Compute final scores and sort them
    score = dict()
    for _id in index_weights:
        score[_id] = sum([index_weights[_id][k] * query_weights[k] for k in query_weights])
    return sorted(score.items(), key=lambda item: item[1], reverse=True)[:top_n]


def load_chunk_from_index(word):
    #only load the part of the index.txt file which contains the term 'word'
    #This mapping between 'word' and chuk location is described in the
    global settings
    lines = open(f'{results_dir}term_index_mapper.txt', 'r').readlines()
    previous = 0
    _index = dict()
    for i in range(len(lines)):
        line = lines[i]
        words, offset = line.split(':')
        offset = int(offset)
        min, max = words.split('-')
        if word >= min and word <= max:
            _f = open(f'{results_dir}index.txt', 'r')

            _f.seek(previous)
            data = _f.read(offset - previous)

            index_lines = [x for x in data.split('\n') if x != '' and '=' in x and ';' in x]

            for _l in index_lines:
                _l = _l.replace(';','')

                ids = _l.split('=')[1].replace('|', ' ').split(' ')

                # if compression, ids will be represented by integer pointers
                ids_count = {int(k.split(':')[0]): float(k.split(':')[1]) for k in ids}

                _index[(_l.split('=')[0])] = ids_count

        previous = offset

    return _index


def open_file(file, min_length):  # Reads file inside zip archive and yields each line after it being processed
    with gzip.open(file, 'rb') as f:
        # ReviewId, ReviewHeadline + ReviewBody
        line = f.readline()  # ignore first line (contains column names)
        line = f.readline()
        while line:
            yield get_terms((line.decode(encoding='utf-8')).lower(), min_length)
            line = f.readline()


def get_terms(row, min_length):  # Returns terms from text as array
    global tokenTime, id_stream, pointer, stop_words
    st = time.time()
    row = row.split('\t')
    text = re.sub("[^a-z \t]+", '', f'{row[-3]} {row[-2]}').lower().split(" ")  # Remove all non-alphabetic chars


    # If we are applying compression
    # join all ids in a string
    id_stream.extend([row[2]])
    size = len(row[2])
    pointer += size + 1  # pointer accounts for '#' char
    final_pointer = pointer - size - 1
    # pointer to id on string will be used on dictionary


    # Next, filter all words with size under min_length and remove words in stop-words list
    tokens = [x for x in text if len(x) >= min_length and x not in stop_words]


    return [apply_transformation(x) for x in tokens], final_pointer  # Turn tokens from text into terms


def apply_transformation(string):  # apply snow_stemmer and string compression
    if string in token_transforms:
        # if string has been transformed before, use previously calculated value
        return token_transforms[string]
    # apply stemmer and compression to token
    compressed = string_compression(normalize_token(string))
    token_transforms[string] = compressed  # save value for usage in next operations

    return compressed


def normalize_token(token):  # Use snow stemmer on word from text
    stem = snow_stemmer.stem(token)
    return stem


def string_compression(string):
    # turn words like AAABAA into A3BA2
    last = string[0]
    result = []
    count = 1
    for c in string[1:]:
        if c == last:
            count += 1
        elif count == 1:
            result.append(last)
            last = c
        else:
            result.append(last + str(count))
            count = 1
            last = c
    result.append(last)
    if count != 1:
        result.append(str(count))
    final_string = ''.join(result)
    return final_string


def spimi(text, _id):  # Indexation algorithm
    global index, blockNumber, indexTime, block_size, id_stream, id_count, doc_size

    text = collections.Counter(text)
    start_time = time.time()
    doc_size[str(_id)] = 0
    for term in text:
        t_count = termFrequency(text[term], mode=settings['w'])
        # if bm25
        if settings['w'] == 'bm25':
            doc_size[str(_id)] += t_count
        # for each term in array of transformed tokens
        # save the corresponding review_id and term frequency
        if term not in index:
            index[term] = {_id: t_count}
        else:
            index[term][_id] = t_count
    id_count += 1

    if sys.getsizeof(index) >= block_size:
        # if index size is greater than block_size (default = 75 000)
        # write block and current id_stream to disk to free memory
        write_file(f'BLOCK{blockNumber}.txt')
        blockNumber += 1  # increase block count
        # clear out some memory

        id_stream.clear()  # clear id_stream

        index.clear()  # clear index from memory

    return index


def write_file(filename):  # used to write blocks and final index into disk
    global id_stream, term_list


    # add current id_stream to memory
    id_file = open(f'{results_dir}id_stream.txt', 'a+', encoding='utf-8')
    # append current id_stream to 'id_stream.txt'
    # this id_stream is cleared in spimi in order to save up some memory
    id_file.write('#' + '#'.join(id_stream))

    # Create block or index file
    open(filename, 'w', encoding='utf-8')

    # Each line will be added by appending
    block_file = open(filename, 'a+', encoding='utf-8')
    _srt = sorted(index.keys())
    # Writing index to file as: word:ID1POINTER|ID2POINTER|ID3POINTER...
    for token in _srt:
        items_str = '|'.join([f'{k}:{v}' for k, v in index[token].items()])
        block_file.write(f'{token}={items_str}\n')

    block_file.close()



def readline_from_file(file):
    #reads the first line of a given file
    #and turns it into an entry in the index
    global settings
    line = file.readline()
    if not line:
        return None
    line = line.replace('\n', '')
    ids = line.split('=')[1].replace('|', ' ').split(' ')
    # if compression, ids will be represented by integer pointers
    ids_count = {int(k.split(':')[0]): float(k.split(':')[1]) for k in ids}
    return (line.split('=')[0], ids_count)


def merge_blocks():  # Merge all written block files into a single large index and then write it to disk
    global index, mergeTime, indexWritten, id_stream, total_terms, blockNumber, id_count

    # first get all recorded transformed tokens = record all index entries for all blocks
    all_tokens = sorted(set(token_transforms.values()))
    total_terms = len(all_tokens)  # statistical data
    token_transforms.clear()  # clear memory -> token_transforms dictionary no longer needed
    st = time.time()
    index.clear()
    index = dict()
    load_id_stream()
    filenames = [f'BLOCK{i}.txt' for i in range(0, blockNumber)]
    files = [open(i, 'r') for i in filenames]
    id_count = len([x for x in id_stream.split('#') if x != ''])
    id_stream = ''

    index_file = open(f'{results_dir}index.txt', 'a+')

    term_mapper = open(f'{results_dir}term_index_mapper.txt', 'a+')

    # Gets the first line of each BLOCK document
    first_line = [readline_from_file(files[k]) for k in range(blockNumber)]

    total_size = 0
    initial_term = None
    lowest_term = None

    while first_line != [None] * blockNumber:
        #   While there are lines to process
        #   Process the lines in alphabetic order, this way, the last processed line will corresponde
        #    to the final term of the final index
        lowest_terms = sorted([_tuple[0] for _tuple in first_line if _tuple])
        if not lowest_terms:
            break

        lowest_term = lowest_terms[0]

        if not initial_term:
            initial_term = lowest_term

        index[lowest_term] = dict()
        files_with_term = [i for i in range(blockNumber) if first_line[i] and first_line[i][0] == lowest_term]
        for i in files_with_term:
            index[lowest_term] = join_dics(index[lowest_term], first_line[i][1])
            first_line[i] = readline_from_file(files[i])

        calculate_index_weights(lowest_term)    #Calculate weight for newest index entry

        #If current merged index size exceeds threshold
        #append new index to index.txt file
        if sys.getsizeof(index) > block_size:
            total_size = write_index(initial_term, lowest_term,total_size, index_file, term_mapper)
            initial_term = None

    write_index(initial_term, lowest_term, total_size, index_file, term_mapper)
    mergeTime = time.time() - st
    save_doc_sizes()
    st = time.time()
    [f.close() for f in files]
    [(os.remove(f)) for f in filenames]  # remove all temporary block files
    write_idf()
    indexWritten = time.time() - st


def save_doc_sizes():
    global doc_size
    f = open(f'{results_dir}doc_len.txt', 'a+')
    for doc in sorted([str(x) for x in (doc_size.keys())]):
        f.write(f'{doc}:{doc_size[doc]}\n')
    doc_size.clear()


def load_doc_size():
    #Load documents sizes from file doc_len.txt into memory
    global doc_size
    f = open(f'{results_dir}doc_len.txt', 'r')
    data = f.readlines()
    for line in data:
        _id, s = line.replace('\n', '').split(':')
        doc_size[_id] = s


def write_index(initial_term, lowest_term, total_size, index_file, term_mapper):
    #Append index line to index.txt file
    _write_size = 0
    for token in index:
        items_str = '|'.join([f'{k}:{v}' for k, v in index[token].items()])
        to_write = f'{token}={items_str};\n'
        _write_size += len(to_write)
        index_file.write(to_write)
    total_size+=_write_size
    #Map the appended terms to a chunk of the index.txt file
    # Example:
    #   term1-term30:70000bits
    #   term31-term60:140000bits
    term_mapper.write(f'{initial_term}-{lowest_term}:{index_file.tell()}\n')
    index.clear()
    return total_size


def write_idf():
    #write calculated idf values to idf.txt file
    global term_idf_dic
    term_file = open(f'{results_dir}idf.txt', 'a+')
    term_file.write(f"{settings['w']}\n")
    for term in term_idf_dic:
        term_file.write(f'{term}:{term_idf_dic[term]}\n')

def load_idf():
    #Load calculated idf values from idf.txt file
    global term_idf_dic, settings
    term_file = open(f'{results_dir}idf.txt', 'r')
    line = term_file.readline()


    line = term_file.readline()
    while line:
        _l = line.split(':')
        term_idf_dic[_l[0]] = float(_l[1])
        line = term_file.readline()

def bm25(N, df, tf, dl, avdl):
    #Returns BM25 normalized weight
    global settings
    b = settings['b']
    k = settings['k1']
    idf = math.log10(N / df)
    up = (k + 1) * tf
    down = (k * ((1 - b) + b * dl / avdl) + tf)
    return round(idf * up / down, 2), idf


def calculate_index_weights(term):
    #   Considers the ranking method selected in order
    # to calculate the final weight of each term in each document

    global index, settings, doc_size, id_count
    mode = settings['w']
    if mode == 'bm25':
        for _id in index[term]:
            #For each document where term occurs
            tf = index[term][_id]
            df = len(index[term])

            #Replace values next to document ids in index with final weights
            index[term][_id], term_idf = bm25(id_count, df, tf, doc_size[str(_id)], doc_size['avg'])
            term_idf_dic[term] = term_idf


    else:
        for _id in index[term]:
            #For each document where term occurs
            term_idf = idocumentFrequency(len(index[term]), mode=settings['w'][1])
            #Calculate non-normalized weight
            w = round(index[term][_id] * term_idf, 2)

            if int(w) == w:
                w = int(w)
            #Update document size in doc_size dictionary
            doc_size[str(_id)] += w ** 2

            term_idf_dic[term] = term_idf
            #Replace term frequency values next to document ids in index with final weights
            index[term][_id] = w


def normalize_weights(query_terms, mode='c'):
    global index, doc_size
    #Method not used with bm25 ranking
    #Normalize weights with calculated document length
    ids = set([k for term in query_terms for k in index[term]])

    normalized = dict()
    if mode == 'c':
        for _id in ids:
            normalized[_id] = dict()

            _size = math.sqrt(float(doc_size[str(_id)]))
            for term in query_terms:
                w = 0
                if _id in index[term]:
                    w = index[term][_id]
                normalized[_id][term] = w / _size
    return normalized


def idocumentFrequency(no_docs, mode='n'):
    # log(Total Number Of Documents / Number Of Documents with term in it)
    global id_count, term_idf_dic

    if mode == 'n':
        return 1
    else:
        return math.log(
            id_count/ no_docs)


def termFrequency(_count, mode='lnc'):
    global settings
    if mode == 'bm25':
        return _count

    elif mode[0] == 'l':
        _log = math.log(_count)

        return 1 + _log
    elif mode[0] == 'b':
        return 1 if _count > 0 else 0

    else:
        return _count


def join_dics(dic1, dic2):
    #Join 2 dictionaries into 1
    for x in dic2:
        dic1[x] = dic2[x]
    return dic1


def write_results(final_results,query):
    with open(f'{search_dir}{settings["w"]}.txt', 'a+') as f:
        f.write(f'\nQ: {query}\n')
        [f.write(f'{line[0].upper()}\n') for line in final_results]
if __name__ == "__main__":
    main()
