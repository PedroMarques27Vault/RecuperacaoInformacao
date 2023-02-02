import gzip
import os
import re
import sys
import time
from nltk.stem.snowball import SnowballStemmer

#Trabalho realizado por
    #Pedro Marques, 92926
    #Inês leite, 92928

block_size = 750000 #size of index to be written in blocks
MAX_ID_COUNTER = 150000 #number of maximum ids to be recorded before saving block

snow_stemmer = None
stop_words = []     #List of stopwords to be readed from stopwords.txt

id_stream = ''      #Stream which will contain all review ids
index = {}          #Final index
blockNumber = 0     #Number o fblock being written

compression = False #Use compression or not
total_terms = 0
pointer = 1         #pointer to last added id
token_transforms = {}   #all previously calculated transformations
id_count = 0        #counter of all recorded review_ids
# Time measurements
tokenTime = 0
indexTime = 0
mergeTime = 0
stemTime = 0
writeBlockTime = 0
compressionTime = 0
indexWritten = 0


def main():
    global snow_stemmer, stop_words, index, indexTime, compression, blockNumber
    args = sys.argv[1:]
    min_length = 1          #minimum word length is 1 which equals to disabling the min length filter
    if len(args) <= 1 or '-f' not in args:
        raise Exception(
            "Arguments needed: -f filename\n > Optional Arguments:  -m mininum length of token (inclusive), "
            "-s use stop words, "
            "-ps use stemmer, -c use compression.\n\t Example: indexer.py -f file.txt -m 2 -s -ps -c")

    open('id_stream.txt', 'w')  #Create file if it does not exist or delete pre-existing 

    filename = args[args.index('-f') + 1]
    if "-m" in args:            #-m =  minimum length of words
        min_length = int(args[args.index('-m') + 1])
    settings = {"len": min_length}

    if "-ps" in args:           #-ps = if in arguments, using snowstemmer
        snow_stemmer = SnowballStemmer(language='english')
        settings['ps'] = True

    if "-s" in args:            #-s = if in arguments, using stopwords
        stop_words = [x.replace('\n', '') for x in open('stopwords.txt', 'r').readlines()]
        settings["s"] = True
    if "-c" in args:            #if in arguments use compression techniques
        compression = True
        settings["c"] = True

    start_time = time.time()
    row_count = 0
    for text, _id in open_file(filename, min_length):   #read file line by line and operate over each line and respective review_id
        st = time.time()
        index = spimi(text, _id)                                    #indexing algorithm
        indexTime += time.time() - st
        row_count += 1
        if row_count % 10000 == 0:                      #unnecessary, just terminal data
            print('Read ' + str(row_count) + ' rows in ' + (str(time.time() - start_time)) + "s")

    write_file(f'BLOCK{blockNumber}.txt')               #write final block (accounts for leftover data notyet recorded)
    blockNumber += 1                                    #number of blocks

    indexTime -= writeBlockTime                         #from indexation time remove time used while writing blocks to disk

    
    merge_blocks()                                      #merge all block files
    index.clear()                                       #clear index from memory to clear memory for other usage
    total_time = time.time() - start_time
    word, ids_returned, t = load_search()               #begin search for specific word
    write_times_to_file(filename, settings, total_time, (word, ids_returned, t))    #write statistical data to index.txt


def open_file(file, min_length):                        #Reads file inside zip archive and yields each line after it being processed 
    with gzip.open(file, 'rb') as f:
        # ReviewId, ReviewHeadline + ReviewBody
        for line in f.readlines()[1:]:                  #ignore first line (contains column names)
            yield get_terms((line.decode(encoding='utf-8')).lower(), min_length)

def get_terms(row, min_length):                         #Returns terms from text as array
    global tokenTime, id_stream, pointer, stop_words
    st = time.time()
    row = row.split('\t')
    text = re.sub("[^a-z \t]+", '', f'{row[-3]} {row[-2]}').lower().split(" ") #Remove all non-alphabetic chars

    if compression:
        #If we are applying compression
        #join all ids in a string
        id_stream = '#'.join([id_stream, row[2]])

        pointer += len(row[2]) + 1              #pointer accounts for '#' char
        final_pointer = pointer - len(row[2]) - 1
        #pointer to id on string will be used on dictionary
    else:
        final_pointer = row[2]
        #retrieve review id
    
    #Next, filter all words with size under min_length and remove words in stop-words list
    tokens = [x for x in text if len(x) >= min_length and x not in stop_words]

    tokenTime += time.time() - st
    return [apply_transformation(x) for x in tokens], final_pointer           #Turn tokens from text into terms


def apply_transformation(string):     #apply snow_stemmer and string compression
    if string in token_transforms:
        #if string has been transformed before, use previously calculated value
        return token_transforms[string]
    #apply stemmer and compression to token
    compressed = string_compression(normalize_token(string))
    token_transforms[string] = compressed   #save value for usage in next operations
    return compressed

def normalize_token(token): #Use snow stemmer on word from text
    if snow_stemmer:        #if snow_stemmer option is enabled
        global stemTime
        st = time.time()
        stem = snow_stemmer.stem(token) 
        stemTime += time.time() - st
        return stem
    return token

def string_compression(string):
    #turn words like AAABAA into A3BA2
    if not compression:     #If compression is disabled return string without alterations
        return string
    global compressionTime
    st = time.time()
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
    compressionTime += time.time() - st
    final_string = ''.join(result)
    return final_string




def spimi(text, _id):       #Indexation algorithm
    global index, blockNumber, indexTime, block_size, id_stream, id_count, MAX_ID_COUNTER

    start_time = time.time()
    for term in text:       
        #for each term in array of transformed tokens
        #save the corresponding review_id
        if term not in index:
            index[term] = f'{_id}'
        else:
            index[term] = '|'.join([index[term],str(_id)])
    id_count+=1
    if sys.getsizeof(index) >= block_size or id_count>=MAX_ID_COUNTER:
        #if index size is greater than block_size (default = 75 000)
        #write block and current id_stream to disk to free memory
        write_file(f'BLOCK{blockNumber}.txt')
        blockNumber += 1    #increase block count
        #clear out some memory
        id_stream = ''      #clear id_stream
        id_count = 0        #reset current id counter
        index.clear()       #clear index from memory

    indexTime += time.time() - start_time
    return index



def write_file(filename):       #used to write blocks and final index into disk
    global writeBlockTime, id_stream
    start_time = time.time()

    #add current id_stream to memory
    id_file = open('id_stream.txt', 'a+', encoding='utf-8') 
    
    #append current id_stream to 'id_stream.txt' 
    #this id_stream is cleared in spimi in order to save up some memory
    id_file.write(id_stream)    
    
    #Create block or index file
    open(filename, 'w', encoding='utf-8')

    #Each line will be added by appending
    block_file = open(filename, 'a+', encoding='utf-8')

    #Writing index to file as: word:ID1POINTER|ID2POINTER|ID3POINTER...
    for token in index.keys():
        block_file.write(f'{token}:{index[token]}\n')

    block_file.close()

    writeBlockTime += time.time() - start_time

def merge_blocks():         #Merge all written block files into a single large index and then write it to disk
    global index, mergeTime, indexWritten, id_stream, total_terms, blockNumber

    #first get all recorded transformed tokens = record all index entries for all blocks
    all_tokens = sorted(set(token_transforms.values()))
    total_terms = len(all_tokens)       #statistical data
    token_transforms.clear()            #clear memory -> token_transforms dictionary no longer needed
    st = time.time()
    index.clear()                       #clear in-memory index

    id_stream = ''                      #clear memory by clearing id_stream

    index = {k: '' for k in all_tokens} #initialize final dictionary with default value for each term
    
    file_no = 0                         #file number to read from
    while file_no < blockNumber:    
        #read blocks of 50 files -> increases performance
        #loading many blocks into disks risks memory errors
        end = file_no + 50
        if end > blockNumber:   #last file to be read will be the one numbered blockNumber
            end = blockNumber
        tokens_in_files = set() 
        data = [read_file_to_dic(f'BLOCK{i}.txt') for i in range(file_no, end)]     #load all data from files number between file_no and end

        [tokens_in_files.add(x) for dics in data for x in dics ]    #only use tokens in these files

        for token in tokens_in_files:   #iterate over each token
            current_token_ids = []  
            for _index in data:
                if token in _index:
                    current_token_ids += [str(x) for x in _index[token]]    #join all data from each block file

            index[token] = '|'.join(set(index[token].split('|')+current_token_ids)) #remove all repeated indexes and add data form block files to final big index

        file_no = end

    mergeTime = time.time() - st
    st = time.time()
    write_file('index.txt')     #finally, write big index into disk
    indexWritten = time.time() - st
    [(os.remove('BLOCK' + str(i) + ".txt")) for i in range(0, blockNumber)] #remove all temporary block files
    

def read_file_to_dic(filename): #reads data from file filename into dictionary
    f = open(filename, 'r')
    print('Reading file ' + filename)
    line = f.readline()
    _temp = {}
    while line:
        line = line.replace('\n', '')
        ids = line.split(':')[1].split('|') 
        if compression: #if compression, ids will be represented by integer pointers
            ids = [int(x) for x in ids]
        _temp[(line.split(':')[0])] = ids
        line = f.readline() 

    return _temp        #dictionary=> term: list_of_ids

def load_search():         #initialize a seach menu
    global index
    while True:
        #Show menu to enter word for search in index
        word = input('Enter a word or (E)nd: ') 

        if word == "E": #writing 'E' terminates the program
            break
        elif len(word) <= 2:    #Minimal word length must be 3
            raise Exception("Word is not allowed: Minimal length is 3")
        st = time.time()
        token = string_compression(normalize_token(word))   #apply same transformations as terms in index 

        #instead of loading entire index (occupies too much space)
        #only load line corresponding to the given term
        line = get_line_from_file(token)                    
        if line == -1:
            print('Word has not been recorded')
            return word, [], 0
        else:
            values = sorted([x for x in line if len(x) > 0])

            x = len(values)     #number of review_ids
            search_time = time.time() - st
            print(f'Found results in {search_time} seconds')
            print(f'Word occurs in {x} reviews')
            _ids = []
            if compression:
                #if compression, turn pointers into real review_ids
                id_stream = open('id_stream.txt').read()            #load id_stream into memory 
                values = [int(x) for x in values]                   #turn pointers into integers
                for _id in values:
                    final_index = id_stream.find('#', int(_id))
                    _ids.append(id_stream[int(_id):final_index].upper())    #append all review_ids where word occurs to list _ids

            else:
                #list _ids contains all review_ids associated to the given word
                _ids = [x.upper() for x in values]


            return word, _ids, search_time  #return given word, all review_ids and total search time
    return 'No Word Was Searched', [], 0

def get_line_from_file(token):  
    #look for a specific line of a term in disk
    #return code -1 represents not found
    f = open('index.txt', 'r')
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    alphabet_without = alphabet.split(token[0])[0] + token[0]
    line = f.readline()
    while line:
        #only process line if first char of line (1st char of word) 
        #comes before the first char of given token in the alphabet
        if line[0] not in alphabet_without: 
            return -1
        elif line.split(':')[0] == token: #if word in line is the given token, return list of _ids/pointers 
            return line.split(':')[1].split('|')
        else:           #else keep looking
            line = f.readline()
    return -1


def write_times_to_file(filename, settings, total_time, search):        #Write All Statistics To File times.txt
    with open('times.txt', 'a+') as f:
        f.write("\n" + "-" * 33 + filename + "-" * 33)
        f.write("\nMinimum word length of " + str(settings['len']) + "\n")
        if 's' in settings:
            f.write('Removing Stop Words\n')
        else:
            f.write('Not Removing Stop Words\n')
        if 'ps' in settings:
            f.write('Using Snowball\n')
        else:
            f.write('Not Using Snowball\n')
        if 'c' in settings:
            f.write('Using compression\n')
        else:
            f.write('Not Using Compression\n')

        f.write('Word tokenization took ' + str(tokenTime) + " seconds\n")
        f.write('Stemmer took ' + str(stemTime) + " seconds\n")
        f.write('Compression took ' + str(compressionTime) + " seconds\n")
        f.write('Indexing took ' + str(indexTime) + " seconds\n")

        f.write("\n" + "'" * 11 + "\n")
        f.write(f'{blockNumber} temporary blocks\n')
        f.write(f'Block merge took {mergeTime} seconds\n')
        index_on_disk = os.path.getsize('index.txt')
        f.write("Written final index in " + str(indexWritten) + " seconds\n")
        f.write(f'Final index: {total_terms} terms. Total Time: {total_time}\n')

        f.write(f'Search for "{search[0]}": ({search[2]} seconds) {len(search[1])} reviews\n')
        f.write(', '.join(search[1][:10]) + "...\n")
        f.write("Index Size on Disk " + str(index_on_disk) + " bytes\n")
        f.write("-" * 66)



if __name__ == "__main__":
    main()
