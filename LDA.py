#################################################################################################
# based on http://www.cs.duke.edu/courses/spring14/compsci290/assignments/lab02.html
# nltk punkt package needed (nltk.download('punkt'))
# vocabulary size is susceptible, topic number is fixed at 50, analysis on HCA shows a pretty high entropy on this
# The script requires 2GB memory for training 6000 stories, because Python's LDA api requires whole D-T matrix load,
# this is unacceptable in large scale training. Instead, use the saved ldac format, and train it in hda project.
#################################################################################################
from nltk import word_tokenize         
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import string, re, sys
import numpy as np
# import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import lda

stemmer = PorterStemmer()
vocabuary = None
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed
def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

def valid(word):
    '''
        use global var vocaburay to indicate if we already load the voc from Vocabuary.txt file
    '''
    global vocabuary
    if vocabuary: 
        return word in vocabuary
    else:
        if not re.search('[a-zA-Z]', word):
            return False
        if '.' in word or ':' in word or '\'' in word or '`' in word or '*' in word:
            return False 
        if re.search('^[' + string.punctuation + ']', word):
            return False
        if re.search('[' + string.punctuation + ']$', word):
            return False
        if hasNumbers(word):
            return False
        return True

def mytokenize(text):
    tokens = word_tokenize(text)
    # filtering
    # tokens = [i for i in tokens if not hasPunctuation(i)]
    tokens = [i for i in tokens if valid(i)]
    stems = stem_tokens(tokens, stemmer)
    return stems

def loadFile2Arr(filename):
    file = open(filename, 'r')
    ls = []
    for line in file:
        ls.append(line[:-1])
    return ls
def loadFile2Dict(filename):
    file = open(filename, 'r')
    count = 0
    dic = {}
    for line in file:
        dic[line[:-1]] = count
        count += 1
    return dic

def constructVoc(filename):
    docs = open(filename, 'r')
    vocab = []
    nstories = 0
    doc = ""
    for line in docs:
        
        if "Article Delimiter\\" in line:
            vocab.append(unicode(doc, errors='ignore'))
            # vocab += doc
            doc = ""
            nstories += 1
        else :
            doc += line
    print nstories, '[Debugging]:: stories loaded'
    vect = CountVectorizer(tokenizer=mytokenize, stop_words=loadFile2Arr('stopwords.txt'), dtype=np.int16)   #128 is too small
    vect.fit(vocab)
    print "[Debugging]:: Vocabulary size calculated: ", len(vect.get_feature_names())
    file = open('Vocabulary.txt', 'w+')
    for word in vect.get_feature_names():
        file.write(word.encode('utf-8') + '\n')
    file.close()
    docs.close()
    return vect

def constructDocMat(filename):
    docs = open(filename, 'r')
    doc = ""
    samples = []
    global vocabuary
    vocabuary = loadFile2Dict('Vocabulary.txt')
    print '[Debugging]:: vocabuary size loaded: ', len(vocabuary)
    vect = CountVectorizer(tokenizer=mytokenize, stop_words=loadFile2Arr('stopwords.txt'), vocabulary=vocabuary, dtype=np.int16)   #128 is too small
    for line in docs:
        if "Article Delimiter\\" in line and doc:
            samples.append(unicode(doc, errors='ignore'))
            doc = ""
           
            # if count > 17:
            #     break
        else:
            doc += line
    # Cazily time consuming
    print '[Debugging]:: start transforminng...'

    sparse = vect.transform(samples)
    if sparse.min() < 0:
        raise NameError('negative matrix element: vocabuary fault')
    print '     number of non-zero entry is ', sparse.getnnz()
    print '     sum: ', sparse.toarray().sum()
    # write to csr_matrix format
    print '     save matrix as csr sparse format...'
    np.savetxt('samples_data', sparse.data, fmt="%u")
    np.savetxt('samples_indices', sparse.indices, fmt="%d")
    np.savetxt('samples_indptr', sparse.indptr, fmt="%d")
    # write to ldac format
    print '     save matrix as ldac format...'
    doclines = lda.utils.dtm2ldac(sparse.toarray())
    with open('corpus.ldac', 'w+') as f:
        for line in doclines:
            f.write(line + '\n')
    return sparse
    print '[Debugging]:: finished saving sparse matrix'
def ldatrain():
    '''
        Read csr_matrix from samples*, and train the LDA model
        output: write a bunch of parameters into model folder
    '''
    print '[Test]:: start reading sparse matrix...'
    data = np.loadtxt('samples_data', dtype=np.int32)
    indices = np.loadtxt('samples_indices', dtype=np.int32)
    indptr = np.loadtxt('samples_indptr', dtype=np.int32)
    X = csr_matrix((data, indices, indptr)).toarray()
    # X = X[np.nonzero(X.sum(axis=1)),:]
    # do cross validation stuff down here
    print '[Test]:: I/O finished:'
    print '     document-term matrix size: ', X.shape
    # ntopics = X.shape[0] / 50
    model = lda.LDA(n_topics=50, n_iter=1000)
    print '[Test]:: Training...'
    model.fit(X)
    # Report perplexity
    print '[Test]:: training perplexity: ', model.loglikelihood() / X.sum()
    
    # Report topics
    topic_word = model.topic_word_  # model.components_ also works
    n_top_words = 10
    vocab = loadFile2Arr('Vocabulary.txt')
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))

    # save model to model folder 
    print '[Test]:: save model to folder...'
    np.savetxt('model/components', model.components_)  #phi
    np.savetxt('model/nzw', model.nzw_, fmt="%d") 
    np.savetxt('model/ndz', model.ndz_, fmt="%d")
    np.savetxt('model/doc_topic', model.doc_topic_)
    np.savetxt('model/nz', model.nz_, fmt="%d")

def KL(p, q):
    '''
        Implement a symmetric KL divergence between two discrete distribution
    '''
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)

    return ( _KL(p,q) +  _KL(q,p) ) / 2

def _KL(p, q):

    '''
        parameters
        ----------
        ndarray of 1dim topic distributions

        Returns
        ----------
        KL divergence of these two array
    '''
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

# ########
# Dummy test 
# vocab = ["The swimmer likes swimming so he swims, 1234 2300 and the swimmer doesn't wanna eat shit. Absolutely, andy"]
# vect = CountVectorizer(tokenizer=mytokenize, stop_words='english', dtype=np.int8) 
# vect.fit(vocab)

# sentence1 = vect.transform([unicode('The swimmer likes swimming.', errors='ignore')])
# sentence2 = vect.transform(['The swimmer swims.'])

# print('Vocabulary: %s' %vect.get_feature_names())
# print('Sentence 1: %s' %sentence1.toarray())
# print('Sentence 2: %s' %sentence2.toarray())
if __name__ == "__main__":

    if len(sys.argv) != 2:
        raise NameError('usage: LDA prepare/train')

    if sys.argv[1] == 'prepare':       # preparing vocabuary
        vect = constructVoc("ApiArticle.txt")

    elif sys.argv[1] == 'train':    # train LDA model
        constructDocMat('ApiArticle.txt')
        ldatrain() 
    else:
        print 'usage: prepare/train'





