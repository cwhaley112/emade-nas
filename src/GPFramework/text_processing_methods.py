import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer 
from GPFramework.data import EmadeDataPair, EmadeData
import copy as cp
import pdb
import pickle
from sklearn.feature_extraction import _stop_words
from tensorflow.keras.preprocessing.text import Tokenizer, one_hot, text_to_word_sequence
from tensorflow.keras.preprocessing import sequence
#import gensim
import nltk
nltk.download('punkt')
from nltk import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer as wnl, PorterStemmer as ps, SnowballStemmer as ss, LancasterStemmer as ls
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from textblob import TextBlob
import re
#import spacy
import os
#try: 
    #nlp = spacy.load("en_core_web_sm")
#except:
    #os.system('python -m spacy download en_core_web_sm')
    #nlp = spacy.load("en_core_web_sm")

"""
Programmed by Jason Zutty
Modified by VIP Team
Implements a number of machine learning methods for use with deap
"""
  
def tokenizer(data_pair, max_length, num_words):
    """Uses keras tokenizer to make the text into sequences of numbers which are mapped to the words

    Args:
        data_pair: given dataset
        binary: True if you want to (use 1 or 0) represent if a word exists in the dataset and false if you want to put the number of times a word exists at the spot instead of 1
        Ngram_start: the lower end of the ngram range
        Ngram_end: the higher end of the ngram range, both ngram_start and ngram_end are used to represent the ngram_range
        whichstopword: represents which stop word list to use 

    Returns:
       the data_pair where the train and test data are tokenized
    """
    train_data = data_pair.get_train_data().get_numpy().flatten()
    test_data = data_pair.get_test_data().get_numpy().flatten()

    #max_length = max_length % 600 + 100 #100
    #num_words = num_words % 2000 + 5000 #1000
    tokenizer = Tokenizer(num_words = num_words)
    tokenizer.fit_on_texts(train_data)
    x_train = tokenizer.texts_to_sequences(train_data)
    x_test = tokenizer.texts_to_sequences(test_data)

    x_train = sequence.pad_sequences(x_train, maxlen = max_length)
    x_test = sequence.pad_sequences(x_test, maxlen = max_length)
    
    data_list = []

    for transformed, dataset in zip([x_train, x_test], [data_pair.get_train_data(), data_pair.get_test_data()]):
        instances = cp.deepcopy(dataset.get_instances())
        for i, instance in enumerate(instances):
            instance.get_features().set_data([transformed[i]])
        new_dataset = EmadeData(instances)
        data_list.append(new_dataset)
    data_pair = EmadeDataPair(train_data=(data_list[0], None),
                                       test_data=(data_list[1], None))
    vocabsize = np.amax(x_train) +1
    return data_pair, vocabsize, tokenizer


stop_words1 = ["in", 'of', 'at', 'a', 'the', 'an']
stop_words2 = "english"
stop_words3 = None
stop_words4 = _stop_words.ENGLISH_STOP_WORDS
stop_words5 = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']

list = [stop_words1, stop_words2, stop_words3, stop_words4, stop_words5]
MAX_NGRAM_VALUE = 3

def count_vectorizer(data_pair, binary, ngram_start, ngram_end, whichStopWordList):
    """Vectorize text data using traditional bag of words techniques

    Args:
        data_pair: given dataset
        binary: True if you want to (use 1 or 0) represent if a word exists in the dataset and false if you want to put the number of times a word exists at the spot instead of 1
        ngram_start: the lower end of the ngram range
        ngram_end: the higher end of the ngram range, both ngram_start and ngram_end are used to represent the ngram_range
        whichStopWordList: represents which stop word list to use

    Returns:
        the data_pair where the train and test data are vectorized
    """
    #STEMMING code for future reference!!
    # stemmer = PorterStemmer()
    # analyzer = CountVectorizer().build_analyzer()()()

    # def stemming(doc):
    #     return (stemmer.stem(w) for w in analyzer(doc))

    ngram_start = ngram_start % MAX_NGRAM_VALUE + 1
    ngram_end = ngram_end % MAX_NGRAM_VALUE + 1
    if (ngram_start > ngram_end):
        ngram_start, ngram_end = ngram_end, ngram_start
    whichStopWordList =  whichStopWordList % len(list)

    train_data = data_pair.get_train_data().get_numpy().flatten()
    test_data = data_pair.get_test_data().get_numpy().flatten()
    vectorizer = CountVectorizer(stop_words = list[whichStopWordList], binary = binary, ngram_range = (ngram_start, ngram_end))
    x_train = vectorizer.fit_transform(train_data)
    x_test = vectorizer.transform(test_data)

    data_list = []
    #print(x_train)
    for transformed, dataset in zip([x_train, x_test], [data_pair.get_train_data(), data_pair.get_test_data()]):
        instances = cp.deepcopy(dataset.get_instances())
        for i, instance in enumerate(instances):
            instance.get_features().set_data(transformed[i])
        new_dataset = EmadeData(instances)
        data_list.append(new_dataset)
    data_pair = EmadeDataPair(train_data=(data_list[0], None),
                                       test_data=(data_list[1], None))


    #print(type(x_test))

    #print(data_pair.get_train_data().get_instances()[1].get_features())


   #data_pair = GTMOEPDataPair(train_data = GTMOEPData(x_train), test_data = GTMOEPData(x_test))
    #print(data_pair.get_train_data().get_numpy())
    #gc.collect(); 
    return data_pair

def tfidf_vectorizer(data_pair, binary, ngram_start, ngram_end, whichStopWordList):
    """Vectorize text data using TFIDF bag of words techniques

    Args:
        data_pair: given dataset
        binary: True if you want to (use 1 or 0) represent if a word exists in the dataset and false if you want to put the number of times a word exists at the spot instead of 1
        ngram_start: the lower end of the ngram range
        ngram_end: the higher end of the ngram range, both ngram_start and ngram_end are used to represent the ngram_range
        whichStopWordList: represents which stop word list to use 

    Returns:
        the data_pair where the train and test data are vectorized
    """
    ngram_start = ngram_start % MAX_NGRAM_VALUE + 1
    ngram_end = ngram_end % MAX_NGRAM_VALUE + 1
    if (ngram_start > ngram_end):
        ngram_start, ngram_end = ngram_end, ngram_start
    whichStopWordList =  whichStopWordList % len(list)

    train_data = data_pair.get_train_data().get_numpy().flatten()
    test_data = data_pair.get_test_data().get_numpy().flatten()

    vectorizer = TfidfVectorizer(stop_words = list[whichStopWordList], binary = binary, ngram_range = (ngram_start, ngram_end))
    x_train = vectorizer.fit_transform(train_data)
    x_test = vectorizer.transform(test_data)
    print(x_train)
    data_list = []
    for transformed, dataset in zip([x_train, x_test], [data_pair.get_train_data(), data_pair.get_test_data()]):
        instances = cp.deepcopy(dataset.get_instances())
        for i, instance in enumerate(instances):
            instance.get_features().set_data(transformed[i])
        new_dataset = EmadeData(instances)
        data_list.append(new_dataset)
    data_pair = EmadeDataPair(train_data=(data_list[0], None),
                                       test_data=(data_list[1], None))
    return data_pair

def count_vectorizer_mod(data_pair, binary, ngram_start, ngram_end, whichStopWordList):
    """Vectorize text data using TFIDF bag of words techniques

    Args:
        data_pair: given dataset
        binary: True if you want to (use 1 or 0) represent if a word exists in the dataset and false if you want to put the number of times a word exists at the spot instead of 1
        ngram_start: the lower end of the ngram range
        ngram_end: the higher end of the ngram range, both ngram_start and ngram_end are used to represent the ngram_range
        whichStopWordList: represents which stop word list to use 

    Returns:
        the data_pair where the train and test data are vectorized
    """
    ngram_start = ngram_start % MAX_NGRAM_VALUE + 1
    ngram_end = ngram_end % MAX_NGRAM_VALUE + 1
    if (ngram_start > ngram_end):
        ngram_start, ngram_end = ngram_end, ngram_start
    whichStopWordList =  whichStopWordList % len(list)

    train_data = data_pair.get_train_data().get_numpy()
    test_data = data_pair.get_test_data().get_numpy()
    x_train = []
    x_test = []
    for i in range(train_data.shape[1]):
        vectorizer = CountVectorizer(stop_words = list[whichStopWordList], binary = binary, ngram_range = (ngram_start, ngram_end))
        train = vectorizer.fit_transform(train_data[:,i])
        test = vectorizer.transform(test_data[:,i])
        x_train.append(train)
        x_test.append(test)
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    data_list = []
    for transformed, dataset in zip([x_train, x_test], [data_pair.get_train_data(), data_pair.get_test_data()]):
        instances = cp.deepcopy(dataset.get_instances())
        for i, instance in enumerate(instances):
            instance.get_features().set_data([transformed[i]])
        new_dataset = EmadeData(instances)
        data_list.append(new_dataset)
    data_pair = EmadeDataPair(train_data=(data_list[0], None),
                                       test_data=(data_list[1], None))
    return data_pair


def hashing_vectorizer(data_pair, binary, ngram_start, ngram_end, whichStopWordList):
    """Vectorize text data using bag of words techniques but by hashing the words to make it efficient

    Args:
        data_pair: given dataset
        binary: True if you want to (use 1 or 0) represent if a word exists in the dataset and false if you want to put the number of times a word exists at the spot instead of 1
        ngram_start: the lower end of the ngram range
        ngram_end: the higher end of the ngram range, both ngram_start and ngram_end are used to represent the ngram_range
        whichStopWordList: represents which stop word list to use 

    Returns:
        the data_pair where the train and test data are vectorized
    """
    
    ngram_start = ngram_start % MAX_NGRAM_VALUE + 1
    ngram_end = ngram_end % MAX_NGRAM_VALUE + 1
    if (ngram_start > ngram_end):
        ngram_start, ngram_end = ngram_end, ngram_start
    whichStopWordList =  whichStopWordList % len(list)

    train_data = data_pair.get_train_data().get_numpy().flatten()
    test_data = data_pair.get_test_data().get_numpy().flatten()

    vectorizer = HashingVectorizer(stop_words = list[whichStopWordList], binary = binary, ngram_range = (ngram_start, ngram_end))
    x_train = vectorizer.fit_transform(train_data)
    x_test = vectorizer.transform(test_data)
    data_list = []
    for transformed, dataset in zip([x_train, x_test], [data_pair.get_train_data(), data_pair.get_test_data()]):
        instances = cp.deepcopy(dataset.get_instances())
        for i, instance in enumerate(instances):
            instance.get_features().set_data(transformed[i])
        new_dataset = EmadeData(instances)
        data_list.append(new_dataset)
    data_pair = EmadeDataPair(train_data=(data_list[0], None),
                                       test_data=(data_list[1], None))
    return datapair



def word2vec(data_pair, whichStopWordList, size, window, min_count , sg):
     """Uses gensim's word2vec vectorizer to make a word2vec model which creates vectors associated with each word. Afterward, each row of text becomes an average of each of the vectors associated with its' words

     Args:
         data_pair: given dataset
         whichstopword: represents which stop word list to use 
         size: corresponds with the dimension of the word vector
         window: the minimum distance between the vectors, smaller window should give terms that are more related bigger window increases accuracy but takes much longer
         min_count: word has to appear this many times to affect the model
         sg: true if skip-gram technique is going to be used false if continuous bag of words technique is going to be used to make the word2vec model

     Returns:
        the data_pair where the train and test data are vectorized with averaged word2vec vectors
     """
     size = size % 500 + 5 #10 #this corresponds with number of layers in the word
     window = window % 500 + 1 #minimum distance between vectors, smaller window should give your terms that are more related bigger value increasing accuracy but takes much longer
     workers = 8 #workers corresponds with number of cores you have more workers better should probably keep this constant
     min_count = min_count % 3 + 1 #word has to appear this many times to affect the model
     technique = -1
     if sg == True:
         technique = 1
     else:
         technique = 0
     sg = sg % 2 #use cbow or skip-gram method, skip-gram method is better but much slower, and parameter range is different for sg
     train_data = data_pair.get_train_data().get_numpy().flatten()
     test_data = data_pair.get_train_data().get_numpy().flatten()
     stop_words = list[whichStopWordList % len(list)]
    
     def tokenize(val):
         ans = text_to_word_sequence(val)
         ans = [a for a in ans if not a in stop_words]
         return ans
     reviews = np.concatenate((train_data, test_data))
     words = [tokenize(val) for val in reviews.tolist()]

     model = gensim.models.Word2Vec(sentences=words, size= size, window = window, workers = workers, min_count=min_count, sg = technique)

     def method(list, wv):
         mean = []
         for word in list:
             if word in wv.vocab:
                 mean.append(wv[word])
             else:
                 mean.append(word)
         mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
         return mean
     x_train = words[:train_data.shape[0]]
     x_train = np.array([method(review, model.wv) for review in x_train])
     x_test = words[test_data.shape[0]:]
     x_test = np.array([method(review, model.wv) for review in x_test])

     data_list = []
     for transformed, dataset in zip([x_train, x_test], [data_pair.get_train_data(), data_pair.get_test_data()]):
         instances = cp.deepcopy(dataset.get_instances())
         i = 0;
         for instance in instances:
             instance.get_features().set_data(np.array([transformed[i]]))
             i+=1
         new_dataset = EmadeData(instances)
         data_list.append(new_dataset)
     data_pair = EmadeDataPair(train_data=(data_list[0], None),
                                        test_data=(data_list[1], None))
     return data_pair

def sentiment(data_pair, sentence_vec):

    def document_sentiment(data, sentence_vec=False):

        sentiments = []

        for review in data:

            review_sentiments = []

            if sentence_vec:
                sentences = nltk.sent_tokenize(review)
                for sentence in sentences:
                    blob = TextBlob(sentence)
                    ps_list = np.array ( [blob.sentiment.polarity, blob.sentiment.subjectivity] ) # Polarity and subjectivity list
                    review_sentiments.append(ps_list)
            else:
                blob = TextBlob(review)
                for word in blob.words:
                    word = re.sub('[^A-Za-z0-9]+', '', str(word)) # remove punc.
                    wordBlob = TextBlob(word)
                    ps_list = np.array ( [wordBlob.sentiment.polarity, wordBlob.sentiment.subjectivity] )
                    review_sentiments.append(ps_list)
            # print(np.mean(review_sentiments, axis=0))
            # print(review_sentiments)
            sentiments.append(np.mean(review_sentiments, axis=0))

        return np.array(sentiments)

    train_data = data_pair.get_train_data().get_numpy().flatten()
    test_data = data_pair.get_test_data().get_numpy().flatten()
    first_review = train_data[0]
    # Convert to sentiment
    x_train = document_sentiment(train_data, sentence_vec)
    x_test = document_sentiment(test_data, sentence_vec)
    
    # for i, x in enumerate(x_train):
    #     print(f"xtrain[{i}]: {x}")

    # for i, x in enumerate(x_test):
    #     print(f"xtest[{i}]: {x}")
    # print(first_review)

    data_list = []

    for transformed, dataset in zip([x_train, x_test], [data_pair.get_train_data(), data_pair.get_test_data()]):
        instances = cp.deepcopy(dataset.get_instances())
        for i, instance in enumerate(instances):
            instance.get_features().set_data([transformed[i]])
        new_dataset = EmadeData(instances)
        data_list.append(new_dataset)
    data_pair = EmadeDataPair(train_data=(data_list[0], None),
                                       test_data=(data_list[1], None))
    return data_pair


def spacy_pos_tagger(word):
    wordObject = spacy_english_model(word)
    return wordObject[0].pos_
def nltk_pos_tagger(word):
    return nltk.pos_tag([word])[0][1][0].upper()
def nltk_porter_stemmer(word, pos_tagger):
    return ps().stem(word)
def nltk_snowball_stemmer(word, pos_tagger):
    return ss("english").stem(word)
def nltk_lancaster_stemmer(word, pos_tagger):
    return ls().stem(word)
def nltk_lemmatizer(word, pos_tagger):
    pos = pos_tagger(word)
    tag_dict = {"J": wordnet.ADJ,   #nltk tagger
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV,
                "ADJ": wordnet.ADJ, #spacy tagger
                "NOUN": wordnet.NOUN,
                "VERB": wordnet.VERB,
                "ADV": wordnet.ADV}
    pos = tag_dict.get(pos, None)
    return wnl().lemmatize(word, pos=pos) if pos != None else wnl().lemmatize(word)
def stemlemmatize(string, func, pos_tagger):
    transformed = []
    for w in word_tokenize(string):
        transformed.append(func(w, pos_tagger))
        transformed.append(" ")
    return "".join(transformed)
funcs = [nltk_porter_stemmer, nltk_snowball_stemmer, nltk_lemmatizer]
pos_taggers = [spacy_pos_tagger, nltk_pos_tagger]
pos_taggers = [nltk_pos_tagger]
def stemmatizer(data_pair, func_index=None, pos_tagger_index=None):
    """Stemming and lemmatization primitive
    Args:
        data_pair: given dataset
        func: func
        lemma: Input lemmatizer if given, won't use otherwise
    Returns:
        the data_pair after being transformed
    """
    if func_index == None or pos_tagger_index == None:
        return data_pair
    train_data = data_pair.get_train_data().get_numpy().flatten()
    test_data = data_pair.get_test_data().get_numpy().flatten()

    #stem/lemmatize here
    func = funcs[func_index % len(funcs)]
    pos_tagger = pos_taggers[pos_tagger_index % len(pos_taggers)]

    lam = lambda one_review, func, pos_tagger: stemlemmatize(one_review, func, pos_tagger)
    x_train = np.array([lam(r, func, pos_tagger) for r in train_data])
    x_test = np.array([lam(r, func, pos_tagger) for r in test_data])

    #repackage stemmatized data into data_pair and return.
    data_list = []
    for transformed, dataset in zip([x_train, x_test], [data_pair.get_train_data(), data_pair.get_test_data()]):
        instances = cp.deepcopy(dataset.get_instances())
        for i, instance in enumerate(instances):
            instance.get_features().set_data(transformed[i])
        new_dataset = EmadeData(instances)
        data_list.append(new_dataset)
    data_pair = EmadeDataPair(train_data=(data_list[0], None),
                                       test_data=(data_list[1], None))
    return data_pair

def num_named_entities(data_pair):
    train_data = data_pair.get_train_data().get_numpy()
    test_data = data_pair.get_train_data().get_numpy()
    to_num_named_entities = lambda sent: len(nlp(str(sent)).ents)
    not_normal_train = np.vectorize(to_num_named_entities)(train_data)
    not_normal_test = np.vectorize(to_num_named_entities)(test_data)
    # Normalize train and test by dividing by total number of named entities in row
    normalized_train = not_normal_train / not_normal_train.sum(axis=1)[:,None]
    normalized_test = not_normal_test / not_normal_test.sum(axis=1)[:,None]
    x_test = np.asarray(normalized_train, dtype = np.float32)
    x_train = np.asarray(normalized_test, dtype = np.float32)
    data_list = []
    for transformed, dataset in zip([x_train, x_test], [data_pair.get_train_data(), data_pair.get_test_data()]):
        instances = cp.deepcopy(dataset.get_instances())
        for i, instance in enumerate(instances):
            instance.get_features().set_data(transformed[i])
        new_dataset = EmadeData(instances)
        data_list.append(new_dataset)
    data_pair = EmadeDataPair(train_data=(data_list[0], None),
                                       test_data=(data_list[1], None))
    
    return data_pair
  
def num_named_entities(data_pair):
    train_data = data_pair.get_train_data().get_numpy()
    test_data = data_pair.get_train_data().get_numpy()
    # function to convert a sentence to the number of named entities
    to_num_named_entities = lambda sent: len(nlp(str(sent)).ents)
    # create non-normalized versions of train and test_data using # entity function

    #pdb.set_trace()
    not_normal_train = np.vectorize(to_num_named_entities)(train_data)

    not_normal_test = np.vectorize(to_num_named_entities)(test_data)
    # Normalize train and test by dividing by total number of named entities in row
    normalized_train = not_normal_train / not_normal_train.sum(axis=1)[:,None]
    normalized_test = not_normal_test / not_normal_test.sum(axis=1)[:,None]
    x_test = np.asarray(normalized_train, dtype = np.float32)
    x_train = np.asarray(normalized_test, dtype = np.float32)
    data_list = []

    for transformed, dataset in zip([x_train, x_test], [data_pair.get_train_data(), data_pair.get_test_data()]):
        instances = cp.deepcopy(dataset.get_instances())
        for i, instance in enumerate(instances):
            instance.get_features().set_data([transformed[i]])
        new_dataset = EmadeData(instances)
        data_list.append(new_dataset)
    data_pair = EmadeDataPair(train_data=(data_list[0], None),
                                       test_data=(data_list[1], None))
    return data_pair


def textRank(data_pair):
    """Computes TextRank for given dataset. UNFINISHED
    Args:
        data_pair: given dataset
    Returns:
        the data_pair where the train and test data are vectorized
    """
    from nltk.corpus import stopwords
    
    train_data = data_pair.get_train_data().get_numpy()
    test_data = data_pair.get_test_data().get_numpy()
    
    #This code doesn't make any sense!!
    # Extract word vectors
    #word_embeddings = {}
    #with open('/nv/pace-ice/agurung7/vip/emade/glove_word_embeddings.pickle', 'rb') as f:
    #    word_embeddings = pickle.load(f)

    # remove punctuations, numbers and special characters
    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")

    # make alphabets lowercase
    clean_sentences = [s.lower() for s in clean_sentences]

    # get stopwords from nltk
    stop_words = stopwords.words('english')

    # function to remove stopwords
    remove_stopwords = lambda sen: " ".join([i for i in sen if i not in stop_words])
        
    data_list = []
    #print(x_train)
    for transformed, dataset in zip([x_train, x_test], [data_pair.get_train_data(), data_pair.get_test_data()]):
        instances = cp.deepcopy(dataset.get_instances())
        for i, instance in enumerate(instances):
            instance.get_features().set_data(transformed[i])
        new_dataset = EmadeData(instances)
        data_list.append(new_dataset)
    data_pair = EmadeDataPair(train_data=(data_list[0], None),
                                       test_data=(data_list[1], None))
    return data_pair


def tfisf(data_pair, binary, ngram_start, ngram_end, whichStopWordList):
    """Vectorize text data using TFIDF bag of words techniques
    Args:
        data_pair: given dataset
        binary: True if you want to (use 1 or 0) represent if a word exists in the dataset and false if you want to put the number of times a word exists at the spot instead of 1
        ngram_start: the lower end of the ngram range
        ngram_end: the higher end of the ngram range, both ngram_start and ngram_end are used to represent the ngram_range
        whichStopWordList: represents which stop word list to use 
    Returns:
        the data_pair where the train and test data are vectorized
    """
    ngram_start = ngram_start % MAX_NGRAM_VALUE + 1
    ngram_end = ngram_end % MAX_NGRAM_VALUE + 1
    if (ngram_start > ngram_end):
        ngram_start, ngram_end = ngram_end, ngram_start
    whichStopWordList =  whichStopWordList % len(list)
    train_data = data_pair.get_train_data().get_numpy()
    test_data = data_pair.get_test_data().get_numpy()
    ftrain = train_data[:,0].copy()
    ftest = test_data[:,0].copy()
    for i in range(1,train_data.shape[1]):
        ftrain+=' ' + train_data[:,1].copy()
        ftest+=' ' + test_data[:,1].copy()
    vectorizer = TfidfVectorizer(stop_words = list[whichStopWordList], binary = binary, ngram_range = (ngram_start, ngram_end))
    x_train = vectorizer.fit_transform(ftest)
    x_test = vectorizer.transform(ftrain)
    
    feature_names = vectorizer.get_feature_names()
    print(feature_names)
    x_newtrain = []
    print(x_train[0])
    print(train_data[0])
    for i in range(len(train_data)):
        for j in range(train_data.shape[1]):
            newList = [] ### set of scores per paragraph
            tfidft = 0
            df = pd.DataFrame(x_train[i].T.todense(), index=feature_names, columns=["tfidf"])
            s = df['tfidf']
            length = len(train_data[i,j])
            words = train_data[i,j].split()
            for k, words in enumerate(words): 
                tfidft += s[feature_names[k]]
            newVal = tfidft/length  ##new score
            newList.append(newVal)
        x_newtrain.append(newList)
        
    x_newtest = []
    
    for i in range(len(test_data)):
        for j in range(test_data.shape[1]):
            newList = []
            tfidft = 0
            df = pd.DataFrame(x_train[i].T.todense(), index=feature_names, columns=["tfidf"])
            s = df['tfidf']
            length = len(train_data[i,j])
            words = train_data[i,j].split()
            for k, words in enumerate(words): 
                tfidft += s[feature_names[k]]
            newVal = tfidft/length
            newList.append(newVal)
        x_newtest.append(newList)
        
    x_train = x_newtrain
    x_test = x_newtest
    data_list = []
    for transformed, dataset in zip([x_train, x_test], [data_pair.get_train_data(), data_pair.get_test_data()]):
        instances = cp.deepcopy(dataset.get_instances())
        for i, instance in enumerate(instances):
            instance.get_features().set_data(transformed[i])
        new_dataset = EmadeData(instances)
        data_list.append(new_dataset)
    data_pair = EmadeDataPair(train_data=(data_list[0], None),
                                       test_data=(data_list[1], None))
    return data_pair


