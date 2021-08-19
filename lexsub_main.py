#!/usr/bin/env python
import sys
import string
from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context 
# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import numpy as np
import tensorflow as tf
import gensim
import transformers 
from collections import defaultdict

from typing import List

def tokenize(s): 
    """
    a naive tokenizer that splits on punctuation and whitespaces.  
    """
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def get_candidates(lemma, pos) -> List[str]:
    # Part 1
    candidates = []
    for w in (wn.synsets(lemma,pos)):
      for g in w.lemmas():
        if g.name() != lemma and g.name() not in candidates:
          val = g.name()
          val = val.replace('_',' ')
          candidates.append(val)
    return candidates

def smurf_predictor(context : Context) -> str:
    """
    suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context : Context) -> str:
    d = defaultdict(int)
    for w in wn.synsets(context.lemma,context.pos):
      for g in w.lemmas():
        val = g.name()
        val = val.replace('_',' ')
        d[val]+=g.count()
    maxi = 0
    max_key = ''
    for key in d.keys():
      if maxi<=d[key] and key!=context.lemma:
        max_key = key
        maxi = d[key]
    return max_key # replace for part 2

def compute_overlap(set_1,set_2):
    count = 0
    for word_1 in set_1:
      for word_2 in set_2:
        if word_1.lower() == word_2.lower():
          count+=1
    return count

def wn_simple_lesk_predictor(context : Context) -> str:

    d = defaultdict(int)
    stop_words = stopwords.words('english')
    max_count = 0
    context_list = (context.left_context + [''] + context.right_context)
    context_sentence = ' '.join(context_list)
    context_list = tokenize(context_sentence)
    lemma_dictionary = defaultdict(int) 
    count_list = []
    synset_list = []
    for w in wn.synsets(context.lemma,context.pos):
      definition = w.definition()
      word_list = []
      for word in tokenize(definition):
        if word not in stop_words:
          word_list.append(word)
      for example in (w.examples()):
        for word in tokenize(example):
          if word not in stop_words:
            word_list.append(word)
      for hypernymn in w.hypernyms():
        for example in hypernymn.examples():
           for word in tokenize(example):
            if word not in stop_words:
              word_list.append(word)
        for word in tokenize(hypernymn.definition()):
          if word not in stop_words:
            word_list.append(word) 
      count = compute_overlap(set(context_list),set(word_list))  
      count_list.append(count)
      synset_list.append(w)
    for i in range(0,len(synset_list)):
      w = synset_list[i]
      lemma_list = w.lemmas()
      context_lemma_count = 0
      for lemma in lemma_list:
        val = lemma.name()
        val = val.replace('_',' ')
        if val == context.lemma.replace('_',' '):
          context_lemma_count = lemma.count()
          break
      for lemma in lemma_list:
        val = lemma.name()
        val = val.replace('_',' ')
        if val != context.lemma:
          lemma_dictionary[val] += 1000*count_list[i] +  100 * context_lemma_count + lemma.count()        
    return max(lemma_dictionary,key=lemma_dictionary.get)
      
      
   #replace for part 3        
   

class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self,context : Context) -> str:
        candidates = []
        nearest_value = ''
        max_similarity_score = 0.0
        candidates = get_candidates(context.lemma,context.pos)
        for c in candidates:
          try:
            score = (self.model.similarity(c,context.lemma))
          except: 
            score = 0.0  
          if score > max_similarity_score:
            max_similarity_score = score
            nearest_value = c

        return nearest_value # replace for part 4


class BertPredictor(object):

    def __init__(self): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
        
    def predict(self, context : Context) -> str:

        candidate_synonyms = get_candidates(context.lemma,context.pos)
        token_array = context.left_context+['[MASK]']+context.right_context
        sentence = ' '.join(token_array)
        mask_index = token_array.index('[MASK]')
        input_tokens = self.tokenizer.encode(token_array)
        token_ids = self.tokenizer.convert_ids_to_tokens(input_tokens)
        input_matrix = np.array(input_tokens).reshape((1,-1))
        output = self.model.predict(input_matrix)
        predictions = output[0]
        best_words_tokens = np.argsort(predictions[0][mask_index+1])[::-1]
        best_words = self.tokenizer.convert_ids_to_tokens(best_words_tokens)
        synonymn_words = []
        for word in best_words:
          if word in candidate_synonyms:
            synonymn_words.append(word)
        return synonymn_words[0] # replace for part 5


###########################.  PART 6 ##################################################

# In this approach, I used the word form of the word to literally give some context to BERT
# It is a known fact that the word synonym cannot have the word itself
# Hence giving BERT some context and then checking for Synonyms would probably give better  accuracy

class BertPredictorCustomized(object):

    def __init__(self): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
        
    def predictCustomized(self, context : Context) -> str:

        candidate_synonyms = get_candidates(context.lemma,context.pos)
        token_array = context.left_context+[context.word_form]+context.right_context
        sentence = ' '.join(token_array)
        mask_index = token_array.index(context.word_form)
        input_tokens = self.tokenizer.encode(token_array)
        token_ids = self.tokenizer.convert_ids_to_tokens(input_tokens)
        input_matrix = np.array(input_tokens).reshape((1,-1))
        output = self.model.predict(input_matrix)
        predictions = output[0]
        best_words_tokens = np.argsort(predictions[0][mask_index+1])[::-1]
        best_words = self.tokenizer.convert_ids_to_tokens(best_words_tokens)
        synonymn_words = []
        for word in best_words:
          if word in candidate_synonyms:
            synonymn_words.append(word)
        return synonymn_words[0]#candidate_synonyms[mask_index+1] # replace for part 6   

if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).
    predictor = BertPredictorCustomized()
    #W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    #predictor = Word2VecSubst(W2VMODEL_FILENAME)
    #predictor = BertPredictor()
    #print(get_candidates('slow','a'))
    count = 0
    for context in read_lexsub_xml(sys.argv[1]):
        #prediction = predictor.predict_nearest(context)
        #prediction = predictor.predict(context)
        prediction = predictor.predictCustomized(context)
        #prediction = wn_frequency_predictor(context)
        #prediction = wn_simple_lesk_predictor(context)
        #prediction = smurf_predictor(context)  
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
        #break
