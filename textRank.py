#Implementaion of : https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf
#Based on: https://github.com/davidadamojr/TextRank with some additions
#Text mentioned as example from: http://rare-technologies.com/text-summarization-with-gensim/
"""
@uthor: Prakhar Mishra
"""

# Importing Libraries
import io
import sys
import requests
import nltk
import networkx as nx
import os
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from string import punctuation
import argparse
import itertools
import math
import matplotlib.pyplot as plt
from scipy import spatial
import gensim
import numpy as np
from slidingWindow import SlidingWindow
np.set_printoptions(threshold=np.nan)

# Distance Metric 1
def LavenDistance(firstString, secondString):
    if len(firstString) > len(secondString):
        firstString, secondString = secondString, firstString
    distances = range(len(firstString) + 1)
    for index2, char2 in enumerate(secondString):
        newDistances = [index2 + 1]
        for index1, char1 in enumerate(firstString):
            if char1 == char2:
                newDistances.append(distances[index1])
            else:
                newDistances.append(1 + min((distances[index1], distances[index1+1], newDistances[-1])))
        distances = newDistances
    return distances[-1]


# Distance Metric 2
def CommonTokens(string_1, string_2, graph):
    string_1_ = word_tokenize(string_1)
    string_2_ = word_tokenize(string_2)
    
    string_length_1 = len(string_1_)
    string_length_2 = len(string_2_)
    
    # +1 to avoid no common token problem 
    similar_tokens = len(list(set(string_1_).intersection(string_2_)))
   
    similarity =  similar_tokens / float(math.log(string_length_1) + math.log(string_length_2))

    return similarity
 

def getVector(string_1, string_2, model):
    #return string_1, string_2
    vec1 = model[string_1][0]
    vec2 = model[string_2][0]
    return vec1, vec2

def CosineSimilarity(string_1, string_2, model):
    string_1_vec, string_2_vec = getVector(string_1, string_2, model)
    #return [string_1_vec, string_2_vec]
    r = 1 - spatial.distance.cosine(string_1_vec, string_2_vec)
    return r    


def computeTfIdf(nodes_list):
    pass


# Graph generation
def generateGraph(nodes_list, distanceMetric, model):

    
    # Initialize of an graph
    gr = nx.Graph() 
    gr.add_nodes_from(nodes_list)
    
    # Fully connected graph construction
    nodePairs = list(itertools.combinations(nodes_list, 2))

    if distanceMetric == 'LD':
        for node in nodePairs:
            string_1 = node[0]
            string_2 = node[1]
            #add edges to the graph (weighted by Levenshtein distance)
            levDistance = LavenDistance(string_1, string_2)
            gr.add_edge(string_1, string_2, weight=levDistance)
        return gr

    elif distanceMetric == 'CT':
        for node in nodePairs:
            string_1 = node[0]
            string_2 = node[1]
            #add edges to the graph (weighted by Common Token count)
            commonTokenCount = CommonTokens(string_1, string_2, gr)  
	    gr.add_edge(string_1, string_2, weight=commonTokenCount)
        return gr
   
    elif distanceMetric == 'w2vCS':
        for node in nodePairs:
	    string_1 = node[0]
	    string_2 = node[1]
	    cosineSimilarity = CosineSimilarity(string_1, string_2, model)
	    #return cosineSimilarity
	    gr.add_edge(string_1, string_2, weight=cosineSimilarity)
	
	return gr


# Pre-processing function
stopModList = set(stopwords.words('english'))
def cleanText(text):
    s_f = []
    
    # For every sentence in sentenceTokens
    for s in text:
        stopRemoved = ' '.join([w for w in s.split() if w not in stopModList])
        puncRemoved = ''.join([w for w in stopRemoved if w not in punctuation])
        s_f.append(puncRemoved)
    return s_f
    

# Draw the graph
def drawGraph(G):
	try:
	    pos = nx.spring_layout(G)
	    #nx.draw_circular(G, with_labels = True)
	    #elarge=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] > 2]
	    #esmall=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] <= 2]
	    edge_weight=dict([((u,v,),int(d['weight'])) for u,v,d in G.edges(data=True)])
	    #nx.draw_networkx_edges(G,pos,edgelist=elarge, width=6)
	    #nx.draw_networkx_edges(G,pos,edgelist=esmall, width=6,alpha=0.2,style='dashed')

	    nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_weight)
	    nx.draw_networkx_nodes(G,pos)
	    nx.draw_networkx_edges(G,pos)
	    nx.draw_networkx_labels(G,pos)
	    plt.axis('off')
	    plt.show()

	    #nx.draw_networkx_edge_labels(graph,pos,edge_labels=edge_weight)
	    #nx.draw_circular(graph)
	    #plt.show()
	except Exception as e:
		return 0
	return 1


# Main function
def extractSummary(text, sent, dist, draw='F'):
    # Tokenize on the sentence
    sentenceTokens= sent_tokenize(text)
    
    # Clean the text and remove stop words
    s_f = cleanText(sentenceTokens)

    graph = generateGraph(s_f,dist)
    return graph 
    if draw != 'F':
        if drawGraph(graph):
            pass
	else:
	    print "Failed to draw\n"
	    sys.exit()  
    else:
        pass

    calculated_page_rank = nx.pagerank(graph, max_iter=100,weight='weight')
    #return calculated_page_rank
    #most important sentences in ascending order of importance
    sentences = sorted(calculated_page_rank, key=calculated_page_rank.get, reverse=True)

    #return sentences
    number_sent = int(sent)
    
    summary = '. '.join([s for s in sentences[:number_sent]])

    return summary


def removeDuplicates(taggedTokens):
    L = []
    for i in taggedTokens:
        if i not in L:
	    L.append(i)
	else:
            pass

    return L


def generateCoocurrenceMatrix(taggedTokens,windowLength):
    i = 0
    matrix_row = matrix_col = len(taggedTokens)
    w = []
    # First considering a fully connected graph
    # Filling diagonal entries to be `ZERO` {ignoring self loop}
    mat = np.zeros((matrix_row, matrix_col), int)

    # Matrix of `matrix_row` X `matrix_col` will be made
    # Logic to fill in the cell with co-occurence distance
    sliding_w = SlidingWindow(windowLength, taggedTokens)	

    for i in range(len(taggedTokens)):
        prev, curr, nxt = sliding_w[i][0], sliding_w[i][1], sliding_w[i][2]
	for _i in prev:
	    w.append(_i)
	w.append(curr)
	for i_ in nxt:
	    w.append(i_)
	
	for j in range(len(taggedTokens)):
	    if i==j:
	        mat[i][j] = 0
            else:
	        if taggedTokens[j] in w:
		    distance = abs(w.index(taggedTokens[j]) - w.index(taggedTokens[i]))
		    mat[i][j] = distance
        w = []
    return mat


def makeTokenGraph(nodePairsAndWeight):
    gr = nx.Graph()
    for i in nodePairsAndWeight:
        gr.add_node(i[0])
	gr.add_node(i[1])
	gr.add_edge(i[0],i[1])
	gr[i[0]][i[1]]['weight'] = i[2]

    return gr



def postProcessingKeywords(words,wordTokens):
    j = 1
    kPhrase = ''
    Phrases = []
    
    while j < len(wordTokens):
        fWord = wordTokens[j-1]
	sWord = wordTokens[j]
	
	if fWord in words and sWord in words:
	    kPhrase = fWord + ' '  + sWord     
            Phrases.append(kPhrase)
	    words.remove(fWord)
	    words.remove(sWord)
	j += 1

    for i in words:
        Phrases.append(i)
    return Phrases

def extractKeywords(text,dist,w, draw, window=11):
    
    text1 = text.lower()

    # Clean the input 
    # `[text]` because `cleanText` accepts that format
    # Also tokenize it 
    text = cleanText([text1])
    wordTokens = nltk.word_tokenize(text[0]) 
    
    # Do POS Tagging and extract Nouns and Adjectives
    taggedTokens = nltk.pos_tag(wordTokens)
    taggedTokens = [i[0] for i in taggedTokens if i[1] in ["NN","JJ","NNP","NNS","JJS","JJR","NNPS"]]
     
    #return taggedTokens   
    # Remove the duplicate ones | Not required here
    uniqueTokens = removeDuplicates(taggedTokens)
  
    #return uniqueTokens 
    # Window size over which co-occurence matrix will be generated
    # Sliding Window of length of 5 means [w1,w2,w3,w4,w5,W,w6,w7,w8,w9,w10]
    windowSize = window
    mat = generateCoocurrenceMatrix(uniqueTokens,windowSize) 
    #return mat
    # Node pairs that will make the graph
    n1,n2 = np.nonzero(mat)[0],np.nonzero(mat)[1]
    nodePairs = [(node1,node2) for node1,node2 in zip(n1,n2)]
    
    # Weight for all the connected node pairs
    weightList = []
    for n in nodePairs:
        w = mat[n[0]][n[1]]
	weightList.append(w)
    
    # Nodes in graph(pair) and their connected weight
    nodePairsAndWeight = [(node1,node2,weight) for node1,node2,weight in zip(n1,n2,weightList)] 
    #return nodePairsAndWeight
    print "Generating graph...\n"
    graph = makeTokenGraph(nodePairsAndWeight)
   
    if draw != 'F':
        if drawGraph(graph):
	    pass
	else:
            print 'Failed to draw\n'
	    sys.exit()
    else:
	    pass
    

    # Apply PageRank
    calculated_page_rank = nx.pagerank(graph, weight='weight')
    #return calculated_page_rank
    #return uniqueTokens
    
     
    words = sorted(calculated_page_rank, key=calculated_page_rank.get, reverse=True)
    #return words
    #return words
    # Joining the keywords : if keywords selected occcur adjacent to eachother in the original text
    
    words_w = [uniqueTokens[i] for i in words[:30]]
    text1 = "".join([i for i in text1 if i not in punctuation])

    words_w_f = postProcessingKeywords(words_w,text1.split())
    return words_w_f
    #return [uniqueTokens[i] for i in words[:10]]
    #number_words = int(w)
    
    #return words[:number_words]

def readCommandLineArguments():
    parser = argparse.ArgumentParser(description='Auto Summarization of text')
    parser.add_argument('-s', action="store", dest="sent")
    parser.add_argument('-d', action="store", dest="dist")
    parser.add_argument('-draw', action="store", dest="draw")
    val = parser.parse_args()
    
    return val.sent, val.dist, val.draw


if __name__=="__main__":
    
    #`Example 1`
    #text = requests.get('http://rare-technologies.com/the_big_lebowski_synopsis.txt').text
    #print text
    text = open('data.txt','r').read()
    
    sent, dist ,draw = readCommandLineArguments()
    keywords = extractKeywords(text.decode('utf-8'),dist,sent, draw)
    #summary = extractSummary(text.decode('utf-8'),sent, dist,draw)
    print keywords
