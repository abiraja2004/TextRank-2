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
    """
    # Remove nodes for which no similar/common words/tokens are found
    if similar_tokens != 0:
        similarity = similar_tokens / float(math.log(string_length_1) + math.log(string_length_2))
    else:
	graph.remove_node(string_1)
	graph.remove_node(string_2)
   	similarity = 0 
    return similarity,graph
    """
  
def CosineSimilarity(string_vec_1, string_vec_2):
    pass

def computeTfIdf(nodes_list):
    pass


# Graph generation
def generateGraph(nodes_list, distanceMetric):
 
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
   
    elif distanceMetric == 'CS':
	computeTfIdf(nodes_list)
	for node in nodePairs:
	    string_1 = node[0]
	    string_2 = node[1]
            #add edges to graph (weughteed by cosine similarity)
	    cosineScore = CosineSimilarity(string_1, string_2)
	    gr.add_edge(string_1,string_2,weight=cosineScore,time=str(cosineScore))
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
def drawGraph(graph):
	try:
	    #nx.draw_networkx_edge_labels(graph,pos,edge_labels=edge_weight)
	    nx.draw_circular(graph)
	    plt.show()
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
    
    if draw != 'F':
        if drawGraph(graph):
            pass
	else:
	    print "Failed to draw\n"
	    sys.exit()  
    else:
        pass

    calculated_page_rank = nx.pagerank(graph, weight='weight')
    #return calculated_page_rank
    #most important sentences in ascending order of importance
    sentences = sorted(calculated_page_rank, key=calculated_page_rank.get, reverse=True)

    #return sentences
    number_sent = int(sent)
    
    summary = '. '.join([s for s in sentences[:number_sent]])

    return summary


def readCommandLineArguments():
    parser = argparse.ArgumentParser(description='Auto Summarization of text')
    parser.add_argument('-s', action="store", dest="sent")
    parser.add_argument('-d', action="store", dest="dist")
    parser.add_argument('-draw', action="store", dest="draw")
    val = parser.parse_args()
    
    return val.sent, val.dist, val.draw


if __name__=="__main__":
    
    #`Example 1`
    #text = requests.get('http://rare-technologies.com/the_matrix_synopsis.txt').text
    
    text = open('data.txt','r').read()
    sent, dist ,draw = readCommandLineArguments()
    summary = extractSummary(text.decode('utf-8'),sent, dist,draw)
    print summary
