#Implementaion of : https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf
#Based on: https://github.com/davidadamojr/TextRank with some additions
#Text mentioned as example from: http://rare-technologies.com/text-summarization-with-gensim/
"""
@uthor: Prakhar Mishra
"""

# Importing Libraries
import io
import requests
import nltk
import networkx as nx
import os
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from string import punctuation
import argparse
import itertools

# Distance Metric 1
def lDistance(firstString, secondString):
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


# Graph generation
def generateGraph(nodes_list, distanceMetric):
    
    # Initialize of an graph
    gr = nx.Graph() 
    gr.add_nodes_from(nodes_list)
    
    # Fully connected graph construction
    nodePairs = list(itertools.combinations(nodes_list, 2))

    if distanceMetric == 'L':
        #add edges to the graph (weighted by Levenshtein distance)
        for pair in nodePairs:
            string_1 = pair[0] # Node 1
            string_2 = pair[1] # Node 2
            levDistance = lDistance(string_1, string_2)
            #return [string_1, string_2, levDistance]
            gr.add_edge(string_1, string_2, weight=levDistance)
        return gr
    else:
        # Will add more similarity metrics
        pass

    
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
    

# Main function
def extractSummary(text, sent, dist):
    # Tokenize on the sentence
    sentenceTokens= sent_tokenize(text)
    
    # Clean the text and remove stop words
    s_f = cleanText(sentenceTokens)

    graph = generateGraph(s_f,dist)
    
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
    val = parser.parse_args()
    
    return val.sent, val.dist


if __name__=="__main__":
    
    #`Example 1`
    #text = requests.get('http://rare-technologies.com/the_matrix_synopsis.txt').text
    #`Example 2`
    text = "Thomas A. Anderson is a man living two lives. By day he is an average computer programmer and by night a hacker known as Neo. Neo has always questioned his reality, but the truth is far beyond his imagination. Neo finds himself targeted by the police when he is contacted by Morpheus, a legendary computer hacker branded a terrorist by the government. Morpheus awakens Neo to the real world, a ravaged wasteland where most of humanity have been captured by a race of machines that live off of the humans' body heat and electrochemical energy and who imprison their minds within an artificial reality known as the Matrix. As a rebel against the machines, Neo must return to the Matrix and confront the agents: super-powerful computer programs devoted to snuffing out Neo and the entire human rebellion. "
    #`Example 3`
    #text = "As a conversation takes place between Trinity (Carrie-Anne Moss) and Cypher (Joe Pantoliano), two free humans a table of random green numbers are being scanned and individual numbers selected, an ordinary phone number, as if a code is being deciphered or a call is being traced. Trinity discusses some unknown person. Cypher taunts Trinity, suggesting she enjoys watching him. Trinity counters that  just as the sound of a number being selected alerts Trinity that someone may be tracing their call. She ends the call.Armed policemen move down a darkened, decrepit hallway in the Heart O' the City Hotel, their flashlight beam bouncing just ahead of them. They come to room 303, kick down the door and find a woman dressed in black, facing away from them. It's Trinity. She brings her hands up from the laptop she's working on at their command. Outside the hotel a car drives up and three agents appear in neatly pressed black suits. They are Agent Smith (Hugo Weaving), Agent Brown (Paul Goddard), and Agent Jones (Robert Taylor). Agent Smith and the presiding police lieutenant argue. Agent Smith admonishes the policeman that they were given specific orders to contact the agents first, for their protection. The lieutenant dismisses this and says that they can handle and that he has two units that are bringing her down at that very moment. Agent Smith replies: No, Lieutenant. Your men are already dead.Inside, Trinity easily defeats the six policemen sent to apprehend her, using fighting and evasion techniques that seem to defy gravity. She calls Morpheus, letting him know that the line has been traced, though she doesn't know how. Morpheus informs her that she will have to  and that Agents are heading up after her. A fierce rooftop chase ensues with Trinity and an Agent leaping from one building to the next, astonishing the policemen left behind. Trinity makes a daring leap across an alley and through a small window. She has momentarily lost her pursuers and makes it to a public phone booth on the street level. The phone begins to ring. As she approaches it a garbage truck, driven by Agent Smith, careens towards the phone booth. Trinity makes a desperate dash to the phone, picking it up just moments before the truck smashes the booth into a brick wall. The three Agents reunite at the front of the truck. There is no body in the wreckage. She got out, one says. The other says, The informant is real." "We have the name of their next target,His name is Neo.Neo (Keanu Reeves), a hacker with thick black hair and a sallow appearance, is asleep at his monitor. Notices about a manhunt for a man named Morpheus scroll across his screen as he sleeps. Suddenly Neo's screen goes blank and a series of text messages appear:  just as he reads it, a knock comes at the door of his apartment, 101. It's a group of ravers and Neo gives them a contraband disc he has secreted in a copy of Simulacra and Simulation. The lead raver asks him to join them and Neo demurs until he sees the tattoo of a small white rabbit on the shoulder of a seductive girl in the group. At a rave bar Neo stands alone and aloof as the group he's with continue partying. Trinity approaches him and introduces herself. Neo recognizes her name; she was a famous hacker and had cracked the IRS database. She tells him that he is in great danger, that they are watching him and that she knows that he is searching for answers, particularly to the most important question of all: what is the Matrix? The pulsing music of the bar gives way to the repetitious blare of Neo's alarm clock; it's 9:18 and he's late for work.At his job at Metacortex, a leading software company housed in an ominous high rise, Neo is berated by his boss for having a problem with authority, for thinking he's special. Neo listens to his boss, but his attention is on the persons cleaning the window of the office. Back at his bleak cubicle Neo receives a delivery as Thomas Anderson. Upon opening the package he finds a cellphone which immediately rings. On the other end is Morpheus, who informs Neo that they've both run out of time and that are coming for him. Morpheus tells him to slowly look up, toward the elevator. Agents Smith, Jones, and Brown are there, obviously looking for him, as a woman points towards Neo's cube. Morpheus tries to guide Neo out of the building but when he is instructed to get on a scaffolding and take it to the roof Neo rejects Morpheus's advice, allowing himself to be taken by the Agents."

    sent, dist = readCommandLineArguments()
    summary = extractSummary(text,sent,dist)
    print summary
