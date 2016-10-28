# TextRank

* `slidingWindow.py`  :  Implements a Sliding Window over tokens in a list. Takes in parameter for `window size` and `Tokens list`. For `length = 11` ; `5`  tokens on `each side` of the current word will define your window. Returns a `co-occurence matrix` with cell value as the distance between the tokens in a window.
* `data.txt`  :  Sample data file . Can be used for testing purpose.
* `textRank.py`  : Main code ; Has functions for both `summarization` and `keyword extraction`.

###### Runs summarization ( 3 : sentences  || distance metric : common tokens || draw tree : True)
`> python textRank.py -sent 3 -dist CT -draw T`
You might need to uncomment the corresponding function call in the code.
###### Runs Keyword extraction ( 10 : Words  || distance metric :   || draw tree: True ) 
`> python textRank.py -sent 10 -dist CT -draw T` You might need to uncomment the corresponding function call in the code.


#### Feel free to send any questions by raising an issue.
