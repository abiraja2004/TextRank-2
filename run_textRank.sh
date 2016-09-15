#!/bin/bash

echo "Running summarization..."
echo "=========================="
echo "Sentences - 2"
echo "Weight Metric - Levenshtein"
echo "=========================="
python textRank.py -s 2 -d L -draw T
