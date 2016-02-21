#!/usr/bin/env bash

cd ./wv
BASEURL="https://s3-us-west-1.amazonaws.com/textclf/wv/"

FS="IMDB-GloVe-100dim.txt
IMDB-GloVe-300dim.txt
Yelp-GloVe-300dim.txt
glove.42B.300d.120000.txt"

echo "Starting downloads..."
for F in $FS; do
	wget ${BASEURL}${F}
done
echo "Download complete"
cd ..
