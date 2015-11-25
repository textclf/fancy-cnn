#!/usr/bin/env bash

DATA_FILE="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

ZIPFILE="aclImdb_v1.tar.gz"
TARFILE="aclImdb_v1.tar"
DIR="aclImdb"

echo "Downloading IMDB dataset..."

wget $DATA_FILE

echo "Unzipping..."
gunzip $ZIPFILE

echo "Extracting..."
tar -xf $TARFILE

echo "Cleaning up..."
rm $TARFILE

echo "Data tree in $DIR."
