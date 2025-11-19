#!/bin/bash

echo "===> Running CLS stage..."
cd CLS

python test_KNN_3.py
python test_BIO_true.py

echo "===> Running REG stage..."
cd ../REG

python test.py

echo "===> All tasks completed."
