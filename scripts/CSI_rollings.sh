#!/bin/bash
nohup bash -c '
set -e


python rolling_train.py -d CSI300 -m LSTM -g 5 -l 0.001 -s 16 --seed 1234 
python rolling_train.py -d CSI300 -m LSTM -g 5 -l 0.001 -s 33 --seed 1234 
python rolling_train.py -d CSI300 -m LSTM -g 5 -l 0.001 -s 64 --seed 1234 

python rolling_train.py -d CSI2259 -m LSTM -g 5 -l 0.001 -s 8 --seed 1234 
python rolling_train.py -d CSI2259 -m LSTM -g 5 -l 0.001 -s 16 --seed 1234 
python rolling_train.py -d CSI2259 -m LSTM -g 5 -l 0.001 -s 32 --seed 1234 
python rolling_train.py -d CSI2259 -m LSTM -g 5 -l 0.001 -s 64 --seed 1234 

python rolling_train.py -d CSI -m LSTM -g 5 -l 0.001 -s 8 --seed 1234 
python rolling_train.py -d CSI -m LSTM -g 5 -l 0.001 -s 16 --seed 1234 
python rolling_train.py -d CSI -m LSTM -g 5 -l 0.001 -s 32 --seed 1234 
python rolling_train.py -d CSI -m LSTM -g 5 -l 0.001 -s 64 --seed 1234 
wait

wait
' &