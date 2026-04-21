#!/bin/bash
nohup bash -c '
set -e

python traincsi.py -d CSI -m LSTM -g 7 -l 0.001 -s 8  -v 2018 -t 2019 --seed 1234 
python traincsi.py -d CSI -m LSTM -g 7 -l 0.001 -s 16  -v 2018 -t 2019 --seed 1234 
python traincsi.py -d CSI -m LSTM -g 7 -l 0.001 -s 32  -v 2018 -t 2019 --seed 1234 
python traincsi.py -d CSI -m LSTM -g 7 -l 0.001 -s 64  -v 2018 -t 2019 --seed 1234 

python traincsi.py -d CSI -m LSTM -g 7 -l 0.001 -s 8  -v 2019 -t 2020 --seed 1234 
python traincsi.py -d CSI -m LSTM -g 7 -l 0.001 -s 16  -v 2019 -t 2020 --seed 1234 
python traincsi.py -d CSI -m LSTM -g 7 -l 0.001 -s 32  -v 2019 -t 2020 --seed 1234 
python traincsi.py -d CSI -m LSTM -g 7 -l 0.001 -s 64  -v 2019 -t 2020 --seed 1234 

python traincsi.py -d CSI -m LSTM -g 7 -l 0.001 -s 8  -v 2020 -t 2021 --seed 1234 
python traincsi.py -d CSI -m LSTM -g 7 -l 0.001 -s 16  -v 2020 -t 2021 --seed 1234 
python traincsi.py -d CSI -m LSTM -g 7 -l 0.001 -s 32  -v 2020 -t 2021 --seed 1234 
python traincsi.py -d CSI -m LSTM -g 7 -l 0.001 -s 64  -v 2020 -t 2021 --seed 1234 



wait
' &