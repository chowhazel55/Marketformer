#!/bin/bash
nohup bash -c '
set -e


python traincsi.py -d CSI300 -m LSTM -g 5 -l 0.001 -s 32 -v 2022 -t 2023 --seed 1234 
python traincsi.py -d CSI300 -m LSTM -g 5 -l 0.001 -s 64 -v 2022 -t 2023 --seed 1234 

python traincsi.py -d CSI300 -m LSTM -g 5 -l 0.001 -s 8 -v 2023 -t 2024 --seed 1234 
python traincsi.py -d CSI300 -m LSTM -g 5 -l 0.001 -s 16 -v 2023 -t 2024 --seed 1234 
python traincsi.py -d CSI300 -m LSTM -g 5 -l 0.001 -s 32 -v 2023 -t 2024 --seed 1234 
python traincsi.py -d CSI300 -m LSTM -g 5 -l 0.001 -s 64 -v 2023 -t 2024 --seed 1234 

python traincsi.py -d CSI300 -m LSTM -g 5 -l 0.001 -s 8 -v 2024 -t 2025 --seed 1234 
python traincsi.py -d CSI300 -m LSTM -g 5 -l 0.001 -s 16 -v 2024 -t 2025 --seed 1234 
python traincsi.py -d CSI300 -m LSTM -g 5 -l 0.001 -s 32 -v 2024 -t 2025 --seed 1234 
python traincsi.py -d CSI300 -m LSTM -g 5 -l 0.001 -s 64 -v 2024 -t 2025 --seed 1234 
wait

wait
' &