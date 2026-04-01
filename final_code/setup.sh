#!/bin/bash

MODE=0

while getopts "m:" flag; do
    case "${flag}" in 
    m) MODE=$OPTARG ;;
    esac
done

if [ "$MODE" == 'env' ]; then
    rm -rf ./env

    python3 -m venv env
    source ./env/bin/activate
    pip install kaggle
    pip install ipython
    pip3 install cairosvg
    
    pip install trl
    pip install peft
    pip install datasets
    pip install transformers
    
    pip3 install torch
    pip install -U bitsandbytes>=0.46.1
    pip install pandas matplotlib
    
    pip install unsloth datasets trl transformers accelerate peft bitsandbytes pandas lxml 

elif [ "$MODE" == 'data' ]; then
    rm -rf ./dataset
    mkdir ./dataset

    kaggle competitions download -c dl-spring-2026-svg-generation
    unzip dl-spring-2026-svg-generation.zip 
    rm dl-spring-2026-svg-generation.zip 
    mv sample_submission.csv ./dataset/sample_submission.csv
    mv test.csv ./dataset/test.csv
    mv train.csv ./dataset/train.csv
fi