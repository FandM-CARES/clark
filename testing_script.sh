#!/bin/bash

counter=1

while [ $counter -le 2 ]
do
    echo $counter
    python3 baseline.py -config testBaseline.config
    ((counter++))
done

echo ****