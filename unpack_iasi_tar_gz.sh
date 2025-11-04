#!/bin/bash

path_data="/mnt/e/VAMPIRE2/IASI/raw/new/est/"

for a in "$path_data"*.tar.gz;
do
    tar -zxvf $a;
done
exit