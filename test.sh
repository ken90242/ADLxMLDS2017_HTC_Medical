#!/bin/bash

echo "========================================================="
if [ ! -f test-save_24.npy ]; then
		echo "[-] download test-save_24.npy"
		echo ""
    curl ftp://140.112.107.150/test-save_24.npy -o test-save_24.npy
fi

echo "========================================================="
echo "[-] install judger_medical"
echo ""
sudo pip install judger_medical/judger_medical-1.0-py2.py3-none-any.whl

echo "Start Testing..."

python judge.py
