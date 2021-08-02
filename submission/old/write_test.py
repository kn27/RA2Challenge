

import os

try:
    with open('./train/test.txt', 'w') as f:
        f.write('test')
except OSError as E:
    print(E)

try:
    with open('./test/test.txt', 'w') as f:
        f.write('test')
except OSError as E:
    print(E)
    

try:
    with open('./output/test.txt', 'w') as f:
        f.write('test')
except OSError as E:
    print(E)
    

try:
    with open('./temp/test.txt', 'w') as f:
        f.write('test')
except OSError as E:
    print(E)
