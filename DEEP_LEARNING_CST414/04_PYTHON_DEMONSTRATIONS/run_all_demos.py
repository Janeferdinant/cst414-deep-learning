"""Master Demonstration Runner for CST414 Deep Learning"""
import os, sys

print('KTU CST414 Deep Learning Demonstration Suite')
print('Faculty: Jane Rose | Department of Computer Science & Engineering')
for mod in range(1, 6):
    print(f'Running Module {mod} demos...')
    os.system(f'python MODULE_{mod}/module_{mod}_all_demos.py')
