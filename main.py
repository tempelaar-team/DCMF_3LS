# This is a sample Python script.
import os
import sys
from shutil import copyfile
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np

def main():
    args = sys.argv[1:]
    if not args:
        print('Usage: python main.py inputfile')
    inputfile = args[0]
    with open(inputfile) as f:
        for line in f:
            line1 = line.replace(" ", "")
            line1 = line1.rstrip('\n')
            exec(str(line), globals())
    if 'status' in globals() and status=='RESTART':
        global calcdir
        calcdir = calcdir + '/RESTART'
        print('RESTART with endpoints of Q, P, and wavefn')
        os.mkdir(calcdir)
    else:
        if not (os.path.exists(calcdir)):
            os.mkdir(calcdir)
    copyfile(inputfile, 'inputfile.tmp')
    copyfile(inputfile, calcdir + '/inputfile_bk.txt')
    from mixQC_DCMF_3LS import runCalc
    runCalc()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
