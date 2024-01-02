# Ehrenfest modeling of cavity vacuum fluctuations and how to achieve emission from a three-level atom
Authors: Ming-Hsiu Hsieh, Alex Krotz, Roel Tempelaar

*J. Chem. Phys.* **2023**, 159, 221104

Arxiv: https://arxiv.org/abs/2309.01912

Publication: https://pubs.aip.org/aip/jcp/article/159/22/221104/2929335

## Usage of the code
The code has been run under Python 3.8.

Required packages: Numpy, Numba, Ray, time, matplotlib

To run a simulation with MF or DC-MF:
```
python3 main.py input.txt
```
In line 29 of main.py, one can run the MF dynamics by calling `runCalc` in `mixQC_MF_3LS` and DC-MF dynamics by calling `runCalc` in `mixQC_DCMF_3LS`.

To get the results and plots for both MF and DC-MF dynamics:
```
python3 analyze.py
```

Alternatively, to run a simulation with CISDT (note that this also reads input.txt):
```
python3 exact_CISDT_RK4_noscipy.py
```

## Explanation to each file
+ input.txt: input file where one sets all parameters
+ main.py: the main executive file
+ mixQC_MF.py: functions for MF dynamics
+ mixQC_DCMF.py: functions for DC-MF dynamics
+ analyze.py: analyze the MF or DC-MF results and get figures
+ exact_CISDT_RK4_noscipy.py: executive file for the CISDT result

