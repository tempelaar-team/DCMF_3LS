import numpy as np
import time
import scipy.sparse
import scipy.sparse.linalg
from numba import jit, prange

with open('input_threelevel.txt') as f:
    for line in f:
        line1 = line.replace(" ", "")
        line1 = line1.rstrip('\n')
        exec(str(line), globals())

start_time_all = time.time()

#Hamiltonian
start_time_H=time.time()

#assign index
w = np.append(0, w)
alpha = np.append(0, alpha)

#assign index for modes
num_modesbasis=int(1 + 4*N + (2*N)*(2*N-1)/2 + 2*N + (2*N)*(2*N-1) + (2*N)*(2*N-1)*(2*N-2)/6)
index_vac = 0
index_onephoton = 1 + index_vac
index_twophoton = int(2 * N + index_onephoton)
index_threephoton = int(2 * N + (2 * N) * (2 * N - 1) / 2 + index_twophoton)

indexab=np.zeros((num_modesbasis, 3))
iter= int(1+4*N)
indexab[0]=((0,0,0))
for i in range(1,int(2 * N+1)):
    indexab[i] = ((i, 0, 0))
    indexab[int(i+2*N)] = ((i, i, 0))
    for j in range(i + 1, int(2 * N+1)):
        indexab[iter] = ((i,j, 0))
        iter = iter+1

iter=int(index_threephoton)
for i in range(1, int(2 * N+1)):
    indexab[iter] = ((i, i, i))
    iter += 1

iter=int(index_threephoton + 2*N)
for i in range(1,int(2*N+1)):
    for j in range(1,int(2*N+1)):
        if j != i:
            indexab[iter] = ((i, i, j))
            iter += 1

iter=int(index_threephoton + 2*N + (2*N)*(2*N-1))
for i in range(1, int(2 * N+1)):
    for j in range(i+1, int(2 * N+1)):
        for k in range(j+1, int(2 * N+1)):
            indexab[iter] = ((i, j, k))
            iter = iter+1

#assign index with atoms
total_number=int(energy.size**num_atom*num_modesbasis)
index = np.zeros((total_number, int(num_atom+3)))

for n in range(num_atom):
    iii = 0
    for i in range(energy.size**(num_atom-n)):
        index[int(iii):int(iii+(n+1)*(num_modesbasis)), int(num_atom-1-n)] = i%energy.size
        if n == 0:
            index[int(iii):int(iii + num_modesbasis), int(num_atom):int(num_atom+3)] = indexab
        iii += (n+1)*(num_modesbasis)

# make index for indexab_d and indexabc_d
iter=0
indexab_d = np.zeros((int((2*N)*(2*N-1)/2),2), dtype=int)
for i in range(1, int(2*N+1)):
    for j in range(i+1, int(2*N+1)):
        indexab_d[iter] = ((i,j))
        iter +=1

#@jit(nopython=True, fastmath=True)#, parallel=True)
def get_H():
    H=np.array([energy[0]], dtype='float64')
    ni=nj=0
    #generate the diagonal terms
    Hii = [energy[int(index[i][ni])] + hbar * w[int(index[i][1])]
         + hbar * w[int(index[i][2])] + hbar * w[int(index[i][3])] for i in range(1, total_number)]
    Hi = [i for i in range(0, total_number)]
    Hj = Hi.copy()
    H = np.concatenate((H, np.array(Hii)))

    # Only focus on the offdiagonal terms from the middle electronic state
    # zero photon state
    # |e, 0, 0, 0> -> |g, i, 0, 0>
    nowi = index_vac + num_modesbasis
    Hij = np.sqrt((w[1:]) / (epsilon * l)) * np.sin(
        np.pi * alpha[1:] * r_atom[nj] / l)
    H = np.concatenate((H, nu[0] * Hij, nu[1] * Hij))
    nowHi= [nowi for j in range(index_onephoton, index_twophoton)]
    Hi += nowHi
    Hj += [j for j in range(index_onephoton, index_twophoton)]
    # |e, 0, 0, 0> -> |e2, i, 0 ,0> #CRW
    Hi += nowHi
    Hj += [j for j in range(num_modesbasis*2 + index_onephoton, num_modesbasis*2 + index_twophoton)]

    # one photon state
    # |e, i, 0, 0> -> |g, i, i, 0>
    H = np.concatenate((H, np.sqrt(2)*nu[0] * Hij, np.sqrt(2)*nu[1] * Hij, nu[0] * Hij, nu[1] * Hij))
    nowHi = [i for i in range(num_modesbasis + index_onephoton, num_modesbasis + index_twophoton)]
    Hi += nowHi
    Hj += [j for j in range(index_twophoton, int(index_twophoton + 2*N))]
    # |e, i, 0, 0> -> |e2, i, i, 0> #CRW
    Hi += nowHi
    Hj += [j for j in range(num_modesbasis*2+index_twophoton, int(num_modesbasis*2+index_twophoton + 2 * N))]
    # |e, i, 0, 0> -> |g, 0, 0, 0> #CRW
    Hi += nowHi
    Hj += [index_vac for j in range(num_modesbasis + index_onephoton, num_modesbasis + index_twophoton)]
    # |e, i, 0, 0> -> |e2, 0, 0, 0>
    Hi += nowHi
    Hj += [index_vac + num_modesbasis*2 for j in range(num_modesbasis + index_onephoton, num_modesbasis + index_twophoton)]
    # |e, i (j), 0, 0> <- |g, i, j, 0>
    index_for_ab=0
    nowHij=[]
    for i in range(int(index_twophoton + 2 * N), index_threephoton):
        Hi += [i, i]
        nowHj =(num_modesbasis + index_onephoton + indexab_d[index_for_ab] - 1).tolist()
        Hj += nowHj
        preHij = np.sqrt((w[indexab_d[index_for_ab]]) / (epsilon * l)) * np.sin(
                np.pi * alpha[indexab_d[index_for_ab]] * r_atom[nj] / l)
        nowHij += (nu[0]*np.flip(preHij)).tolist()
        # |e, i (j), 0, 0> <- |e2, i, j, 0> #CRW
        Hi += [num_modesbasis*2 + i, num_modesbasis*2+i]
        Hj += nowHj
        nowHij += (nu[1]*np.flip(preHij)).tolist()
        index_for_ab+=1
    H = np.concatenate((H, nowHij))

    # two photon state
    # |e, i, i, 0> -> |g, i, 0, 0> #CRW
    H = np.concatenate((H, np.sqrt(2)*nu[0] * Hij, np.sqrt(2)*nu[1] * Hij,
                        np.sqrt(3)*nu[0] * Hij, np.sqrt(3)*nu[1] * Hij))
    nowHi = [i for i in range(int(num_modesbasis + index_twophoton), int(num_modesbasis + index_twophoton+2*N))]
    Hi += nowHi
    Hj += [j for j in range(index_onephoton, index_twophoton)]
    # |e, i, i, 0> -> |e2, i, 0, 0>
    Hi += nowHi
    Hj += [j for j in range(num_modesbasis*2+index_onephoton, num_modesbasis*2+index_twophoton)]
    # |e, i, i, 0> -> |g, i, i, i>
    Hi += nowHi
    Hj += [j for j in range(index_threephoton, int(index_threephoton + 2*N))]
    # |e, i, i, 0> -> |e2, i, i, i> #CRW
    Hi += nowHi
    Hj += [j for j in range(num_modesbasis*2+index_threephoton, int(num_modesbasis*2+index_threephoton + 2*N))]

    index_for_ab=0
    nowHij=[]
    for i in range(int(num_modesbasis + index_twophoton), int(num_modesbasis + index_twophoton+2*N)):
        # |e, i, i, 0> -> |g, i, i, k>
        Hi += [i for j in range(int(index_threephoton + 2 * N + index_for_ab*(2*N-1)),
                                  int(index_threephoton + 2*N + (2*N-1) + index_for_ab*(2*N-1)))]
        nowHj = [j for j in range(int(index_threephoton + 2 * N + index_for_ab*(2*N-1)),
                                  int(index_threephoton + 2*N + (2*N-1) + index_for_ab*(2*N-1)))]
        Hj += nowHj
        nowHij += (nu[0] * Hij[:index_for_ab]).tolist()
        nowHij += (nu[0] * Hij[index_for_ab + 1:]).tolist()
        # |e, i, i, 0> -> |e2, i, i, k> #CRW
        Hi += [i for j in range(int(index_threephoton + 2 * N + index_for_ab*(2*N-1)),
                                  int(index_threephoton + 2*N + (2*N-1) + index_for_ab*(2*N-1)))]
        nowHj = [j for j in range(int(num_modesbasis*2 + index_threephoton + 2 * N + index_for_ab*(2*N-1)),
                                  int(num_modesbasis*2 + index_threephoton + 2*N + (2*N-1)  + index_for_ab*(2*N-1)))]
        Hj += nowHj
        nowHij += (nu[1] * Hij[:index_for_ab]).tolist()
        nowHij += (nu[1] * Hij[index_for_ab + 1:]).tolist()
        index_for_ab+=1
    H = np.concatenate((H, nowHij))

    index_for_ab=0
    nowHij=[]
    for i in range(int(num_modesbasis+index_twophoton + 2 * N), num_modesbasis+index_threephoton):
        # |e, i, j, 0> -> |g, i (j), 0, 0> #CRW
        Hi += [i, i, i, i]
        nowHj =(index_onephoton + indexab_d[index_for_ab] - 1).tolist()
        Hj += nowHj
        preHij = np.sqrt((w[indexab_d[index_for_ab]]) / (epsilon * l)) * np.sin(
                np.pi * alpha[indexab_d[index_for_ab]] * r_atom[nj] / l)
        nowHij += (nu[0]*np.flip(preHij)).tolist()
        # |e, i, j, 0> -> |e2, i (j), 0, 0>
        nowHj =(num_modesbasis*2+ index_onephoton + indexab_d[index_for_ab] - 1).tolist()
        Hj += nowHj
        nowHij += (nu[1]*np.flip(preHij)).tolist()
        index_for_ab+=1
    H = np.concatenate((H, nowHij))

    # |e, i, j, 0> <- |g, i, j, k>
    # |e, i, j, 0> <- |e2, i, j, k> #CRW
    nowHij=[]
    iter = 0
    index_first=0
    index_for_third = 1
    index_for_third2 = 0
    index_first_prev = -1
    num_third = int(2 * N - 2)
    torealindex_e = num_modesbasis + index_twophoton + 2 * N
    for i in range(1, int(2 * N + 1)):
        for j in range(i + 1, int(2 * N + 1)):
            index_for_second=1
            index_for_third = (2 -(index_first - index_first_prev)) * index_for_third + (index_first - index_first_prev -1)
            index_for_third2 = (2-(index_first-index_first_prev)) * index_for_third2
            num_third = num_third - (index_first - index_first_prev -1)
            for k in range(j + 1, int(2 * N + 1)):
                torealindex_g = index_threephoton + 2*N + (2*N)*(2*N-1) + iter
                torealindex_e2 = torealindex_g + num_modesbasis*2
                Hi += [torealindex_g, torealindex_g, torealindex_g,
                       torealindex_e2, torealindex_e2, torealindex_e2]
                Hj += [torealindex_e+index_first, torealindex_e+index_first+index_for_second,
                       torealindex_e+index_first + index_for_third - index_for_third2 + num_third]
                Hj += [torealindex_e+index_first, torealindex_e+index_first+index_for_second,
                       torealindex_e+index_first + index_for_third - index_for_third2 + num_third]
                preHij = np.sqrt((w[[k,j,i]]) / (epsilon * l)) * np.sin(
                    np.pi * alpha[[k,j,i]] * r_atom[nj] / l)
                nowHij += (nu[0] * preHij).tolist()
                nowHij += (nu[1] * preHij).tolist()
                iter += 1
                index_for_second += 1
                index_for_third += 1
            index_first_prev = Hj[-3] - torealindex_e
            index_first += 1
            index_for_third2 += 1
    H = np.concatenate((H, nowHij))

    # three photon state
    # |e, i, i, i> -> |g, i, i, 0> #CRW
    H = np.concatenate((H, np.sqrt(3)*nu[0] * Hij, np.sqrt(3)*nu[1] * Hij))
    nowHi = [i for i in range(int(num_modesbasis + index_threephoton), int(num_modesbasis + index_threephoton+2*N))]
    Hi += nowHi
    Hj += [j for j in range(index_twophoton, int(index_twophoton+2*N))]
    # |e, i, i, i> -> |e2, i, i, 0>
    Hi += nowHi
    Hj += [j for j in range(int(num_modesbasis*2 +index_twophoton), int(num_modesbasis*2 + index_twophoton+2*N))]

    index_for_ab=0
    nowHij=[]
    for i in range(int(index_twophoton), int(index_twophoton+2*N)):
        # |g, i, i, 0> -> |e, i, i, k> #CRW
        Hi += [i for j in range(int(index_threephoton + 2 * N + index_for_ab * (2 * N - 1)),
                                int(index_threephoton + 2 * N + (2 * N - 1) + index_for_ab * (2 * N - 1)))]
        nowHj = [j for j in range(int(num_modesbasis+index_threephoton + 2 * N + index_for_ab*(2*N-1)),
                                  int(num_modesbasis+index_threephoton + 2*N + (2*N-1) + index_for_ab*(2*N-1)))]
        Hj += nowHj
        nowHij += (nu[0] * Hij[:index_for_ab]).tolist()
        nowHij += (nu[0] * Hij[index_for_ab+1:]).tolist()
        # |e2, i, i, 0> -> |e, i, i, k>
        Hi += [num_modesbasis*2 + i for j in range(int(index_threephoton + 2 * N + index_for_ab*(2*N-1)),
                                  int(index_threephoton + 2*N + (2*N-1) + index_for_ab*(2*N-1)))]
        Hj += nowHj
        nowHij += (nu[1] * Hij[:index_for_ab]).tolist()
        nowHij += (nu[1] * Hij[index_for_ab+1:]).tolist()
        index_for_ab+=1
    H = np.concatenate((H, nowHij))

    # |e, i, j, k> -> |g, i, j, 0> #CRW
    # |e, i, j, k> -> |e2, i, j, 0>
    nowHij=[]
    iter = 0
    index_first=0
    index_for_third_g = 1
    index_for_third2_g = 0
    index_for_third_e2 = 1
    index_for_third2_e2 = 0
    index_first_prev_g = -1
    index_first_prev_e2 = -1
    num_third_g = int(2 * N - 2)
    num_third_e2 = int(2 * N - 2)
    torealindex_g = index_twophoton + 2 * N
    torealindex_e2 = num_modesbasis*2 + index_twophoton + 2 * N
    for i in range(1, int(2 * N + 1)):
        for j in range(i + 1, int(2 * N + 1)):
            index_for_second=1
            index_for_third_g = (2 -(index_first - index_first_prev_g)) * index_for_third_g\
                              + (index_first - index_first_prev_g -1)
            index_for_third2_g = (2-(index_first-index_first_prev_g)) * index_for_third2_g
            num_third_g = num_third_g - (index_first - index_first_prev_g -1)
            index_for_third_e2 = (2 -(index_first - index_first_prev_e2)) * index_for_third_e2\
                              + (index_first - index_first_prev_e2 -1)
            index_for_third2_e2 = (2-(index_first-index_first_prev_e2)) * index_for_third2_e2
            num_third_e2 = num_third_e2 - (index_first - index_first_prev_e2 -1)
            for k in range(j + 1, int(2 * N + 1)):
                torealindex_e = num_modesbasis + index_threephoton + 2*N + (2*N)*(2*N-1) + iter
                Hi += [torealindex_e, torealindex_e, torealindex_e,
                       torealindex_e, torealindex_e, torealindex_e]
                Hj += [torealindex_g+index_first, torealindex_g+index_first+index_for_second,
                       torealindex_g+index_first + index_for_third_g - index_for_third2_g + num_third_g]
                Hj += [torealindex_e2+index_first, torealindex_e2+index_first+index_for_second,
                       torealindex_e2+index_first + index_for_third_e2 - index_for_third2_e2 + num_third_e2]
                preHij = np.sqrt((w[[k,j,i]]) / (epsilon * l)) * np.sin(
                    np.pi * alpha[[k,j,i]] * r_atom[nj] / l)
                nowHij += (nu[0] * preHij).tolist()
                nowHij += (nu[1] * preHij).tolist()
                iter += 1
                index_for_second += 1
                index_for_third_g += 1
                index_for_third_e2 += 1
            index_first_prev_g = Hj[-6] - torealindex_g
            index_first_prev_e2 = Hj[-3] - torealindex_e2
            index_first += 1
            index_for_third2_g += 1
            index_for_third2_e2 += 1
    H = np.concatenate((H, nowHij))
    Hindex = np.vstack((np.array(Hi,dtype=int), np.array(Hj,dtype=int))).transpose()
    return Hindex, H

@jit(nopython=True, fastmath=True)#, parallel=True)
def my_matmul(Hindex, H, wavefn):
    wavefn_new=np.zeros_like(wavefn)
    #wavefn_new = wavefn.copy()
    for h in range(H.size):
        i = int(Hindex[h,0])
        j = int(Hindex[h,1])
        Hij = H[h]
        if i == j:
            wavefn_new[i] += Hij * wavefn[i]
        else:
            wavefn_new[j] += Hij * wavefn[i]
            wavefn_new[i] += Hij * wavefn[j]
    return wavefn_new

@jit(nopython=True, fastmath=True, parallel=True)
def RK4(Hindex, H, wavefn, dt):
    K1 = -1j / hbar * my_matmul(Hindex, H, wavefn)
    K2 = -1j / hbar * my_matmul(Hindex, H, (wavefn + 0.5 * dt * K1))
    K3 = -1j / hbar * my_matmul(Hindex, H, (wavefn + 0.5 * dt * K2))
    K4 = -1j / hbar * my_matmul(Hindex, H, (wavefn + dt * K3))
    wavefn_new = wavefn + dt * 0.166667 * (K1 + 2 * K2 + 2 * K3 + K4)
    return wavefn_new

def get_eig(H):
    sparse_H=scipy.sparse.csc_matrix(H)
    evals, evecs = scipy.sparse.linalg.eigsh(sparse_H, k=1, which='SA')
    return evals, evecs

@jit(nopython=True, fastmath=True, parallel=True)
def time_prop(all_ini_wavefn):
    wavefn = all_ini_wavefn
    wavefn_save = np.zeros((int(steps.size/savestep + 1),all_ini_wavefn.size), dtype='complex128')
    wavefn_save[0] = wavefn
    for t in range(1,int(tmax/dt+1),1):
        wavefn = RK4(Hindex, H, wavefn, dt)
        if t % savestep == 0:
            wavefn_save[int(t/savestep)] = wavefn
        if t == steps[-1]:
            wavefnend = wavefn
    return wavefn_save, wavefnend

Hindex, H = get_H()

all_ini_wavefn=np.zeros(total_number, dtype='complex128')

end_time_H=time.time()
print('Hamiltonian generated. Basis size = '+ str(total_number) +', half of H non-zero size = '+ str(H.size) + ', 2N=' + str(2*N))
print('Sparsity = ' + str(H.size*2/total_number**2))
print('Running Wall Time = %10.3f second' % (end_time_H - start_time_H))

start_time_eigh=time.time()
Hsparse = scipy.sparse.coo_matrix((np.concatenate((H,H)),
                             (np.concatenate((Hindex[:,0],Hindex[:,1])),
                              np.concatenate((Hindex[:,1],Hindex[:,0])))))
Hsparse = Hsparse.tocsr()
evals, evecs = get_eig(Hsparse)
end_time_eigh=time.time()
#
print('Found the lowest eigenstate. Running Wall Time = %10.3f second' % (end_time_eigh - start_time_eigh))

start_time_run=time.time()
#all_ini_wavefn[np.where((index[:,0] == 1) & (index[:,1] == 0))]=1 # |e, 0>
#all_ini_wavefn[np.where((index[:,0] == 1) & (index[:,1] == 0) & (index[:,2] == -1))]=1 # |e, g, 0>
#all_ini_wavefn=evecs[:,0] # "Real ground state"

all_ini_wavefn[np.where((index[:,0] == 1))]\
    = evecs[:,0][np.where((index[:,0] == 0))]
all_ini_wavefn[np.where((index[:,0] == 2))]\
    = evecs[:,0][np.where((index[:,0] == 1))]
all_ini_wavefn=all_ini_wavefn/np.linalg.norm(all_ini_wavefn)

wavefn_save, wavefnend=time_prop(all_ini_wavefn)

end_time_all = time.time()

np.save('index.npy', index)
np.savez_compressed('rho.npz', rho=wavefn_save)
np.savetxt('final_rho.csv', wavefnend, delimiter=',')
print('Finished time evolution. Running Wall Time = %10.3f second' % (end_time_all - start_time_run))
print('Calculation Finished. Running Wall Time = %10.3f second' % (end_time_all - start_time_all))
