import numpy as np
import math
from numba import jit
from numba import vectorize
from os import path
import ray
import time

ray.init(ignore_reinit_error=True)

# input file reading part
with open('inputfile.tmp') as f:
    for line in f:
        line1 = line.replace(" ", "")
        line1 = line1.rstrip('\n')
        exec(str(line), globals())
#wgrid = np.transpose(np.tile(w, [num_trj, 1]))
wgrid = w[:,np.newaxis]

if num_atom != np.array(ini_wavefn).size/energy.size or num_atom != np.array(r_atom).size:
    print('WARNING! the number of atoms does not equal to given wavefn or positions.')

# @jit(nopython=True)
def Wigner(wavefn, lambda_al):
    avg = 0
    stdP = np.sqrt(hbar * w / (2))
    stdQ = np.sqrt(hbar / (2 * w))
    P0 = np.random.normal(avg, stdP)
    Q0 = np.random.normal(avg, stdQ)
    return Q0, P0

def Boltzmann(wavefn, lambda_al):
    avg = 0
    stdP = np.ones_like(w)*np.sqrt(kT)
    stdQ = np.sqrt(kT/w**2)
    P0 = np.random.normal(avg, stdP)
    Q0 = np.random.normal(avg, stdQ)
    return Q0, P0

@jit(nopython=True)
def Q_F(wavefn, lambda_alpha_grid):
    force = 2 * wgrid * nu12 * lambda_alpha_grid * np.real(np.conjugate(wavefn[0]) * wavefn[1])
    force = force + 2 * wgrid * nu23 * lambda_alpha_grid * np.real(np.conjugate(wavefn[1]) * wavefn[2])
    #force=0
    return force

# Quantum Hamiltonian redo
@jit(nopython=True)
def H_q(Q, Qzp, wavefn, lambda_alpha):
    Hq = np.zeros((energy.size, energy.size, num_trj))
    Hq[0, 0, :] = energy[0]
    Hq[1, 1, :] = energy[1]
    Hq[2, 2, :] = energy[2]
    for n in range(num_trj):
        intQ = Qzp[:, n]
        Hq[1, 0, n] = np.sum(nu12 * w * lambda_alpha * intQ)
        Hq[2, 1, n] = np.sum(nu23 * w * lambda_alpha * intQ)
    Hq[0, 1, :] = Hq[1, 0, :]
    Hq[1, 2, :] = Hq[2, 1, :]
    #print(Hq,'Hq')
    return Hq

@jit(nopython=True, fastmath=True)
def get_eigs(H):
    evals, evecs = np.linalg.eigh(H)
    return evals, evecs

@jit(nopython=True, fastmath=True)
def vec_db_to_adb(rho_db, evecs):
    return np.transpose(evecs.astype('complex128'))@rho_db.astype('complex128')

@jit(nopython=True, fastmath=True)
def vec_adb_to_db(rho_adb, evecs):
    return evecs.astype('complex128')@rho_adb.astype('complex128')

@jit(nopython=True,fastmath=True)
def prop_Q_exact(psiA, q, Qzp, lambda_alpha, dt):
    Hq=H_q(q, Qzp, psiA, lambda_alpha)
    psiA_db = psiA.copy()
    # a loop for Hq
    for n in range(num_trj):
        evalsA, evecsA = get_eigs(Hq[:,:,n])
        evalsA_exp = np.exp(-1.0j * evalsA * dt)
        psiA_adb = vec_db_to_adb(psiA[:,n], evecsA)
        psiA_adb = evalsA_exp * psiA_adb
        psiA_db[:,n] = vec_adb_to_db(psiA_adb, evecsA)
    return psiA_db

@jit(nopython=True,fastmath=True)
def RK4(p_bath, q_bath, QF, dt):
    Fq = np.real(QF)
    Fp = Fq * 0
    K1 = dt * (p_bath + Fp)
    L1 = -dt * (wgrid**2 * q_bath + Fq)
    K2 = dt * ((p_bath + 0.5 * L1) + Fp)
    L2 = -dt * (wgrid**2 * (q_bath + 0.5 * K1) + Fq)
    K3 = dt * ((p_bath + 0.5 * L2) + Fp)
    L3 = -dt * (wgrid**2 * (q_bath + 0.5 * K2) + Fq)
    K4 = dt * ((p_bath + L3) + Fp)
    L4 = -dt * (wgrid**2 * (q_bath + K3) + Fq)
    q_bath = q_bath + 0.166667 * (K1 + 2 * K2 + 2 * K3 + K4)
    p_bath = p_bath + 0.166667 * (L1 + 2 * L2 + 2 * L3 + L4)
    return p_bath, q_bath

@jit(nopython=True,fastmath=True)
def class_energy(Q, P):
    c_energy = 1/2 * np.sum((P**2) + (wgrid**2)*(Q**2), axis=0)
    return c_energy

#@jit(nopython=True)
def quantum_energy(wavefn, Q, Qzp, lambda_alpha):
    bra = np.conjugate(wavefn)
    q_energy = np.zeros(num_trj)
    qc_energy = np.zeros(num_trj)
    Hq=H_q(Q, Qzp, wavefn, lambda_alpha)
    for n in np.arange(0,num_trj):
        Hq_dia=np.diag(np.diag(Hq[:,:,n]))
        Hq_offdia = Hq[:,:,n] - Hq_dia
        q_energy[n] = np.real(bra[:,n]@Hq_dia@wavefn[:,n][:,None])
        qc_energy[n] = np.real(bra[:,n]@Hq_offdia@wavefn[:,n][:,None])
    return q_energy, qc_energy

def init_classical_parallel(wavefn, lambda_al, num_trj, proc):
    Q = np.zeros((int(2*N), num_trj))
    P = np.zeros((int(2*N), num_trj))
    QaQb = np.zeros((int(2*N), int(2*N), num_trj))
    Qzp = np.zeros((int(2*N), num_trj))
    Pzp = np.zeros((int(2*N), num_trj))
    QaQbzp = np.zeros((int(2*N), int(2*N), num_trj))
    for n in np.arange(num_trj):
        Q[:, n], P[:, n] = Boltzmann(wavefn, lambda_al)
        QaQb[:, :, n] = np.outer(Q[:, n], Q[:, n])
        Qzp[:, n], Pzp[:, n] = Wigner(wavefn, lambda_al)
        QaQbzp[:, :, n] = np.outer(Qzp[:, n], Qzp[:, n])
    Qsplit=np.array(np.array_split(Q, proc, axis=1))
    Psplit=np.array(np.array_split(P, proc, axis=1))
    QaQbsplit=np.array(np.array_split(QaQb, proc, axis=2))
    Qsplitzp=np.array(np.array_split(Qzp, proc, axis=1))
    Psplitzp=np.array(np.array_split(Pzp, proc, axis=1))
    QaQbsplitzp=np.array(np.array_split(QaQbzp, proc, axis=2))
    return Qsplit, Psplit, QaQbsplit, Qsplitzp, Psplitzp, QaQbsplitzp

@jit(nopython=True,fastmath=True)
def get_QaQb(Q):
    QaQb=np.zeros((int(2*N), int(2*N), num_trj))
    for n in np.arange(num_trj):
        QaQb[:, :, n] = np.outer(Q[:, n], Q[:, n])
    return QaQb

#@jit(nopython=True,fastmath=True)
def initial_lambda():
    lambda_al = np.zeros((int(2*N), num_atom))
    for n in np.arange(0, num_atom):
        lambda_al[:, n] = np.sqrt(2/(epsilon*l)) * np.sin(np.pi * alpha * r_atom[n] / l)
    return lambda_al

#@jit(nopython=True,fastmath=True)
def initial_wavefn(ini_wavefn, num_trj, proc):
    wavefn = np.zeros((energy.size * num_atom, num_trj), dtype='complex128')
    for n in np.arange(0, num_atom):
        wavefn[n * 2, :] = ini_wavefn[n * 2]
        wavefn[n * 2 + 1, :] = ini_wavefn[n * 2 + 1]
        wavefn[n * 2 + 2, :] = ini_wavefn[n * 2 + 2]
    wavefnsplit=np.array(np.array_split(wavefn, proc, axis=1))
    return wavefnsplit

# @jit(nopython=True)
@ray.remote
def run_dyn(index, ini_wavefn, Q, P, QaQb, Qzp, Pzp, QaQbzp, lambda_al):
    wavefn = ini_wavefn.copy()
    lambda_al_grid = np.zeros([num_atom, int(2*N), num_trj])
    qE = np.zeros([num_atom, num_trj])
    qcE = np.zeros([num_atom, num_trj])
    for n in np.arange(0, num_atom):
        qE[n, :], qcE[n, :] = quantum_energy(np.array([wavefn[2 * n], wavefn[2 * n + 1], wavefn[2 * n + 2]]), Q, Qzp, lambda_al[:, n])
        lambda_al_grid[n, :, :] = np.transpose(np.tile(lambda_al[:, n], [num_trj, 1]))

    wavefn_square = np.real(wavefn * np.conjugate(wavefn))
    wavefnsquare_save = np.sum(wavefn_square, axis=1)
    QaQb_save = np.sum(QaQb, axis=2)
    QaQbzp_save = np.sum(QaQbzp, axis=2)
    Psquare_save = np.sum(P ** 2, axis=1)
    Psquare_savezp = np.sum(Pzp ** 2, axis=1)
    sys_ene = np.concatenate([[class_energy(Q, P)], [class_energy(Qzp, Pzp)], qE, qcE])
    sys_energy_save = np.sum(sys_ene, axis=-1)

    print('Calculating dynamics for the #'+ str(index) + ' trial on this processor...\n')
    print('Each trial has '+ str(num_trj) + ' trjectories')
    for t in steps:
        QF = 0
        QFzp = 0
        for n in np.arange(0, num_atom):
            now_wavefn = prop_Q_exact(np.array([wavefn[2*n], wavefn[2*n+1], wavefn[2 * n + 2]]), Q, Qzp, lambda_al[:,n], dt)
            wavefn[2*n] = now_wavefn[0]
            wavefn[2*n+1] = now_wavefn[1]
            wavefn[2 * n + 2] = now_wavefn[2]
            nowQF = Q_F(now_wavefn, lambda_al_grid[n,:,:])
            QFzp = QFzp + nowQF

        Pzp, Qzp = RK4(Pzp, Qzp, QFzp, dt)

        # save data
        if t % savestep == 0:
            wavefn_square = np.real(wavefn * np.conjugate(wavefn))
            wavefnsquare_save = np.vstack([wavefnsquare_save, np.sum(wavefn_square, axis=1)])

            # check system energy
            for n in np.arange(0, num_atom):
                qE[n, :], qcE[n, :] = quantum_energy(np.array([wavefn[2*n],wavefn[2*n+1], wavefn[2*n+2]]), Q, Qzp, lambda_al[:, n])
            sys_ene = np.concatenate([[class_energy(Q, P)], [class_energy(Qzp, Pzp)], qE, qcE])
            sys_energy_save = np.vstack([sys_energy_save, np.sum(sys_ene, axis=-1)])
            QaQb_save = np.dstack([QaQb_save, np.sum(get_QaQb(Q), axis=2)])
            QaQbzp_save = np.dstack([QaQbzp_save, np.sum(get_QaQb(Qzp), axis=2)])
            Psquare_save = np.vstack([Psquare_save, np.sum(P ** 2, axis=1)])
            Psquare_savezp = np.vstack([Psquare_savezp, np.sum(Pzp ** 2, axis=1)])

        # save end point
        if t == steps[-1]:
            Pend = P
            Pendzp = Pzp
            Qend = Q
            Qendzp = Qzp
            wavefnend = wavefn

    return wavefnsquare_save, QaQb_save, np.transpose(Psquare_save), sys_energy_save, Pend, \
           Qend, wavefnend, QaQbzp_save, Pendzp, Qendzp, np.transpose(Psquare_savezp)

# run parallel with ray from Alex
def parallel_run_ray(total_trj, proc):
    trials = int(total_trj/num_trj)
    r_ind = 0
    for run in range(0, int(trials/proc)):
        print('running RUN number '+str(run))
        lambda_al = initial_lambda()

        if 'status' in globals() and status=='RESTART':
            global calcdir
            old_calcdir = calcdir
            calcdir = calcdir + '/RESTART'
            print('new calculation directory: ' + str(calcdir))
            #ini_wavefn = np.sqrt(np.loadtxt(old_calcdir + '/rho.csv', delimiter=',')[-1])
            wavefnall = np.load(old_calcdir + '/wavefnend.npz')['wavefnend']
            qall = np.load(old_calcdir + '/Qend.npz')['Qend']
            qallzp = np.load(old_calcdir + '/Qendzp.npz')['Qendzp']
            pall = np.load(old_calcdir + '/Pend.npz')['Pend']
            pallzp = np.load(old_calcdir + '/Pendzp.npz')['Pendzp']
            wavefn = np.array(np.array_split(wavefnall, proc, axis=-1))
            q = np.array(np.array_split(qall, proc, axis=-1))
            Qzp = np.array(np.array_split(qallzp, proc, axis=-1))
            p = np.array(np.array_split(pall, proc, axis=-1))
            Pzp = np.array(np.array_split(pallzp, proc, axis=-1))
            QaQb = np.zeros((int(2*N), int(2*N), total_trj))
            QaQbzp = np.zeros((int(2 * N), int(2 * N), total_trj))
            for n in np.arange(total_trj):
                QaQb[:, :, n] = np.outer(qall[:, n], qall[:, n])
                QaQbzp[:, :, n] = np.outer(qallzp[:, n], qallzp[:, n])
            QaQb = np.array(np.array_split(QaQb, proc, axis=2))
            QaQbzp = np.array(np.array_split(QaQbzp, proc, axis=2))

        else:
            wavefn = initial_wavefn(ini_wavefn, total_trj, trials)
            q, p, QaQb, Qzp, Pzp, QaQbzp = init_classical_parallel(ini_wavefn, lambda_al, total_trj, trials)

        results = [run_dyn.remote(run * proc + i, wavefn[i], q[i], p[i],
                                  QaQb[i], Qzp[i], Pzp[i], QaQbzp[i], lambda_al) for i in range(proc)]
        for r in results:
            simRhoA, simQ1Q2, simP2, simE, Pend, Qend, wavefnend, QaQbzp_save, Pendzp, Qendzp, simP2zp = ray.get(r)
            if run == 0 and r_ind == 0:
                simEdat = np.zeros_like(simE)
                #simEcdat = np.zeros_like(simEc)
                simRhoAdat = np.zeros_like(simRhoA)
                #simRhoBdat = np.zeros_like(simRhoB)
                #simPdat = np.zeros_like(simP)
                #simQdat = np.zeros_like(simQ)
                simP2dat = np.zeros_like(simP2)
                simP2datzp = np.zeros_like(simP2zp)
                #simQ2dat = np.zeros_like(simQ2)
                #simNphdat = np.zeros_like(simNph)
                simQ1Q2dat = np.zeros_like(simQ1Q2)
                simQ1Q2datzp = np.zeros_like(QaQbzp_save)
                Penddat = Pend
                Penddatzp = Pendzp
                Qenddat = Qend
                Qenddatzp = Qendzp
                wavefnenddat = wavefnend
            else:
                Penddat = np.hstack([Penddat, Pend])
                Qenddat = np.hstack([Qenddat, Qend])
                wavefnenddat = np.hstack([wavefnenddat, wavefnend])
                Penddatzp = np.hstack([Penddatzp, Pendzp])
                Qenddatzp = np.hstack([Qenddatzp, Qendzp])

            simEdat += simE
            #simEcdat += simEc
            simRhoAdat += simRhoA
            #simRhoBdat += simRhoB
            #simPdat += simP
            #simQdat += simQ
            simP2dat += simP2
            simP2datzp += simP2zp
            #simQ2dat += simQ2
            #simNphdat += simNph
            simQ1Q2dat += simQ1Q2
            simQ1Q2datzp += QaQbzp_save
            r_ind += 1
    if path.exists(calcdir + '/E.csv'):
        simEdat += np.loadtxt(calcdir + '/E.csv', delimiter=',')
    # if path.exists(calcdir + '/Ec.csv'):
    #     simEcdat += np.loadtxt(calcdir + '/Ec.csv', delimiter=',')
    if path.exists(calcdir + '/rho.csv'):
        simRhoAdat += np.loadtxt(calcdir + '/rho.csv', delimiter=',')
    # if path.exists(calcdir + '/rhoB.csv'):
    #     simRhoBdat += np.loadtxt(calcdir + '/rhoB.csv', delimiter=',')
    if path.exists(calcdir + '/p2.csv'):
        simP2dat += np.loadtxt(calcdir + '/p2.csv', delimiter=',')
    if path.exists(calcdir + '/p2zp.csv'):
        simP2datzp += np.loadtxt(calcdir + '/p2zp.csv', delimiter=',')
    # if path.exists(calcdir + '/q2.csv'):
    #     simQ2dat += np.loadtxt(calcdir + '/q2.csv', delimiter=',')
    # if path.exists(calcdir + '/p.csv'):
    #     simPdat += np.loadtxt(calcdir + '/p.csv', delimiter=',')
    # if path.exists(calcdir + '/q.csv'):
    #     simQdat += np.loadtxt(calcdir + '/q.csv', delimiter=',')
    # # if path.exists(calcdir + '/Nph.csv'):
    #     simNphdat += np.loadtxt(calcdir + '/Nph.csv', delimiter=',')
    if path.exists(calcdir + '/q1q2.npz'):
        simQ1Q2dat += np.load(calcdir + '/q1q2.npz')['Q1Q2']
    if path.exists(calcdir + '/q1q2zp.npz'):
        simQ1Q2datzp += np.load(calcdir + '/q1q2zp.npz')['Q1Q2']
    if path.exists(calcdir + '/Pend.npz'):
        Penddat = np.hstack([Penddat, np.load(calcdir + '/Pend.npz')['Pend']])
    if path.exists(calcdir + '/Pendzp.npz'):
        Penddat = np.hstack([Penddatzp, np.load(calcdir + '/Pendzp.npz')['Pendzp']])
    if path.exists(calcdir + '/Qend.npz'):
        Qenddat = np.hstack([Qenddat, np.load(calcdir + '/Qend.npz')['Qend']])
    if path.exists(calcdir + '/Qendzp.npz'):
        Qenddatzp = np.hstack([Qenddatzp, np.load(calcdir + '/Qendzp.npz')['Qendzp']])
    if path.exists(calcdir + '/wavefnend.npz'):
        wavefnenddat = np.hstack([wavefnenddat, np.load(calcdir + '/wavefnend.npz')['wavefnend']])
    return simRhoAdat, simP2dat, simQ1Q2dat, simEdat, Penddat, Qenddat, \
           wavefnenddat, Penddatzp, Qenddatzp, simQ1Q2datzp, simP2datzp

def runCalc():
    print('Starting Calculation MF-QED with ' + str(num_atom) + ' atoms')
    print('Using '+str(proc)+' processors with '+str(num_trj)+' trajectories in each trial')
    print('Number of total trajectory is '+str(total_trj))
    #print(calcdir, '\n')
    start_time = time.time()
    #resEq, resEc, resRho, resP, resQ, resP2, resQ2, resNph, resQ1Q2 = parallel_run_ray(num_trj, proc)
    resRhoA, resP2, resQ1Q2, resE, Pend, Qend, \
    wavefnend, Pendzp, Qendzp, simQ1Q2datzp, resP2zp = parallel_run_ray(total_trj, proc)
    np.savetxt(calcdir + '/E.csv', resE, delimiter=',')
    # np.savetxt(calcdir + '/Ec.csv', resEc, delimiter=',')
    np.savetxt(calcdir + '/rho.csv', resRhoA, delimiter=',')
    # np.savetxt(calcdir + '/rhxoB.csv', resRhoB, delimiter=',')
    # np.savetxt(calcdir + '/p.csv', resP, delimiter=',')
    # np.savetxt(calcdir + '/q.csv', resQ, delimiter=',')
    np.savetxt(calcdir + '/p2.csv', resP2, delimiter=',')
    np.savetxt(calcdir + '/p2zp.csv', resP2zp, delimiter=',')
    # np.savetxt(calcdir + '/q2.csv', resQ2, delimiter=',')
    # np.savetxt(calcdir + '/Nph.csv', resNph, delimiter=',')
    np.savez_compressed(calcdir + '/q1q2.npz', Q1Q2=resQ1Q2)
    np.savez_compressed(calcdir + '/q1q2zp.npz', Q1Q2=simQ1Q2datzp)
    np.savez_compressed(calcdir + '/Pend.npz', Pend=Pend)
    np.savez_compressed(calcdir + '/Pendzp.npz', Pendzp=Pendzp)
    np.savez_compressed(calcdir + '/Qend.npz', Qend=Qend)
    np.savez_compressed(calcdir + '/Qendzp.npz', Qendzp=Qendzp)
    np.savez_compressed(calcdir + '/wavefnend.npz', wavefnend=wavefnend)
    end_time = time.time()
    print('finished. Running Wall Time = %10.3f second' % (end_time - start_time))
    return
