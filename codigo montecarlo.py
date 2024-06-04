
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from tqdm import tqdm
import os
import glob
import time
from scipy.interpolate import interp1d
start_time = time.time()

################################################################################################################################  
'Selección de graficas'
################################################################################################################################
calcular = False 

graficaspos = False       # Grafica las posiciones de vacantes e intesticiales
graficaN = True 
graficatime = True
times = True
datos = True
datost = True

################################################################################################################################  
'Selección de parametros'
################################################################################################################################

T0 = 80.              # Temperatura inicial en kelvin 
Tf = 150.             # Temperatura final 
t0,tf = 0,300         # Intervalo de tiempos para cada temperatura
N_frenken = 1891      # Numero de pares de Frenkel que se generan
semilla = 3           # 6,14,48,65,73,1,3 
N_frenken_graf = 1891 # Numero de pares de frenkel para el grafico
y_graf = 7            # Numero de semillas que tomara

################################################################################################################################  
'Valores conocidos'
################################################################################################################################

a_W = 2.87            # Parametro de red en Amstrongs 
rfren = (4.*a_W)**0.5 # Radio de Frenkel 
rho = 1*1e-2        # Concentracion por 1/nm^3
rhoA = 1*1e-5       # Concentracion por 1/A^3
Fv = 1e13             # Saltos por segundo para vacantes 
Emv = 0.67            # Energia de migración en eV para vacantes 
Fi = 1e13             # Saltos por segundo para intersticiales 
Emi = 0.34           # Energia de migración en eV para intersticiales 
K = 8.617333262e-5    # Constante de Boltzman
rcap = (3.3* a_W)**0.5     # Radio de captura
np.random.seed(semilla) 

################################################################################################################################
################################################################################################################################
"Definicion de funciones"
################################################################################################################################
################################################################################################################################

# Numba , Ziton ,dusk
################################################################################################################################
'Funcion auxiliar para graficar y dar resultados '
################################################################################################################################

class MathTextSciFormatter(mticker.Formatter):
    def __init__(self, fmt="%1.2e"):
        self.fmt = fmt
    def __call__(self, x, pos=None):
        s = self.fmt % x
        decimal_point = '.'
        positive_sign = '+'
        tup = s.split('e')
        significand = tup[0].rstrip(decimal_point)#np.add.reduce((I-V)**2, axis=0)
        sign = tup[1][0].replace(positive_sign, '')
        exponent = tup[1][1:].lstrip('0')
        if exponent:
            exponent = '10^{%s%s}' % (sign, exponent)
        if significand and exponent:
            s =  r'%s{\times}%s' % (significand, exponent)
        else:
            s =  r'%s%s' % (significand, exponent)
        return "${}$".format(s)
    

################################################################################################################################
'Funciones principales'
################################################################################################################################

def longitud(N,R):
    L = (N*2/R)**(1/3)
    return L 

def generador(N,R,a,f):
    l = longitud(N, R)
    V = np.random.uniform(-l/2, l/2, size=(3,N)).astype(np.float32)  
    angs = np.random.uniform(0, np.pi, size=(2*N)).astype(np.float32)
    theta,phi = angs[0:N], angs[N:]
    I = V + f*np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])
    return l,V.T,I.T

def Temparray(T_ini,T_fin):
    T_list = []
    T_inc = T_ini
    while T_inc <= T_fin:
        T_list.append(T_inc)
        T_inc += 0.03*T_inc
    return np.array(np.round(T_list, decimals=4)).astype(np.float32)
    
def prob(pf,E,k,T):
    return pf*np.exp(-E/(k*T))

# Vacantes, intersticiales, saltos por segundo (Fv), Energia migracion (v e i)
# C. Boltzman, t inicial , parametro de red, factor de Frenkel
def evolucion(Vl, Il, f, Ev, Ei, k, T, a, ff, rc,L):
    Va, In = Vl, Il
    Nvi = [len(Va)]
    t_acum, t = [0], 0
    t_paso,Nvi_paso = [0],[len(Va)]
    p = 0 
    # Agregar tqdm para mostrar la barra de progreso
    for x in tqdm(T): 
        t = 0
        pv, pi = prob(f, Ev, k, x), prob(f, Ei, k, x)
        #print('x =',x,'pv = ',pv,'pi =',pi)
        nac = Nvi[-1]
        Limit = L*np.ones(3)/2
        while t <= tf: 
            p += 1 
            Rv, Ri = nac * pv, nac * pi
            R = Rv + Ri
            th, phi, n, j =  np.random.rand(4).astype(np.float32)
            th, phi = th * np.pi, phi * 2 * np.pi
            ang = np.array([np.sin(th) * np.cos(phi), np.sin(th) * np.sin(phi), np.cos(th)]).astype(np.float32)
            n = n * R
            if n < Rv: 
                j = int(j * nac)-1
                Va[j] = Va[j] + ff * ang 
                dentrosup = Va[j]>Limit 
                dentroinf = -1*Limit> Va[j]
                
                if np.sum(dentrosup) + np.sum(dentroinf) != 0 :
                    Va[j] = Va[j] - (L)*(dentrosup) + (L)*(dentroinf)
                    
                idis = np.add.reduce((In - np.tile(Va[j], (nac, 1)))**2, axis=1)
                idis = idis < rc
                l = np.where(idis)[0]
                if l.size > 0:
                    Va = np.delete(Va, l, 0).astype(np.float32)
                    In = np.delete(In, l, 0).astype(np.float32)
                    nac += -1
                    Nvi_paso.append(nac)
                    t_paso.append(t_acum[-1] + t)
                    
            elif n >= Rv:
                j = int(j * nac) -1
                In[j] = In[j] + ff * ang 
                dentrosup = In[j]>Limit 
                dentroinf = -1*Limit> In[j]
                
                if np.sum(dentrosup) + np.sum(dentroinf) != 0 :
                    In[j] = In[j] - (L)*(dentrosup) + (L)*(dentroinf)
                    
                vdis = np.add.reduce((Va - np.tile(In[j], (nac, 1)))**2, axis=1)
                vdis = vdis < rc
                l = np.where(vdis)[0]
                if l.size > 0:
                    Va = np.delete(Va, l, 0).astype(np.float32)
                    In = np.delete(In, l, 0).astype(np.float32)
                    nac += -1*len(l)
                    Nvi_paso.append(nac)
                    t_paso.append(t_acum[-1] + t)
            t += -np.log(np.random.rand()) / R#rc
        t_acum.append(t_acum[-1] + t)
        Nvi.append(int(nac))
    return Nvi,t_acum,Va,In,t_paso,Nvi_paso
  
    
def guardar_tem(Tem,ti,Nd,nombre):
    fichero = open('Datos_{0}_{3}_semilla{2}_{1}.txt'.format(N_frenken,Tf,semilla,nombre),'w')
    fichero.write('Nº defectos para semilla {0}'.format(semilla)+'\n')
    fichero.write('Paso   Nº Defectos   t[s]       T[K]')
    fichero.write('\n')
    for w in range(len(Tem)):
        fichero.write(' {0}         {3}        {2}       {1}'
                      .format(w,np.round(Tem[w]*1000)/1000,np.round(ti[w+1]),Nd[w+1]))
        fichero.write('\n')
    fichero.close()
    return None 

def guardar_tiempo(ti,Nd,nombre):
    fichero = open('Datos_{0}_{3}_semilla{2}_{1}.txt'.format(N_frenken,Tf,semilla,nombre),'w')
    fichero.write('Nº defectos para semilla {0}'.format(semilla)+'\n')
    fichero.write('Paso   Nº Defectos   t[s] ')
    fichero.write('\n')
    for w in range(len(ti)):
        fichero.write(' {0}         {1}        {2}     '
                      .format(w,Nd[w],np.round(ti[w]),))
        fichero.write('\n')
    fichero.close()
    return None 


def lector(nombre):
    Nd,tim,Tem = [],[],[]
    f = open(nombre,'r')
    ley = f.readline()
    next(f)
    for linea in f:
        ele = linea.split()
        Nd.append(float(ele[1]))
        tim.append(float(ele[2]))
        Tem.append(float(ele[3]))
    return Nd,tim,Tem,ley

def lectort(nombre):
    Nd,tim = [],[]
    f = open(nombre,'r')
    ley = f.readline()
    next(f)
    for linea in f:
        ele = linea.split()
        Nd.append(float(ele[1]))
        tim.append(float(ele[2]))
    return Nd,tim,ley
    
################################################################################################################################
################################################################################################################################
"Calculos"
################################################################################################################################
################################################################################################################################

if calcular == True:
    # Longitud de la celda, lista de vacantes e intesticiales
    L,v_list,i_list = generador(N_frenken,rhoA,a_W,rfren) 
    T_list = Temparray(T0,Tf)  
    Ndef, t_tot,vac,inte,t_evol,N_evol = evolucion(v_list,i_list,Fv,Emv,Emi,K,T_list,a_W,rfren,rcap,L)
    guardar_tem(T_list,t_tot,Ndef,'Temperatura') 
    guardar_tiempo(t_evol,N_evol,'Time') 
    datos = False
    datost = False

if calcular == False:
    N_def_tot,t_tot_tot,T_list_tot,leyendas = [],[],[],[]
    directorio = os.getcwd()
    patron = directorio + '/Datos_{}_Temperatura*'.format(N_frenken_graf)
    nombres = glob.glob(patron)
    for nom in nombres:
        try:
            Ndef, t_tot,T_list,leyen = lector(nom)
            leyendas.append(leyen) 
            N_def_tot.append(Ndef)
            T_list_tot.append(T_list)
            t_tot_tot.append(t_tot)
            datos = True 
        except:
            print('No hay datos para la grafica de temperatura')
            datos = False

if calcular == False:
    N_def_time,t_list,leyendas_t = [],[],[]
    directorio = os.getcwd()
    patron2 = directorio + '/Datos_{}_Time*'.format(N_frenken_graf)
    nombres2 = glob.glob(patron2)
    for nom in nombres2:
        try:
            Ndef,t_tot,leyen = lectort(nom)
            leyendas_t.append(leyen) 
            N_def_time.append(Ndef)
            t_list.append(t_tot)
            datost = True             
        except:
            print('No hay datos para la grafica de tiempo')
            datost = False       
        
#########################size#######################################################################################################
################################################################################################################################
"Graficas"
################################################################################################################################
################################################################################################################################

if graficaspos == True: 
    fig = plt.figure()
    axs = fig.add_subplot(111, projection='3d')
    axs.scatter(v_list[0, :], v_list[1, :], v_list[2, :], color='blue', label='Posiciones de las vacantes')
    axs.scatter(i_list[0, :], i_list[1, :], i_list[2, :], color='red', label='Posiciones de las instersticiales')
    axs.set_xlabel(r'Eje X [$\AA$]')
    axs.set_ylabel(r'Eje Y [$\AA$]')
    axs.set_zlabel(r'Eje Z [$\AA$]')
    axs.legend()
    plt.show()

if graficaN == True  and datos == True: 
    fig = plt.figure('N(T)')
    fig.gca().xaxis.set_major_formatter(MathTextSciFormatter("%1.1e"))
    fig.gca().yaxis.set_major_formatter(MathTextSciFormatter("%1.1e"))
    ax1 = plt.gca()  
    ax1.set_title(r'Nº defectos en funcion de T', fontsize=20)
    ax1.set_ylabel(r'Nº defectos', fontsize=18)
    ax1.set_xlabel('T [K]', fontsize=18)
    for y in range(np.min([len(N_def_tot),y_graf])):
        line, = ax1.plot(T_list_tot[y], N_def_tot[y], linewidth=1.5, label=leyendas[y])
    ax1.grid(True, linestyle='--', alpha=0.8)
    ax1.axis([T0,Tf,500.,2000])
    ax1.legend(fontsize=8)
    plt.show()

if graficaN == True  and datost == True: 
    fig2 = plt.figure('dN(T)')
    fig2.gca().xaxis.set_major_formatter(MathTextSciFormatter("%1.1e"))
    fig2.gca().yaxis.set_major_formatter(MathTextSciFormatter("%1.1e"))
    ax2 = plt.gca()  
    ax2.set_title(r' Derivada del Nº defectos en funcion de T', fontsize=20)
    ax2.set_ylabel(r'$\frac{dN}{dT}$', fontsize=18)
    ax2.set_xlabel('T [K]', fontsize=18)
    for y in range(np.min([len(N_def_tot),y_graf])):
        NN_def_tem_arr = np.array(N_def_tot[y][1:])
        tem_list_arr = np.array(T_list_tot[y][1:])
        ftem = interp1d(tem_list_arr, NN_def_tem_arr)
        T_smooth = np.linspace(tem_list_arr.min(), tem_list_arr.max(), len(tem_list_arr))
        NN_def_time_smooth = ftem(T_smooth)
        dn_def_tem= np.gradient(NN_def_time_smooth, T_smooth)
        line2, = ax2.plot(T_smooth, -1*dn_def_tem, linewidth=1.5, label=leyendas[y])
    ax2.grid(True, linestyle='--', alpha=0.8)
    ax2.axis([T0,Tf,0.,80.])
    ax2.legend(fontsize=8)
    plt.show()

if graficaN == True  and datost == True and datos:
    fig5, ax5 = plt.subplots(1, 2,figsize=(12, 6))
    #fig5.suptitle('Evolucion con la temperatura', fontsize=22)
    for y in range(np.min([len(N_def_tot),y_graf])):
        line1, = ax5[0].plot(T_list_tot[y], N_def_tot[y], linewidth=1.5, label=leyendas[y])
    ax5[0].set_title(r'Nº defectos en funcion de T', fontsize=16)
    ax5[0].set_ylabel(r'Nº defectos', fontsize=18)
    ax5[0].set_xlabel('T [K]', fontsize=18)
    ax5[0].set_title(r'Defectos en función del tiempo' , fontsize=18)
    ax5[0].axis([T0,Tf,500.,2000.])
    ax5[0].grid(True, linestyle='--', alpha=0.8)
    ax5[0].legend()
    
    
    ax5[1].set_title(r' Derivada del Nº defectos en funcion de T', fontsize=16)
    ax5[1].set_ylabel(r'$\frac{dN}{dT}$', fontsize=18)
    ax5[1].set_xlabel('T [K]', fontsize=18)
    for y in range(np.min([len(N_def_tot),y_graf])):
        NN_def_tem_arr = np.array(N_def_tot[y][1:])
        tem_list_arr = np.array(T_list_tot[y][1:])
        ftem = interp1d(tem_list_arr, NN_def_tem_arr)
        T_smooth = np.linspace(tem_list_arr.min(), tem_list_arr.max(), len(tem_list_arr))
        NN_def_time_smooth = ftem(T_smooth)
        dn_def_tem= np.gradient(NN_def_time_smooth, T_smooth)
        line2, = ax5[1].plot(T_smooth, -1*dn_def_tem, linewidth=1.5, label=leyendas[y])
    ax5[1].grid(True, linestyle='--', alpha=0.8)
    ax5[1].axis([T0,Tf,0.,80.])
    ax5[1].legend()
    plt.show()

if graficatime == True  and datost == True: 
    fig3 = plt.figure('N(t)')
    fig3.gca().xaxis.set_major_formatter(MathTextSciFormatter("%1.1e"))
    fig3.gca().yaxis.set_major_formatter(MathTextSciFormatter("%1.1e"))
    ax3 = plt.gca()  
    ax3.set_title(r'Nº defectos en funcion del tiempo', fontsize=20)
    ax3.set_ylabel(r'Nº defectos', fontsize=18)
    ax3.set_xlabel('Tiempo [s]', fontsize=18)
    for y in range(np.min([len(N_def_tot),y_graf])):
        line3, = ax3.plot(t_list[y], N_def_time[y], linewidth=1.5, label=leyendas[y])
    ax3.grid(True, linestyle='--', alpha=0.8)
    #ax3.axis([T0,Tf,0,N_frenken])
    ax3.legend(fontsize=8)
    plt.show()


if graficaN == True  and datost == True: 
    fig4 = plt.figure('dN(t)')
    fig4.gca().xaxis.set_major_formatter(MathTextSciFormatter("%1.1e"))
    fig4.gca().yaxis.set_major_formatter(MathTextSciFormatter("%1.1e"))
    ax4 = plt.gca()  
    ax4.set_title(r' Derivada del Nº defectos en funcion del tiempo ', fontsize=20)
    ax4.set_ylabel(r'$\frac{dN}{dT}$', fontsize=18)
    ax4.set_xlabel('T [K]', fontsize=18)
    for y in range(np.min([len(N_def_tot),y_graf])):
        NN_def_time_arr = np.array(N_def_time[y][1:])
        t_list_arr = np.array(t_list[y][1:])
        ft = interp1d(t_list_arr, NN_def_time_arr)
        t_smooth = np.linspace(t_list_arr.min(), t_list_arr.max(), len(t_list_arr))
        NN_def_time_smooth = ft(t_smooth)
        dn_def_time = np.gradient(NN_def_time_smooth, t_smooth)
        line4, = ax4.plot(t_smooth, -1*dn_def_time, linewidth=1.5, label=leyendas[y])
    
    ax4.grid(True, linestyle='--', alpha=0.8)
    ax4.legend(fontsize=8)
    plt.show()
 
#################################################################################################################################################
'Calculamos tiempo de ejecucion'
#################################################################################################################################################
if times == True: 
    end_time = time.time()
    execution_time = end_time - start_time
    print("Tiempo de ejecución:", execution_time, "segundos")
    



    
