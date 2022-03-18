import numpy as np
import sympy as sp
from sympy.physics.vector import dynamicsymbols
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from matplotlib import animation

def integrate(dv, ti, p):
	thw_list = dv
	m, l, g = p

	th=[]
	w=[]
	for i in range(m.size):
		th.append(thw_list[2*i])
		w.append(thw_list[2*i+1])

	sub={}
	for i in range(m.size):
		sub[M[i]]=m[i]
		sub[L[i]]=l[i]
		sub[theta[i]]=th[i]
		sub[theta_dot[i]]=w[i]
	sub['g']=g

	diffeq=[]
	for i in range(m.size):	
		diffeq.append(w[i])
		diffeq.append(alpha[i].subs(sub))

	print(ti)

	return diffeq

#---SymPy Derivation------------------------

N=3

L=sp.symbols('L0:%d'%N)
M=sp.symbols('M0:%d'%N)
g=sp.symbols('g')
t=sp.Symbol('t')
theta=dynamicsymbols('theta0:%d'%N)

X=[]
Y=[]
v_squared=[]
for i in range(N):
	X.append(L[0]*sp.sin(theta[0]))
	Y.append(-L[0]*sp.cos(theta[0]))
	for j in range(1,i+1):
		X[i]+=L[j]*sp.sin(theta[j])
		Y[i]+=-L[j]*sp.cos(theta[j])
	v_squared.append(X[i].diff(t,1)**2+Y[i].diff(t,1)**2)

T=M[0]*v_squared[0]
V=M[0]*Y[0]
for i in range(1,N):
	T+=M[i]*v_squared[i]
	V+=M[i]*Y[i]
T*=0.5
V*=g
T=sp.simplify(T)

Lg = T - V

diff_lg=[]
theta_dot=[]
theta_ddot=[]
for i in range(N):
	theta_dot.append(theta[i].diff(t,1))
	theta_ddot.append(theta[i].diff(t,2))
	dLdtheta=Lg.diff(theta[i],1)
	dLdthetadot=Lg.diff(theta_dot[i],1)
	ddtdLdthetadot=dLdthetadot.diff(t,1)
	diff_lg.append(ddtdLdthetadot-dLdtheta)

solution_set=sp.solve(diff_lg,theta_ddot)

alpha=[]
for i in range(N):
	alpha.append(sp.factor(sp.simplify(solution_set[theta_ddot[i]])))

#--------------------------------------------

#---functional working variables we will-----
#---use to sunbstitute into our abstract-----
#---SymPy derivation so that we can----------
#---integrate our differential equation.-----

gc=9.8
mass_i=1
mass_f=1
length_i=0.5
length_f=1
theta_0i=90
theta_0f=180
omega_0i=0
omega_0f=0

m=np.linspace(mass_i,mass_f,N)
l=np.linspace(length_i,length_f,N)
theta_0=np.linspace(theta_0i,theta_0f,N)
theta_0*=np.pi/180
omega_0=np.linspace(omega_0i,omega_0f,N)

p = m, l, gc

dyn_var=[]
for i in range(N):
	dyn_var.append(theta_0[i])
	dyn_var.append(omega_0[i])

tf = 5 
nfps = 30
nframes = tf * nfps
ta = np.linspace(0, tf, nframes)

thw = odeint(integrate, dyn_var, ta, args = (p,))

x=np.zeros((N,nframes))
y=np.zeros((N,nframes))
for i in range(nframes):
	sub={}
	sub[L[0]]=l[0]
	sub[theta[0]]=thw[i,0]
	x[0][i]=X[0].subs(sub)
	y[0][i]=Y[0].subs(sub)
	for j in range(1,N):
		sub={}
		sub[X[j-1]]=x[j-1][i]
		sub[L[j]]=l[j]
		sub[theta[j]]=thw[i,2*j]
		x[j][i]=X[j].subs(sub)
		sub.pop(X[j-1])
		sub[Y[j-1]]=y[j-1][i]
		y[j][i]=Y[j].subs(sub)

ke=np.zeros(nframes)
pe=np.zeros(nframes)
for i in range(nframes):
	sub={}
	for j in range(N):
		sub[M[j]]=m[j]
		sub[L[j]]=l[j]
		sub[theta[j]]=thw[i,2*j]
		sub[theta_dot[j]]=thw[i,2*j+1]
	ke[i]=T.subs(sub)
	sub={}
	for j in range(N):
		sub[M[j]]=m[j]
		sub[Y[j]]=y[j][i]
	sub[g]=gc
	pe[i]=V.subs(sub)
E=ke+pe
Emax=max(E)
E/=Emax
ke/=Emax
pe/=Emax

#--aesthetics-------------------------------

lmax,lmin=l.sum()*np.array([1.2,-1.2])
rad=0.05
phi=np.zeros((N,nframes))
dx=np.zeros((N,nframes))
dy=np.zeros((N,nframes))
dxb=np.zeros((N-1,nframes))
dyb=np.zeros((N-1,nframes))
for i in range(nframes):
	phi[0][i]=np.arccos(x[0][i]/l[0])
	dx[0][i]=rad*np.cos(phi[0][i])
	dy[0][i]=np.sign(y[0][i])*rad*np.sin(phi[0][i])
	for j in range(1,N):
		phi[j][i]=np.arccos((x[j][i]-x[j-1][i])/l[j])
		dx[j][i]=rad*np.cos(phi[j][i])
		dy[j][i]=np.sign(y[j][i]-y[j-1][i])*rad*np.sin(phi[j][i])
		dxb[j-1][i]=rad*np.sin(thw[i,2*j])
		dyb[j-1][i]=rad*np.cos(thw[i,2*j])

#--plot/animation---------------------------

fig, a=plt.subplots()

def run(frame):
	plt.clf()
	plt.subplot(211)
	plt.plot([0,x[0][frame]-dx[0][frame]],[0,y[0][frame]-dy[0][frame]],color='xkcd:cerulean')
	circle=plt.Circle((x[0][frame],y[0][frame]),radius=rad,fc='xkcd:red')
	plt.gca().add_patch(circle)
	for i in range(1,N):
		plt.plot([x[i-1][frame]+dxb[i-1][frame],x[i][frame]-dx[i][frame]],[y[i-1][frame]-dyb[i-1][frame],y[i][frame]-dy[i][frame]],color='xkcd:cerulean')
		circle=plt.Circle((x[i][frame],y[i][frame]),radius=rad,fc='xkcd:red')
		plt.gca().add_patch(circle)
	plt.title("The N-Tuple Pendulum (N=%i)"%N)
	ax=plt.gca()
	ax.set_aspect(1)
	plt.xlim([lmin,lmax])
	plt.ylim([lmin,lmax])
	ax.xaxis.set_ticklabels([])
	ax.yaxis.set_ticklabels([])
	ax.xaxis.set_ticks_position('none')
	ax.yaxis.set_ticks_position('none')
	ax.set_facecolor('xkcd:black')
	plt.subplot(212)
	plt.plot(ta[0:frame],ke[0:frame],'xkcd:red',lw=0.5)
	plt.plot(ta[0:frame],pe[0:frame],'xkcd:cerulean',lw=0.5)
	plt.plot(ta[0:frame],E[0:frame],'xkcd:bright green',lw=1.0)
	plt.xlim([0,tf])
	plt.title("Energy (Rescaled)")
	ax=plt.gca()
	ax.legend(['T','V','E'],labelcolor='w',frameon=False)
	ax.set_facecolor('xkcd:black')

ani=animation.FuncAnimation(fig,run,frames=nframes)
#writervideo = animation.FFMpegWriter(fps=nfps)
#ani.save('triple_pendulum.mp4', writer=writervideo)
plt.show()


