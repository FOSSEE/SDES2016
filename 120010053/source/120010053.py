# LC tank with Resistance

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode


def plot_styling(plot_var, title, labels, lims, x_var='None'):
	"""
		Styling of plot.
	"""
	plot_var.spines['right'].set_color('none')
	plot_var.spines['top'].set_color('none')
	plot_var.xaxis.set_ticks_position('bottom')
	plot_var.spines['bottom'].set_position(('data',0))
	plot_var.yaxis.set_ticks_position('left')
	plot_var.spines['left'].set_position(('data',0))
	
	plot_var.set_title(title, fontsize=20)

	xticks, step = np.linspace(lims[0][0], lims[0][1], 11, endpoint=True, retstep=True)
	if (x_var == 'f'):
		den = (lims[0][1] - lims[0][0])//step
		xtick_label = [r'$0$'] + [r'$%d\pi/%d$'%(i*2,den) for i in range(1,len(xticks)-1)] + [r'$2\pi$']
	else:
		xtick_label = xticks

	plot_var.set_xlim(lims[0])
	plot_var.set_xticks(xticks)
	plot_var.set_xticklabels(xtick_label)
	plot_var.set_xlabel(labels[0], fontsize=16)

	yticks = np.linspace(lims[1][0], lims[1][1], 10, endpoint=True)
	plot_var.set_ylim(min(0, 1.1*lims[1][0]), 1.1*lims[1][1])
	plot_var.set_yticks(yticks)
	plot_var.set_ylabel(labels[1], fontsize=16)
	
	for label in plot_var.get_xticklabels() + plot_var.get_yticklabels():
		label.set_fontsize(14	)
	
	return plot_var

def step_response(R,L,C,V):
	w = 1/np.sqrt(L*C)
	xi = R*w/(2*L)
	if (xi < 1):
		wd = w*np.sqrt(1-xi**2)
		fun = lambda t: V/wd*np.exp(-xi*w*t)*np.sin(wd*t)

	elif(xi == 1):
		fun = lambda t: V*t*np.exp(-w*t)/(L*w)

	else:
		s1 = -w*(xi + np.sqrt(xi**2 -1))
		s2 = -w*(xi - np.sqrt(xi**2 -1))
		coef = V*(s1*s2)/(s1 - s2)
		fun = lambda t: coef*(np.exp(s1*t) - np.exp(s2*t))

	return fun

def voltage_variation(R,L,C,V):
	w = 1/np.sqrt(L*C)
	xi = R*w/(2*L)
	curr = step_response(R,L,C,V)

	volt_res = lambda t: R*curr(t)
	volt_ind = lambda t, dt=1e-6 : L*(curr(t+dt) - curr(t-dt))/(2*dt)
	volt_cap = lambda t: V - volt_res(t) - volt_ind(t)

	title = "120010053: Evolution of voltage in components in step response"
	labels = ["Time $(s)$", "Voltage $(V)$"]
	time = np.linspace(0, 50, 500, endpoint=True)
	xlim = (0, 50)
	colors = "bgrcmyk"
	j = 0
	plt.figure(figsize=(10,7), dpi=77)
	ax = plt.subplot(111)

	res_v = [ volt_res(t) for t in time]
	ind_v = [ volt_ind(t) for t in time]
	cap_v = [ volt_cap(t) for t in time]
	current = [ 1000*curr(t) for t in time]

	ylim = (min(res_v + ind_v + cap_v), max(res_v + ind_v + cap_v))
	lims = (xlim, ylim)
	
	colors = "bgrcymk"
	leg_entry = [ "Voltage across resistor $V_R$ ", \
	              "Voltage across inductor $V_I$ ", \
	              "Voltage across capacitor $V_C$ ", \
	              "Current in the circuit $I(t)$" ]
	plt.plot(time, res_v, colors[0], linewidth='1.5', label=leg_entry[0])
	plt.plot(time, ind_v, colors[1], linewidth='1.5', label=leg_entry[1])
	plt.plot(time, cap_v, colors[2], linewidth='1.5', label=leg_entry[2])

	voltage_plot = plot_styling(ax, title, labels, lims)
	plt.legend(loc='upper right', frameon=False)
	plt.savefig("./Voltage_Variation.png", dpi=77)

	plt.figure(figsize=(10,7), dpi=77)
	ax = plt.subplot(111)
	plt.plot(time, current, colors[3], linewidth='2.0', label=leg_entry[3])

	title = "120010053: Current in the circuit"
	ylim = (min(current), max(current))
	lims = (xlim, ylim)
	current_plot = plot_styling(ax, title, labels, lims)
	plt.legend(loc='upper right', frameon=False)
	plt.savefig("./Current_Variation.png", dpi=77)

	#plt.show()

def transient_response_plot(L,C,xi_set,V):
	w = 1/np.sqrt(L*C)
	R_set = []
	for i in xi_set:
		R_set.append(2*w*i*L)
	
	title = "120010053: Transient Response of LC tank for unit voltage"
	labels = ['Time $(t)$', 'Current $I(t)$ (in mA)']
	
	time = np.linspace(0, 20, 500, endpoint=True)
	xlim = (0, 20)
	ylim = (-300, 600)
	lims = (xlim, ylim)
	colors = "bgrcmyk"
	j = 0
	plt.figure(figsize=(10,7), dpi=77)
	ax = plt.subplot(111)
	
	for i in xi_set:
		r = 2*L*w*i
		unit_step_resp = step_response(r,L,C,V)

		leg_entry = r'$ \xi = %s $' %i
		curr = []
		for t in time:
			curr.append(1000*unit_step_resp(t))
		
		lw = '1.0'
		if i == 1:
			lw = '3.0'
		plt.plot(time, curr, colors[j], linewidth=lw, label=leg_entry)
		j += 1
	
	transient_plot = plot_styling(ax, title, labels, lims)
	plt.legend(loc='upper right', frameon=False)
	
	plt.savefig("./Transients.png", dpi=77)

def series_LC_tank(R,L,C,V,f):
	"""
		Takes the elements of the series RLC circuit and gives out the
		amplitude of the current
	"""
	Xl = 2*np.pi*f*L
	Xc = 1/(2*np.pi*f*C)
	Z = np.sqrt(R**2 + (Xl - Xc)**2)
	curr = V/Z*1000
	
	return curr
	
	
def LC_tank_plot(R, L, C, V):
	"""
		Plot variation of source current in an LC tank with resistance
		with source frequency for fixed voltage
	"""
	freq = np.linspace(200, 0, 399, endpoint=False)
	
	plotsave = ""
	curr = []
		
	for i in range(len(freq)):
		curr.append(series_LC_tank(R, L, C, V, freq[i]))
	plotsave = "./SeriesVariation.png"
	xlim = (0, max(freq))
	ylim = (min(curr), max(curr))
		
	plt.figure(figsize=(10,7), dpi=77)
	ax = plt.subplot(111)
	title = "120010053, Variation of Amplitude of current with Source frequency"
	
	lims = [xlim, ylim]
	labels = ['Frequency (Hz)', 'Current (mA)']
	
	series_plot = plot_styling(ax, title, labels, lims)
	plt.plot(freq, curr, 'g', linewidth=2.0, linestyle='-')
	
	c = max(curr)
	t = freq[np.argmax(curr)]

	plt.plot([t,t],[0,c], color ='blue', linewidth=1.5, linestyle="--")
	plt.scatter([t,],[0,], 50, color ='blue')

	annotation = "Resonating frequency\n" r'$\omega \ \approx$ %.2f' %t
	plt.annotate(annotation, xy=(t, 0), xycoords='data',
             xytext=(+30, +70), textcoords='offset points',
             fontsize=14, arrowprops=dict(arrowstyle="->",
             connectionstyle="arc3,rad=.2"))

	
	plt.savefig(plotsave, dpi=77)
	return t 	# Indicates the resonating frequency as per the plot.
	#plt.show()

	
def main():
	source = 1
	res = 1
	cap = 10*10**-6
	ind = 100*10**-3
	resonance = 1/(2*np.pi*np.sqrt(cap*ind))
	
	peak =LC_tank_plot(res, ind, cap, source)
	
	
	with open('params.sty', 'w') as src:
		src.write("\\newcommand{\\serSource}{$%.2f \\ $V}\n" %source)
		src.write("\\newcommand{\\serRes}{$%.2f \\ \Omega $}\n" %res)
		src.write("\\newcommand{\\serCap}{$%.2f \\ \mu F$}\n" %(cap/10**-6))
		src.write("\\newcommand{\\serInd}{$%.2f \\ $mH}\n" %(ind/10**-3))
		src.write("\\newcommand{\\resonSer}{$%.3f \\ $Hz}\n" %resonance)
		src.write("\\newcommand{\\peakSer}{$%.3f \\ $Hz}\n" %peak)

	
	xi = [0.4, 0.6, 0.8, 1, 1.5, 2, 3]	# Various xi for transient response 
	transient_response_plot(1,1,xi,1)

	st_res = 2
	st_ind = 3
	st_cap = 1
	st_src = 2

	voltage_variation(st_res, st_ind, st_cap, st_src)

	with open('params.sty', 'a') as src:
		src.write("\\newcommand{\\Source}{$%.2f \\ $V}\n" %st_src)
		src.write("\\newcommand{\\Res}{$%.2f \\ \Omega $}\n" %st_res)
		src.write("\\newcommand{\\Capc}{$%.2f \\ F$}\n" %st_cap)
		src.write("\\newcommand{\\Ind}{$%.2f \\ $H}\n" %st_ind)


if __name__ == '__main__':
	main()
