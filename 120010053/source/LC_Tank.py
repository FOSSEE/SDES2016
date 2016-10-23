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

def transient_response_plot():
	L = 1
	C = 1
	w = 1
	xi = [0.4, 0.6, 0.8, 1, 2, 3]
	
	title = "120010053: Transient Response of LC tank for unit voltage"
	labels = ['Time $(t)$', 'Current $I(t)$ mA']
	
	fun = lambda t: t
	time = np.linspace(0, 20, 500, endpoint=True)
	xlim = (0, 20)
	ylim = (-300, 600)
	lims = (xlim, ylim)
	colors = "bgrcmyk"
	j = 0
	plt.figure(figsize=(10,7), dpi=77)
	ax = plt.subplot(111)
	
	for i in xi:
		 if(i < 1):
		 	wd = np.sqrt(1-i**2)
		 	#fun1 = lambda t: np.exp(-i*t)*np.cos(wd*t)
		 	#fun = lambda t: (1 + wd**2)*np.sqrt(2)*np.exp(-i*t)*np.sin(wd*t)/(2 - i)
		 	fun = lambda t: 1/wd*np.exp(-i*t)*np.sin(wd*t)
		 elif(i == 1):
		 	fun1 = lambda t: np.exp(-i*t)
		 	fun = lambda t: t*fun1(t)
		 else:
		 	s1 = -i + np.sqrt(i**2 -1)
		 	s2 = -i - np.sqrt(i**2 -1)
		 	coef =  s1*s2/(s1 - s2)
		 	fun = lambda t: coef*(np.exp(s1*t) - np.exp(s2*t))

		 leg_entry = r'$ \xi = %s $' %i
		 curr = []
		 for t in time:
		 	curr.append(1000*fun(t))
		 
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
	
	
def LC_tank_plot(R, L, C, V, circuit = "series"):
	"""
		Plot variation of source current in an LC tank with resistance
		with source frequency for fixed voltage
		Optional arguments:
			circuit indicates the type of circuit connection.
			For now, circuit can be either series or parallel
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
	return t
	#plt.show()

	
def main():
	ser_source = 1
	ser_res = 1
	ser_cap = 10*10**-6
	ser_ind = 100*10**-3
	resonance_ser = 1/(2*np.pi*np.sqrt(ser_cap*ser_ind))
	
	peak_ser =LC_tank_plot(ser_res, ser_ind, ser_cap, ser_source)
	
	
	with open('params.sty', 'w') as src:
		src.write("\\newcommand{\\serSource}{$%.2f \\ $V}\n" %ser_source)
		src.write("\\newcommand{\\serRes}{$%.2f \\ \Omega $}\n" %ser_res)
		src.write("\\newcommand{\\serCap}{$%.2f \\ \mu F$}\n" %(ser_cap/10**-6))
		src.write("\\newcommand{\\serInd}{$%.2f \\ $mH}\n" %(ser_ind/10**-3))
		src.write("\\newcommand{\\resonSer}{$%.3f \\ $Hz}\n" %resonance_ser)
		src.write("\\newcommand{\\peakSer}{$%.3f \\ $Hz}\n" %peak_ser)

	transient_response_plot()	

if __name__ == '__main__':
	main()
	
