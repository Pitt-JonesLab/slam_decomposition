import json
from matplotlib import pyplot as plt
import numpy as np


datapath = "/home/evm9/decomposition_EM/data/Q1_11.07mA_snailPump_length0.5_prepareE_False_freqG_9070_freqC_695-11_processed.json"

dd = json.load(open(datapath))
g2_conv = dd["g2_conv"]
g2_gain = dd["g2_gain"]
g_pct = np.array(dd["g_pct"])


plt.figure()
plt.pcolormesh(g2_conv, g2_gain, g_pct.T, shading='auto', cmap='RdBu', vmin=0, vmax=1)
plt.xlabel(f"g2eff_conv (MHz)")
plt.ylabel(f"g2eff_gain (MHz)")

# plt.xticks(np.linspace(0, 50, 6), map(str, np.linspace(0, 50, 6, dtype=int)))
# plt.yticks(np.linspace(0, 15, 4), map(str, np.linspace(0, 15, 4, dtype=int)))


cbar = plt.colorbar()
cbar.set_label("g pct", rotation=90)
cbar.set_ticks([0.0, 0.5, 1.0])

# namign variables to match old version
glist = g_pct
c_ampArray = g2_conv
g_ampArray = g2_gain

# step 1 data cleaning
glist_temp = glist.copy().T
# iterate through each column
for i in range(glist_temp.shape[1]):
    # find the last value close to 0.5
    idx = np.where(np.isclose(glist_temp[:,i], 0.5, atol=.05))[0]
    if len(idx) > 0:
        idx = idx[-1]
    else:
        # set all values to 0
        glist_temp[:,i] = 0
        continue
    # set all values before that to 0
    glist_temp[:idx,i] = 0
    # set that value to 1
    glist_temp[idx,i] = 1
    # set every value after that to 0
    glist_temp[idx+1:,i] = 0


# remove the obvious outlier
glist_temp[0, 68] = 0


#step 2 reduce dimensionality
coords = []
for i in range(glist_temp.shape[0]):
    for j in range(glist_temp.shape[1]):
        if glist_temp[i,j] == 1:
            coords.append([c_ampArray[j], g_ampArray[i]])

coords.sort(key=lambda x: x[0])


# add in point (1,0) to make sure bounded
coords.append([coords[-1][0],0])

#step 2.5 arbitrary scaling to convert DAC to Hamiltonian (rad/s) ?
real_scaling = 1
conv_slope =1
gain_slope = 1
# slopes are 1 because already scaled in new data

if real_scaling:
    #decide to scale so x and y intercept are near pi/2
    #scale x axis
    x = [x[0]*conv_slope for x in coords] #0.64*np.pi
    # scale y axis
    y = [x[1]*gain_slope for x in coords] #1.7*np.pi

    # step 2.6 normalization so the maximum intercept is pi/2
    # find max between x- and y-axis intercept
    max_intercept = np.max([np.abs(x[-1]), np.abs(y[0])])
    # scale x and y
    x = [x[i]/max_intercept*np.pi/2 for i in range(len(x))]
    y = [y[i]/max_intercept*np.pi/2 for i in range(len(y))]

else:
    #scale x axis
    x = [x[0]*0.64*np.pi for x in coords] #
    # scale y axis
    y = [x[1]*1.7*np.pi for x in coords] #


from scipy.interpolate import UnivariateSpline
N=800
s = spline = UnivariateSpline(x, y, s=0.001)
xs = np.linspace(0, max(x), N)


# %%
#rewrite ConversionGainGate to have a cost function that is dynamic in the parameters
from src.utils.custom_gates import ConversionGainGate
class SpeedLimitedGate(ConversionGainGate):
    def __init__(self, p1, p2, g1, g2, t_el, speed_limit_function=spline):
        self.g1 = g1 #conversion
        self.g2 = g2 #gain
        self.t_el = t_el
        self.slf = speed_limit_function
        self.saved_cost = -1
        super().__init__(p1, p2, g1, g2, t_el)
        # XXX can only assign duration after init with real values
        if all([isinstance(p, (int, float)) for p in self.params]):
            self.duration = self.cost()
    
    @classmethod
    def from_gate(cls, ConversionGainGate):
        return cls(*ConversionGainGate.params, speed_limit_function=spline)
        
    def cost(self):
        if self.saved_cost >= 0:
            return self.saved_cost

        assert not (self.g1 == 0 and self.g2 ==0)
        #norm = np.pi/2 # TODO move norm from preprocessing into here
        # get upper bound of g terms from speed limit data
        xs = np.linspace(0, np.pi/2, N) #XXX, danger! assumes the max intercept was on the x-axis (assumes y-axis intercept is np.pi/2)

        if self.g1 == 0: #avoid case when ratio is undefined by only scaling 
            idx = 0
            scaled_g1, scaled_g2 = float(xs[idx]), float(self.slf(xs[idx]))
        else:
            ratio = self.g2/self.g1 * xs
            tol = .001
            while np.argwhere(np.abs(ratio - self.slf(xs)) < tol).size == 0:
                tol += .001
            idx = max(np.argwhere(np.abs(ratio - self.slf(xs)) < tol))[0]
            scaled_g1, scaled_g2 = float(xs[idx]), float(ratio[idx]) #conversion, gain
        
        #inversely scale time for new g terms
        # should be the same unless either g1 or g2 is 0
        if self.g1 == 0:
            scale = scaled_g2/self.g2
        else:
            scale = scaled_g1/self.g1

        # if scale < 1 means we are decreasing strength, so increase time
        scaled_t = self.t_el / scale

        #cost is duration of gate
        self.c = scaled_t
        return scaled_t

# # %%
# c = ConversionGainGate(0, 0, .25*np.pi, .25*np.pi, 1)
# c.cost()

# # %%
# c = SpeedLimitedGate(0, 0, .25*np.pi, .25*np.pi, 1, speed_limit_function=s)
# c.cost()

# # %%
# s2 = lambda x: -x + np.pi/2
# d = SpeedLimitedGate(0, 0, .25*np.pi, .25*np.pi, 1, speed_limit_function=s2)
# d.cost()
# #rounding error when finding intersection?

# # %%
# # step 3 univariate spline
# from scipy.interpolate import UnivariateSpline
# N=400
# s2 = lambda x: -x + np.pi/2
# xs = np.linspace(0, np.pi/2, N)
# # plt.figure()
# # plt.plot(xs, s2(xs), '-')
# # plt.ylabel("gain amp(DAC)")
# # plt.xlabel('conv amp(DAC)')
# # plt.xlim(0,max(xs))
# # plt.ylim(0,max(s(xs)))

# #plot cnot gate
# gate_c = 0.25*np.pi
# gate_g = 0.25*np.pi
# if gate_c == 0:
#     #plt.plot([0, 0], [0, max(s(xs))], 'ro')
#     pass
# else:
#     ratio = gate_g/gate_c * xs
#     #plt.plot(xs, ratio, 'r--')

# if gate_c == 0:
#     idx = 0
#     print(xs[idx], s2(xs[idx]))
# else:
#     idx = max(np.argwhere(np.abs(ratio - s2(xs)) < 0.01))
#     print(xs[idx], ratio[idx])
# # plt.plot(xs[idx], s2(xs[idx]), 'ro')
# # plt.show()

# # %%
# np.pi/2


