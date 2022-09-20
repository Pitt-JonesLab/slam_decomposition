import matplotlib.pyplot as plt
import numpy as np
# Go to Chao's HaPiCodes repo under pulseAndQueueDev branch to install these data processing modules
from HaPiCodes.data_process import fittingAndDataProcess as f

from HaPiCodes.data_process import loadH5IntoIQData
from HaPiCodes.data_process import postSelectionProcess as psdp
from HaPiCodes.data_process.saveData_dev import getSweepDictAndMSMTInfo
from tqdm import tqdm

# use Hatlab VPN to connect to this network drive where the data is saved
saveDir = r"H:\Data\SNAIL_Pump_Limitation\Q_SNAIL_GC\SweepTwoAmps\2022-09-11\Q3_-1.15mA-C_width_1300-G_width_1300-cPumpParams_{width_1300-ssbFreq_0.1}-gPumpParams_{width_1300-ssbFreq_0.1}\\"

sweepAxes, msmtInfoDict = getSweepDictAndMSMTInfo(saveDir)

c_ampArray = np.linspace(0, 0.8, 81)
g_ampArray = np.linspace(0, 0.8, 81)
initPulse = "off"
cPumpArgs={"ssbFreq": 0.1}
gPumpArgs={"ssbFreq": 0.1}


Ilist = np.zeros([len(g_ampArray), len(c_ampArray)])
Qlist = np.zeros([len(g_ampArray), len(c_ampArray)])
glist = np.zeros([len(g_ampArray), len(c_ampArray)])
sel = msmtInfoDict.get("sel", 0)
geLocation = msmtInfoDict.get('geLocation', "AutoFit")
gefLocation = msmtInfoDict.get('gefLocation', "AutoFit")



if __name__ == '__main__':

    for i, iAmp in enumerate(tqdm(g_ampArray)):
        gPumpArgs["amp"] = iAmp

        IQdata, params = loadH5IntoIQData(saveDir, f'g_Amp{np.round(iAmp, 9)}GHz')
        Id, Qd = IQdata.I_rot, IQdata.Q_rot

        if sel:
            selData = psdp.PostSelectionData(Id, Qd, msmtInfoDict, [1, 0], plotGauFitting=0, geLocation=geLocation)
            selMask = selData.mask_g_by_circle(sel_idx=0, plot=0)
            # selMask = (np.zeros(selMask.shape)+1).astype(bool) #----------------------All true mask-------------------
            I_vld, Q_vld = selData.sel_data(selMask, plot=0)
            g_pct = selData.cal_g_pct(plot=0)
            glist[i] = g_pct

            # plt.figure('cutline')
            # plt.plot(c_ampArray, glist[i])
            # plt.pause(0.1)

        else:
            I_avg, Q_avg = f.average_data(Id, Qd)
            Ilist[i] = I_avg
            Qlist[i] = Q_avg

            # plt.figure('cutline')
            # plt.plot(c_ampArray, Ilist[i])
            # plt.plot(c_ampArray, Qlist[i])
            # plt.pause(0.1)

    if len(g_ampArray) != 1:
        if sel:
            plt.figure()
            plt.pcolormesh(g_ampArray, c_ampArray, glist.T, cmap='RdBu', vmax=1, vmin=0)
            plt.xlabel("gain amp(DAC)")
            plt.ylabel('conv amp(DAC)')
            cbar = plt.colorbar()
            cbar.set_label("g pct", rotation=90)


        else:
            plt.figure(figsize=(11, 5))
            plt.subplot(121)
            plt.pcolormesh(g_ampArray, c_ampArray, Ilist.T)
            plt.colorbar()
            plt.subplot(122)
            plt.pcolormesh(g_ampArray, c_ampArray, Qlist.T)
            plt.colorbar()
