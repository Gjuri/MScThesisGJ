import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd 
import numpy as np 
import matplotlib.ticker as ticker
from IPython.display import clear_output
from EDA import *

def list_months_to_season(distinct_months : list) -> list:
    season = []
    autumn = ["September","October","November"]
    winter = ["December","January","February"]
    summer = ["June","July","August"]
    spring = ["March","April","May"]
    for month in distinct_months:
        if month in autumn :
            season.append("autumn")
        if month in winter :
            season.append("winter")
        if month in spring :
            season.append("spring")
        if month in summer :
            season.append("summer")
    
    return season

def PLGM_OverallBoxplot(plgm):
    #import matplotlib.pyplot as plt
    ve = plgm['EpiV[L]']
    vh = plgm['HypV[L]']
    vt = ve + vh
    PLGMe = plgm['EpiP[ug]']#*9.8421E-07#.describe()
    PLGMh = plgm['HypP[ug]']#*9.8421E-07#.describe()
    PLGM = PLGMe + PLGMh#.describe()
    OBSe = plgm['PEPItrue[ug]'].interpolate()#*9.8421E-07)#.describe()
    OBSh = plgm['HYPtrue[ug]'].interpolate()#*9.8421E-07)#.describe()
    OBS =  OBSe + OBSh
    data0 = [OBS/vt,PLGM/vt,OBSe/ve,PLGMe/ve,OBSh/vh,PLGMh/vh]
    colors  = ['lightgrey','lightgrey','cornflowerblue','cornflowerblue','yellowgreen','yellowgreen','salmon','salmon']

    xl = ["Overall",'',"Epilimnion",'','Hypolimnion','']
    xs = [0.5,0,2.5,0,4.5,0]
    xs = list(range(12)[::2])

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_xlim(0, 13)
    ax.set_xticks(xs)
    ax.set_xticklabels(xl)
    plt.tight_layout()
    colors = ['lightgrey', 'lightgrey', 'cornflowerblue', 'cornflowerblue', 'yellowgreen', 'yellowgreen', 'salmon', 'salmon']
    ec = ['black','red']*len(data0)
    mp = ['black']*len(data0)
    lss = ['-','-']*len(data0)
    TTT = [False,True]*len(data0)

    for i, d in enumerate(data0):
        x_pos = xs[i]+1  # Get the x position for the current boxplot
        ax.boxplot([d], positions=[x_pos], patch_artist=True, widths=0.5
                   ,boxprops = dict(facecolor=colors[i],linestyle='-.', linewidth=0.5, color=ec[i])
                   ,medianprops=dict(color="black", alpha=0.7,linestyle = lss[i])
                  , whiskerprops = dict(color =ec[i],alpha = 0.7,linestyle = lss[i])
                  ,capprops = dict(color = ec[i],alpha = 0.7,linestyle = lss[i]),notch=TTT[i])
        # ax.text(x_pos, ax.get_ylim()[0] - 0.5, labels[i], ha='center')
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_xticks([x+2 for x in xs])  # Set the x-ticks positions
    ax.set_xticklabels(xl)  # Set the x-tick labels
    ax.set_ylabel("Phosphorus [\u00B5g L\u207B\u00B9]", fontsize=16, labelpad =15)
    return plt.show()


def PLGM_OverallTimeSeries(plgm):
    ve = plgm['EpiV[L]']
    vh = plgm['HypV[L]']
    vt = ve + vh

    PLGMe = plgm['EpiP[ug]']#*9.8421E-07#.describe()
    PLGMh = plgm['HypP[ug]']#*9.8421E-07#.describe()
    PLGM = PLGMe + PLGMh#.describe()
    OBSe = plgm['PEPItrue[ug]'].interpolate()#*9.8421E-07)#.describe()
    OBSh = plgm['HYPtrue[ug]'].interpolate()#*9.8421E-07)#.describe()
    OBS =  OBSe + OBSh
    # PLGMs = plgm['Psed[ug]']/plgm['Sedv[L]']

    lwo = 0.2
    lw1 = 0.5
    lwp = 0.2
    mz = 2
    ms = 'o'
    mst = '-o'
    ylabelfz = 20
    tfz = 25

    fig, axs = plt.subplots(3, 1, figsize=(50,30))

    lwo = 4         # observed line withd
    lwp = 3.3       # Predicted line withd
    mz = 6          # Predicted mark size
    ms = 'o'        # mark size
    tfz = 50        # Title Font size
    tlz = 40        # x & y ticks size
    colors  = ['silver','silver','cornflowerblue','cornflowerblue','yellowgreen','yellowgreen','salmon','salmon']

    color = colors[::2]
    axs = axs.flatten()

    axs[0].plot(plgm.index,OBS/vt,mst,color = color[0],linewidth = lwo,markersize = mz, label = 'Obs')
    axs[0].plot(plgm.index,PLGM/vt, ms,color = color[1],linewidth = lwp,markersize = mz,label = 'PLGM')

    axs[1].plot(plgm.index,OBSe/ve,color = color[0],linewidth = lwo)#, label = 'Obs')
    axs[1].plot(plgm.index,PLGMe/ve, ms,color = color[2],linewidth = lwp,markersize = mz,label = 'PLGMe')

    axs[2].plot(plgm.index,OBSh/vh,color = color[0],linewidth = lwo)#, label = 'Obs')
    axs[2].plot(plgm.index,PLGMh/vh, ms,color = color[3],linewidth = lwp,markersize = mz,label = 'PLGMh')

    # axs[3].plot(plgm.index,PLGMs, ms,color = color[3],linewidth = lwp,markersize = mz,label = 'PLGMs')

    TITLE = ["Whole Water Column","Epilimnion","Hypolimnion"]
    handles = []
    labels = []
    for i in range(3):
        axs[i].tick_params(axis='both', which='major', labelsize=tlz,rotation = 45)
        axs[i].set_ylabel('Phosphorus [\u00B5g L\u207B\u00B9]', fontsize=tlz , labelpad=20)
        axs[i].set_title(TITLE[i], fontsize = tfz,fontweight='bold')
        axs[i].set_xlim(plgm.index[0],plgm.index[9000])
        years = mdates.YearLocator()    
        axs[i].legend().legendHandles[0].set_linewidth(10)
        axs[i].xaxis.set_major_locator(years)
        h, l = axs[i].get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    fig.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.1,
                        hspace=0.4)  

    fig.legend(handles, labels, bbox_to_anchor=(1.09, 0.8),
               handlelength = 10,handleheight = 5,
               markerscale = 10, prop={'size': 35});
    return plt.show()



def PLGM_SeasonalBoxplot(plgm):
    
    ####
    
    lwo = 0.2
    lw1 = 0.5
    lwp = 0.2
    mz = 2
    ms = 'o'
    mst = '-o'
    ylabelfz = 20
    tfz = 25
    ec = ['black','red']*6
    
    fig, axs = plt.subplots(2, 2, figsize=(30,19))

    colors  = ['silver','silver','cornflowerblue','cornflowerblue','yellowgreen','yellowgreen','salmon','salmon']
    color = colors[::2]
    xl = ["Overall",'',"Epilimnion",'','Hypolimnion','']

    axs = axs.flatten()
    for ix, i in enumerate(plgm.Season.unique()):
        mask = plgm.Season ==i
        ve = plgm[mask]['EpiV[L]']
        vh = plgm[mask]['HypV[L]']
        vt = ve + vh

        PLGMe = plgm[mask]['EpiP[ug]']#*9.8421E-07#.describe()
        PLGMh = plgm[mask]['HypP[ug]']#*9.8421E-07#.describe()
        PLGM = PLGMe + PLGMh#.describe()
        OBSe = plgm[mask]['PEPItrue[ug]'].interpolate()#*9.8421E-07)#.describe()
        OBSh = plgm[mask]['HYPtrue[ug]'].interpolate()#*9.8421E-07)#.describe()
        OBS =  OBSe + OBSh
        Etrue,Htrue,Epred,Hpred = OBSe,OBSh,PLGMe,PLGMh
        TTtrue = Etrue +Htrue
        TTpred = Epred +Hpred
        data0 = [TTtrue/vt,TTpred/vt,
                 Etrue/ve,Epred/ve,
                 Htrue/vh,Hpred/vh ]
        lss = ['-','-']*len(data0)
        TTT = [False,True]*len(data0)
        p = 0
        for di, d in enumerate(data0):            
            axs[ix].boxplot(d, positions=[di], patch_artist=True, widths=0.5
                       ,boxprops = dict(facecolor=colors[di],linestyle='-.', linewidth=1, color=ec[di])
                       ,medianprops=dict(color="black", alpha=0.7,linestyle = lss[di])
                      , whiskerprops = dict(color =ec[di],alpha = 0.7,linestyle = lss[di])
                      ,capprops = dict(color = ec[di],alpha = 0.7,linestyle = lss[di]),notch=TTT[di])
            axs[ix].set_title(i.title(), fontsize = tfz, fontweight = 'bold')
            axs[ix].set_xticks([x+0.5 for x in list(range(6))])  # Set the x-ticks positions
            axs[ix].set_xticklabels(xl,fontsize = ylabelfz  + 5)  # Set the x-tick labels
            axs[ix].set_ylim(0,250)
            axs[ix].set_ylabel("Phosphorus [\u00B5g L\u207B\u00B9]",fontsize = ylabelfz+5,labelpad = 20)
            # axs[ix].set_yticks([int(x) for x in axs[ix].get_yticks()], fontsize = ylabelfz+5)
            # axs[i].set_yticklabels(axs[ix].get_yticks(),fontsize = ylabelfz+10)
            axs[ix].tick_params(axis='y', which='major', labelsize=ylabelfz+5)

    for i in list(range(4))[1::2]:
        axs[i].yaxis.set_visible(False)    


    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.1,
                        hspace=0.4)
    return plt.show()


'_____________________'

'''PLGM, RNN, PGRNN'''

'_____________________'


def Bench_OverallBoxplot(plgm):
    #import matplotlib.pyplot as plt
    vol = plgm['EpiV[L]']
    t = plgm['PEPItrue']/vol
    R = plgm['RNNpred']/vol
    P = plgm['HYB0pred']/vol
    M = plgm['MMpred']/vol

    ftz = 10
    data0 = [M,R,P]
    colors  = ['lightgrey','lightgrey','cornflowerblue','cornflowerblue','yellowgreen','yellowgreen','salmon','salmon']

    xl = ["LG","PLGM","RNN",'PGRNN']
    xs = [0.5,1,1.5,2]
    fig, ax = plt.subplots(figsize=(12, 6))

    plt.tight_layout()
    colors = ['lightgrey', 'lightgrey', 'cornflowerblue', 'cornflowerblue', 'yellowgreen', 'yellowgreen', 'salmon', 'salmon']
    c = colors[::2]
    ec = ['black','red']*len(data0)
    mp = ['black']*len(data0)
    lss = ['-','-']*len(data0)
    TTT = [False,True]*len(data0)

    for i, d in enumerate([t]+data0):
        x_pos = xs[i]  # Get the x position for the current boxplot
        ax.boxplot(d, positions=[x_pos], patch_artist=True,
                  boxprops=dict(facecolor=c[i]),medianprops= dict(color ='black',linewidth = 2))

    ax.set_xlim(0.3, 2.3)
    ax.set_ylim(0,200)
    ax.set_xticks(xs,fontsize = ftz)
    ax.set_xticklabels(xl,fontsize = ftz)
    ax.set_title('Epilimnion', fontsize=16)
    ax.set_ylabel("Phosphorus [\u00B5g L\u207B\u00B9]", fontsize=16)
    return plt.show()


def Bench_SeasonalBoxplot(plgm):
    lwo = 0.2
    lw1 = 0.5
    lwp = 0.2
    mz = 2
    ms = 'o'
    mst = '-o'
    ylabelfz = 20
    tfz = 25
    ftz = 30

    ec = ['black','red']*6

    fig, axs = plt.subplots(2, 2, figsize=(30,30))

    colors  = ['silver','silver','cornflowerblue','cornflowerblue','yellowgreen','yellowgreen','salmon','salmon']
    color = colors[::2]
    axs = axs.flatten()
    for ix, i in enumerate(plgm.Season.unique()):    
        mask = plgm.Season == i
        vol = plgm[mask]['EpiV[L]']
        t = plgm[mask]['PEPItrue']/vol
        R = plgm[mask]['RNNpred']/vol
        P = plgm[mask]['HYB0pred']/vol
        M = plgm[mask]['MMpred']/vol

        ftz = 10
        # colors  = ['lightgrey','lightgrey','cornflowerblue','cornflowerblue','yellowgreen','yellowgreen','salmon','salmon']

        xl = ["LG","PLGM","RNN",'PGRNN']
        xs = [0,1,2,3]

        data0 = [M,R,P]
        lss = ['-','-']*len(data0)
        p = 0
        for di, d in enumerate([t]+data0):            
            axs[ix].boxplot(d, positions=[xs[di]], patch_artist=True, widths=0.5
                       ,boxprops = dict(facecolor=color[di],linestyle='-.', linewidth=1)
                       ,medianprops=dict(color="black", alpha=0.7,linestyle = lss[di])
                      , whiskerprops = dict(color =ec[di],alpha = 0.7,linestyle = lss[di])
                      # ,capprops = dict(color = ec[di],alpha = 0.7,linestyle = lss[di])
                           )
            axs[ix].set_title(i.title(), fontsize = tfz)
            # axs[ix].set_xticks(xs)  # Set the x-ticks positions
            # axs[ix].set_xticklabels(xl,fontsize = ylabelfz )  # Set the x-tick labels
            # axs[ix].set_ylim(0,250)
            axs[ix].set_ylabel("Phosphorus [\u00B5g L\u207B\u00B9]",fontsize = ylabelfz,labelpad = 20)
            # axs[ix].set_xlim(0.3, 2.3)
            axs[ix].set_ylim(0,200)
            axs[ix].set_xticks(xs,fontsize = ftz)
            axs[ix].set_xticklabels(xl,fontsize = ftz+10)
            axs[ix].tick_params(axis='y', labelsize=20)  ### THIS!!!!


    for ii in list(range(4))[1::2]:
        axs[ii].set_yticks([int(x) for x in axs[ii].get_yticks()])
        axs[ii].set_yticklabels( [str(x) for x in axs[ii].get_yticks()])    
        axs[ii].yaxis.set_visible(False)
        print(i)



    plt.subplots_adjust(left=0.03,
                        bottom=0.1,
                        right=0.9,
                        top=0.8,
                        wspace=0.1,
                        hspace=0.4)
    return plt.show()


def Bench_SeasonalTimeSeries(plgm):
    fig, axs = plt.subplots(3, 1, figsize=(45,25))

    lwo = 4         # observed line withd
    lwp = 3.3       # Predicted line withd
    mz = 4          # Predicted mark size
    ms = 'o'        # mark size
    tfz = 30        # Title Font size
    tlz = 25        # x & y ticks size
    colors  = ['silver','silver','cornflowerblue','cornflowerblue','yellowgreen','yellowgreen','salmon','salmon']

    color = colors[::2]
    vol = plgm['EpiV[L]'].values
    axs = axs.flatten()
    t = plgm['PEPItrue']/vol
    R = plgm['RNNpred']/vol
    P = plgm['HYB0pred']/vol
    M = plgm['MMpred']/vol

    axs[0].plot(plgm.index,t,color = color[0],linewidth = lwo, label = 'Obs')
    axs[0].plot(plgm.index,M, ms,color = color[1],linewidth = lwp,markersize = mz,label = 'PLGM')

    axs[1].plot(plgm.index,t,color = color[0],linewidth = lwo)#, label = 'Obs')
    axs[1].plot(plgm.index,R, ms,color = color[2],linewidth = lwp,markersize = mz,label = 'RNN')

    axs[2].plot(plgm.index,t,color = color[0],linewidth = lwo)#, label = 'Obs')
    axs[2].plot(plgm.index,P, ms,color = color[3],linewidth = lwp,markersize = mz,label = 'PGRNN')
    minn = [0,-20,-5]
    TITLE = ["PLGM","RNN","PGRNN"]
    handles = []
    labels = []
    for i in range(3):
        axs[i].tick_params(axis='both', which='major', labelsize=tlz,rotation = 45)
        axs[i].set_ylabel('Phosphorus [\u00B5g L\u207B\u00B9]', fontsize=tlz+5 , labelpad=20)
        axs[i].set_title(TITLE[i], fontsize = tfz,fontweight='bold')
        axs[i].set_xlim(plgm.index[0],plgm.index[-1])
        axs[i].set_ylim(minn[i],100)
        years = mdates.YearLocator()
        # if i == 0:
        #     axs[0].legend().legendHandles[0].set_linewidth(10)
        axs[i].xaxis.set_major_locator(years)
        h, l = axs[i].get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    fig.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.1,
                        hspace=0.4)  

    # Create a single legend outside of the subplots


    fig.legend(handles, labels, bbox_to_anchor=(1.07, 0.93),
               handlelength = 10,handleheight = 5,
               markerscale = 10, prop={'size': 30})
    #,loc='upper right')

    plt.show()