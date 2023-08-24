import pandas as pd
import numpy as np 
from sklearn.metrics import*
import math
import itertools
import matplotlib.pyplot as plt
from scipy.stats import *
import scipy.stats as stats
from IPython.display import clear_output

import warnings

def count_combinations(lst = None, combination_size = 2):
    n = len(lst)
    return math.comb(n, combination_size)

def position(s = None): 
    for i in list(itertools.product(['x','y'] , ['0','1'])):
        n = i[0]+i[1]
        s = s.replace(f'{n}=','')
    s =[float(i) for i in s.split(',')]
    return s[-2:]

def norm_n5000(data0,data1=None):
    ddd = [[data0] if data1 is None else [data0,data1]][0]
    n = []
    for dx in range(len(ddd)) :
        dd = ddd[dx]
        d = (dd - np.mean(dd)) / np.std(dd)
        quantiles = np.percentile(d, np.arange(0.1, 100, 0.1))
        res = norm.ppf((np.arange(0.1, 100, 0.1) - 0.5) / 100)
        
        fig, axs = plt.subplots(2, 1, figsize=(20,20))
        axs[0].hist(d)
        axs[0].set_title(f'Histogram {dx}')
        
        axs[1].scatter(res, quantiles)
        axs[1].plot(res, res, color='red', linestyle='--')
        axs[1].set_title(f'Q-Q Plot {dx}')
        axs[1].set_xlabel('Theoretical Quantiles')
        axs[1].set_ylabel('Sample Quantiles')
        plt.show()
        exec(f'n{dx} = eval(input("Not Normal?:"))')
        eval(f'n.append(n{dx})')
        plt.clf()
        plt.close()
    return [False if np.asarray(n).any() else True][0]

def Ttest(data0,data1,ad,norm = None):
        if norm:
            statistic, p = stats.ttest_ind(data0, data1)
            r = '(NORM) '+['Not dif' if p > ad else 'Sig dif'][0]
        else:
            statistic, p = mannwhitneyu(data0, data1)
            r = '(NOT NORM) '+['Not dif' if p > ad else 'Sig dif'][0]
        return r,p

    
def ANOVA(data0,an,ad):
    normal = np.asarray([True if shapiro(x)[1] < an else False for x in data0]).any()
    if not normal:
        k = 'kruskal('
        n = '(NOT NORM) '
    else:
        k  = 'f_oneway('
        n = '(NORM) '
        
    for ix,i in enumerate(data0):
        if ix < len(data0)-1:
            k += f'data0[{ix}],'#[0].values,'
        else:
            k+=  f'data0[{ix}])'#[0].values)'
    r = n +['Not dif' if eval(k)[1] > ad else 'Sig dif'][0]
    return(r)

def mean_to_mean(data0,data1,an,ad, printt):    
    stat0, p0 = shapiro(data0)
    stat1, p1 = [shapiro(data1) if data1 is not None else [False,False]][0] 

    # Normal distribution check
    norm = [True if (p1 is not None and p0 > an and p1 >an) else 
            (False if (p1 is not None and (p0 < an or p1 < an)) else 'NT')][0]
    if norm == 'NT':
        r = ['Normal distributed' if p0 > an else 'Not normal'][0]
        print(r)
    else:
        if len(data0) > 5000 or len(data1) > 5000:
            own_norm = norm_n5000(data0,data1) 
            r = Ttest(data0,data1,ad = ad,norm = own_norm)
        else:
            r = Ttest(data0,data1,ad = ad,norm = norm)
        if printt == True:
            print(r)
    return r
def PH(data0,ad,gn):
    ''' Post-Hoc test'''
    f = lambda g1,g2: [data0.index(g1),data0.index(g2)]
    f1 = lambda x: [gn[x[0]],gn[x[1]]]
    
    for g1,g2 in list(itertools.combinations(data0, 2)):
        ts, p  = stats.ttest_ind(g1, g2)
        if p < ad:
            #group_names = [data0 if gn is None else gn]
            group_names = [f(g1,g2) if gn is None else f1(f(g1,g2))][0]
            print(f"{p:.1e} between {group_names[0]} and {group_names[1]}")
            
def InferencialAnalysis(data0,data1 = None,an = 0.05, ad = 0.05, printt = False,gn = None):    

    cond = np.asarray(
                [False if type(x)!= list else (True if type(x)== list and len(data0)>=2 else False) for x in data0]
                      ).any()
    if cond and data1 is None:
        r = ANOVA(data0,an,ad)        
        if [False if 'Sig dif' not in r else (False if printt == False else True)][0]:
             PH(data0,ad,gn)
    else:
        r = mean_to_mean(data0,data1,an,ad,printt)
    return r

def error_percentage(observed=None,predicted=None, ABS = False, norm = False, percent = False, mean = False):
    #assert([False if (observed is None or predicted is None) else True][0]), 'Insert observed (y) and predited (p) values'
    k= [1 if percent == False else 100][0]
    if norm == True:
        observed,predicted = own_norm(observed),own_norm(predicted)        
    if ABS == False:
        f0 = lambda obs, pred: ((obs - pred) / pred)*k
    else:
        f0 = lambda obs, pred: (abs(obs - pred) / pred)*k
        
    if mean == True:
        pe = np.median(np.asarray([f0(obs,pred) for obs, pred in zip(observed,predicted)]))
    else:
        pe = np.asarray([f0(obs,pred) for obs, pred in zip(observed,predicted)])
    return np.asarray(pe)

def To_Series(data = None,index = None):
    series = pd.Series(data,index = index)
    return series

def data(plgm = None, compare='PLGM',mask0 = None,norm =False,datatype = None , norm_method = 'None'):
    '''
    Compare:  
            PLGM: Compartment level(Epi,Hyp,Total)
            Models: Among models (PLGM RNN PGRNN)
    '''
    mask = [plgm.Season == plgm.Season if mask0 is None else mask0][0]
    if compare == 'PLGM':        
        Etrue = plgm[mask]['PEPItrue[ug]'].interpolate()#*9.8421E-07
        Htrue = plgm[mask]['HYPtrue[ug]'].interpolate()#*9.8421E-07
        Epred = plgm[mask]['EpiP[ug]'].interpolate()#*9.8421E-07
        Hpred = plgm[mask]['HypP[ug]'].interpolate()#*9.8421E-07
        if norm:
            Etrue,Htrue,Epred,Hpred = [own_norm(x,norm = norm_method) for x in [Etrue,Htrue,Epred,Hpred]]
            if datatype == 'Series':
                Etrue,Htrue,Epred,Hpred = [To_Series(x,plgm.index) for x in [Etrue,Htrue,Epred,Hpred]]
        output = Etrue,Htrue,Epred,Hpred
    else:
        Etrue = plgm[mask]['PEPItrue[ug]'].interpolate()#*9.8421E-07
        Htrue = plgm[mask]['HYPtrue[ug]'].interpolate()#*9.8421E-07
        Epred = plgm[mask]['EpiP[ug]'].interpolate()#*9.8421E-07
        Hpred = plgm[mask]['HypP[ug]'].interpolate()#*9.8421E-07  
        if norm:
            true,PLGM,RNN,PGRNN = [own_norm(x,norm = norm_method) for x in [true,PLGM,RNN,PGRNN]]
            if datatype == 'Series':
                true,PLGM,RNN,PGRNN = [To_Series(x,plgm.index) for x in [true,PLGM,RNN,PGRNN]]
        output = true,PLGM,RNN,PGRNN
        
    return output


def quartiles(observed_data = None,predicted_data = None, printt = False):
    observed_cv = np.std(observed_data) / np.mean(observed_data) * 100
    observed_iqr = np.percentile(observed_data, 75) - np.percentile(observed_data, 25)
    if predicted_data is not None:
        predicted_iqr = np.percentile(predicted_data, 75) - np.percentile(predicted_data, 25)
        predicted_cv = np.std(predicted_data) / np.mean(predicted_data) * 100
        output = [predicted_iqr,predicted_cv,observed_iqr,observed_cv]
    else:
        output = [observed_iqr,observed_cv]
    if printt == True:
        if predicted_data is not None:
            print("Predicted data:")
            print("Interquartile Range (IQR):", predicted_iqr)
            print("Coefficient of Variation (CV):", predicted_cv)
            tt = 'Observed data:'
        else:
            tt = 'IQR & CV'
            print(f"{tt}")
            print("Interquartile Range (IQR):", observed_iqr)
            print("Coefficient of Variation (CV):", observed_cv)
    else:
        pass
    return output

def own_metrics(y_true = None, pred = None,m = 'all', dictionary = True):
    assert([False if (y_true is None or pred is None) else True][0]), 'Insert observed (y) and predited (p) values'
    '''
    metrics: [rmse,mse,mae,r2,mep,iqr]
    '''
    mse,r2,rmse,mae,mep = mean_squared_error,r2_score,mean_squared_error,mean_absolute_error,error_percentage#, quartiles
    metrics = ['rmse','mse','mae','r2','mep']
    m = [metrics if m == 'all' else m][0]
    if type(m) is list:
        assert(not np.asarray([True if x not in metrics else False for x in m]).any()),'Use listed metrics'
        output = {}
        for mm in metrics:
            output[mm] = eval(f'{mm}(y_true,pred)')
        output = [pd.DataFrame(list(output.values()), columns = list(output.keys())) if dictionary == 'df' else output][0]
    else:
        output = eval(f'{m}(y_true,pred)')
    return output
                      

def own_norm(df,c = None, mask0 = None, norm = 'mm',ds = 2, lb = 10, inf = False):
    '''
    ########################################
    
    [df]: DataFrame or Vector    
            - If DataFrame 
            ° [c]     : Column(s) to standarize
            ° [mask0] : Column value filter
        
    [norm]: Normlaization method     
            -mm  : 'MinMax'       
            -zs  : 'Z-score'
            -rs  : 'Robust Scaling' 
            -uvs : 'Unit Vector Scaling'
            -ds  : 'Decimal Scaling' 
                            *(ds): umber of decimal places to scale
            -ls  : 'Logaritmic Scaling' 
                            *(lb): base of the logarithm functio
            -None
    
    [inf]: Inf fix
           *(True if np.isinf(x) for x in vector1/vector else False).any() == True
           * 1e-10 if inf == True for x == 0 in vector1 or vector2
           
    ########################################
    '''
    # Functions 
    if norm == 'mm':
        f = lambda d: [ (x - np.nanmin(d)) / (np.nanmax(d) - np.nanmin(d)) for x in d]
    elif norm == 'zs':
        f = lambda d: [ ((x - np.nanmean(d))/np.nanstd(d)) for x in d]
    elif norm == 'ds':
        f = lambda d: [ x**ds for x in d]
    elif norm == 'uvs':
        f = lambda d: [ x/np.linalg.norm(d) for x in d]
    elif norm == 'rs':
        f = lambda d: [ ((x - np.nanmean(d))/quartiles(d)[0]) for x in d]
    elif norm == 'ls':
        f = lambda d: [ np.log(x)/np.log(lb) for x in d]
    elif norm is None or norm == 'None' or norm == False: 
        f = lambda d: [x for x in d]
    else:
        raise ValueError("Invalid Normalization Method")  
        
    if type(df) == pd.core.frame.DataFrame:
        if mask0 is None:
            warnings.warn("No column filter were applied", UserWarning)
            clear_output(wait=True)
        mask = [df.Season.values == df.Season.values if mask0 is None else mask0][0]
        if c is not None and type(c) is not list:
            c = [c]
        else:
            pass 
        
        if c is not None and len(c) < 2: 
            on = f(df[mask][c].values.T)[0]            
        else:
            if c is None:
                ct = list(df.loc[:,[True if (df[col].dtype == np.float32 or df[col].dtype==np.float64)else False for col in df.columns]].columns)
                warnings.warn(f"No colums were passed, all columns numeric columns are taken:\n {ct}")
                c = ct
                clear_output(wait=True)
            
            on = {}
            for col in c:
                n_v = f(df[mask][col])
                on[col] ={'n_v':n_v}
                   
    elif type(df) == list or type(df)== np.ndarray or type(df)==pd.Series:
        on = np.asarray(f(df))
        if inf == True:
            on = np.array([1e-10 if i == 0 else i for i in on])
    else:
        raise ValueError("Input data has to be: pd.DataFrame pd.Series  \nnp.array  \nlist \n")
    
        
    
    return on