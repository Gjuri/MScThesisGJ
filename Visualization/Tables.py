import pandas as pd 
import numpy as np 
import matplotlib.ticker as ticker
from lmfit import minimize, Parameters, Parameter, report_fit
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from math import sqrt
from IPython.display import clear_output
from EDA import *
def make_borders(df):
    return [
        dict(selector=f"tbody tr:nth-child({i})",
             props=[("border-bottom", "1px solid black")])
        for i in df.index.get_level_values(0).value_counts().cumsum()
    ]

def R2S(actual, predict):
    corr_matrix = np.corrcoef(actual, predict)
    corr = corr_matrix[0,1]
    return corr**2

'''PLGM'''
def PLGM_OverallTable(plgm):
    ve = plgm['EpiV[L]']
    vh = plgm['HypV[L]']
    vt = ve + vh

    PLGMe = plgm['EpiP[ug]']#*9.8421E-07#.describe()
    PLGMh = plgm['HypP[ug]']#*9.8421E-07#.describe()
    PLGM = PLGMe + PLGMh#.describe()
    OBSe = plgm['PEPItrue[ug]'].interpolate()#*9.8421E-07)#.describe()
    OBSh = plgm['HYPtrue[ug]'].interpolate()#*9.8421E-07)#.describe()
    OBS =  OBSe + OBSh
    Etrue,Htrue,Epred,Hpred = OBSe,OBSh,PLGMe,PLGMh
    Tt = Etrue +Htrue
    Tp = Epred +Hpred
    d = [[Tt,Tp],
        [Etrue,Epred],
        [Htrue,Hpred ]]

    names = ['HWC','EPI','HYP']
    vols = [vt,ve,vh]
    fe = lambda t,p: (p-t)/t
    g = {}
    for nx,n in enumerate(names):
        dt = d[nx]
        df_t = dt[0].groupby([dt[0].index.year,dt[0].index.month]).agg([np.median]).values.T[0]
        df_p = dt[1].groupby([dt[1].index.year,dt[1].index.month]).agg([np.median]).values.T[0]
        df_v = (vols[nx]).groupby([(vols[nx]).index.year,(vols[nx]).index.month]).agg([np.median]).values.T[0]
        # e = np.median(fe(own_norm(df_t,norm = 'None',inf = True),own_norm(df_p,norm = 'None',inf = True)))#np.nanmedian((df_p-df_t)/df_t)#fe(df_t,df_p)

        e = np.median(fe(df_t/df_v,df_p/df_v))#np.nanmedian((df_p-df_t)/df_t)#fe(df_t,df_p)

        q=quartiles((dt[0]/vols[nx]).values,(dt[1]/vols[nx]).values); 
        IQRp= q[0];CVp = q[1];  IQRt = q[2];CVt = q[3]
        # rr = mean_squared_error(dt[0],dt[1])**(1/2)
        # mse = mean_squared_error(dt[0],dt[1])
        # mae = mean_absolute_error(dt[0],dt[1])
        # r2 = r2_score(dt[0],dt[1])
        rr = mean_squared_error(own_norm(dt[0]/vols[nx]),own_norm(dt[1]/vols[nx]))**(1/2)
        mse = mean_squared_error(own_norm(dt[0]/vols[nx]),own_norm(dt[1]/vols[nx]))
        mae = mean_absolute_error(own_norm(dt[0]/vols[nx]),own_norm(dt[1]/vols[nx]))
        r2 = R2S(own_norm(dt[0]),own_norm(dt[1]))
        #[predicted_iqr,predicted_cv,observed_iqr,observed_cv]
        g[n] = {'Metric':{'Predicted':{'MEP':e,'MAE':mae,'MSE':mse,
                                       'R\u00b2':r2,'RMSE':rr,
                                       '  IQR \n[\u00B5gP L\u207B\u00B9]':IQRp,'  CV \n[\u00B5gP L\u207B\u00B9]':CVp,
                                       'IQRe':(IQRp-IQRt)/IQRt, 'CVe':(CVp-CVt)/CVp
                                      },

                          'Observed':{'  IQR \n[\u00B5gP L\u207B\u00B9]':IQRt,'  CV \n[\u00B5gP L\u207B\u00B9]':CVt}}} #predicted_iqr,predicted_cv,observed_iqr,observed_cv
    gd = []
    for key, value in g.items(): # ['HWC', 'EPI', 'HYP']
        for keym,valuem in g[key]['Metric'].items():#['Metric']['pred','obs']:
            for keyp,valuesp in g[key]['Metric'][keym].items():#['E','IQR', 'CV',]
                v = {
                'Compartment': key,
                'DF':keym,
                'Metric': keyp,
                'Value': valuesp
                    }
                gd.append(v)
    GD = pd.DataFrame(gd)

    ind = pd.MultiIndex.from_product([GD.Compartment.unique()])
    cols = [('Value',  'Observed',  '  CV \n[\u00B5gP L\u207B\u00B9]'),    
    ('Value',  'Observed', '  IQR \n[\u00B5gP L\u207B\u00B9]'),
    ('Value', 'Predicted',  '  CV \n[\u00B5gP L\u207B\u00B9]'),
    ('Value', 'Predicted', '  IQR \n[\u00B5gP L\u207B\u00B9]'),
    ('Value', 'Predicted', 'IQRe'),
    ('Value', 'Predicted', 'CVe'),
    ('Value', 'Predicted', 'MEP'),
    ('Value', 'Predicted', 'RMSE'),
    ('Value', 'Predicted', 'MSE'),
    ('Value', 'Predicted', 'MAE'),
    ('Value', 'Predicted', 'R\u00b2')]#,
    values = GD.pivot_table(GD,index = ['Compartment'], columns = ['DF','Metric']).reindex(['HWC', 'EPI', 'HYP']).loc[:,cols]#.values
    Table = pd.DataFrame(np.round(values.values,3),index = ind,columns = pd.MultiIndex.from_tuples(cols))
    Table.index = ['WcP',
                'EpiP',
                'HypP']
    _table_styles = [
        dict(selector="thead", props=[("border-bottom", "1.5pt solid black")]),
        dict(selector="thead tr:first-child", props=[("display", "None")]),  # Hides first row of header
        dict(selector=".col_heading", props=[("text-align", "center"), ("vertical-align", "middle")]),
        dict(selector='th', props= [('text-align', 'center')]),
        dict(selector='th.col_heading', props= [('white-space', 'pre-wrap')]),
        dict(selector="tbody tr", props=[("background-color", "white")]),
        dict(selector="tbody tr td", props=[("text-align", "center")]),
        dict(selector=".row_heading", props=[("text-align", "left"), ("font-weight", "bold")]),
    ];
    Table0 = Table.style.set_table_styles(_table_styles + make_borders(Table)).set_table_attributes(
                                        'style="border-collapse: collapse;"'                                 
                                        '<colgroup>'
                                        '<col style="border-right: 1pt solid black;">'  # Vertical line after first column
                                        '<col>'
                                        '<col style="border-right: 1pt solid black;">'  # Vertical line after fourth column
                                        '<col>'
                                        '<col>'
                                        '<col>'
                                        '<col>'
                                        # '<col>'
                                        # '<col>'   
                                        '<col>'
                                        '<col>'
                                        '<col style="border-right: 1pt solid black;">'  # Vertical line after fourth column
                                        '</colgroup>'
                                        'style="border-top: 1.5pt solid black; border-bottom: 1.5pt solid black;"'

    ).format(
        {
            ('Value', 'Observed', '  CV \n[\u00B5gP L\u207B\u00B9]'): '{:.1f}',  # Format for 'CV' under 'Observed' of 'Value'
            ('Value', 'Observed', '  IQR \n[\u00B5gP L\u207B\u00B9]'): '{:.1f}',  # Format for 'IQR' under 'Observed' of 'Value'
            # Format for 'CV' under 'Predicted' of 'Value'
            ('Value', 'Predicted', '  CV \n[\u00B5gP L\u207B\u00B9]'): '{:.1f}',  # Format for 'CV' under 'Observed' of 'Value'
            ('Value', 'Predicted', '  IQR \n[\u00B5gP L\u207B\u00B9]'): '{:.1f}',
            ('Value', 'Predicted', 'IQRe'): '{:.0%}',
            ('Value', 'Predicted', 'CVe'): '{:.0%}',
            ('Value', 'Predicted', 'RMSE'): '{:.3f}',
            ('Value', 'Predicted', 'MEP'): '{:.1%}',
            ('Value', 'Predicted', 'MAE'): '{:.3f}',
            ('Value', 'Predicted', 'MSE'): '{:.3f}',
            ('Value', 'Predicted', 'R\u00b2'): '{:.0%}'
            #('Value', 'Predicted', 'Error'): '{:.2%}'  # Format for 'Error' under 'Predicted' of 'Value'
        }
    );
    clear_output()
    return Table0

def PLGM_SeasonalTable(plgm):
    fe = lambda t,p: (p-t)/t
    s = {}
    #gb = lambda x: [x[0].index.day,x[1].index.year]
    gb = lambda x: x[1].index.year#[x[0].index.day,x[1].index.year]
    for ix, i in enumerate(plgm.Season.unique()):
        mask = plgm.Season == i

        plgm0 = plgm.interpolate()
        PLGMe = plgm0[mask]['EpiP[ug]']#*9.8421E-07#.describe()
        PLGMh = plgm0[mask]['HypP[ug]']#*9.8421E-07#.describe()
        PLGM = PLGMe + PLGMh#.describe()
        OBSe = plgm0[mask]['PEPItrue[ug]']#*9.8421E-07)#.describe()
        OBSh = plgm0[mask]['HYPtrue[ug]']#*9.8421E-07)#.describe()

        OBS =  OBSe + OBSh

        Etrue,Htrue,Epred,Hpred = OBSe,OBSh,PLGMe,PLGMh
        TTtrue = Etrue +Htrue
        TTpred = Epred +Hpred

        ev = plgm0[mask]['EpiV[L]']
        hv = plgm0[mask]['HypV[L]']
        tv = ev + hv

        vols = [[tv,tv],[ev,ev],[hv,hv]]
        d = [[TTtrue,TTpred],[Etrue,Epred],[Htrue,Hpred]]    
        names = ['HWC','EPI','HYP']

        for nx,n in enumerate(names):
            dt = d[nx]
            df_t = (dt[0]).groupby([dt[0].index.year,dt[0].index.month]).agg([np.median]).values.T[0]
            df_p = (dt[1]).groupby([dt[1].index.year,dt[1].index.month]).agg([np.median]).values.T[0]
            df_v = (vols[nx][0]).groupby([(vols[nx][0]).index.year,(vols[nx][0]).index.month]).agg([np.median]).values.T[0]
            e = np.median(fe(df_t/df_v,df_p/df_v))#np.nanmedian((df_p-df_t)/df_t)#fe(df_t,df_p)
            q=quartiles(dt[0]/vols[nx][0],dt[1]/vols[nx][0]); 
            IQRp= q[0];CVp = q[1];  IQRt = q[2];CVt = q[3]
            rr = mean_squared_error(own_norm(dt[0]/vols[nx][0]),own_norm(dt[1]/vols[nx][0]))**(1/2)
            mse = mean_squared_error(own_norm(dt[0]/vols[nx][0]),own_norm(dt[1]/vols[nx][0]))
            mae = mean_absolute_error(own_norm(dt[0]/vols[nx][0]),own_norm(dt[1]/vols[nx][0]))
            # r2 = r2_score(own_norm(dt[0]),own_norm(dt[1]))
            r2 = R2S(own_norm(dt[0]),own_norm(dt[1]))
            if n not in s:
                s[n] = {i:{'Metric':{'Predicted':{'MEP':e,'MAE':mae,'MSE':mse,
                                               'R\u00b2':r2,'RMSE':rr,
                                               '  IQR \n[\u00B5gP L\u207B\u00B9]':IQRp,'  CV \n[\u00B5gP L\u207B\u00B9]':CVp,
                                               'IQRe':(IQRp-IQRt)/IQRt, 'CVe':(CVp-CVt)/CVp
                                              },

                                  'Observed':{'  IQR \n[\u00B5gP L\u207B\u00B9]':IQRt,'  CV \n[\u00B5gP L\u207B\u00B9]':CVt}}} #predicted_iqr,predicted_cv,observed_iqr,observed_cv
                       }
            else:
                s[n][i] = {'Metric':{
                        'Predicted':{'MEP':e,'MAE':mae,'MSE':mse,
                                   'R\u00b2':r2,'RMSE':rr,
                                   '  IQR \n[\u00B5gP L\u207B\u00B9]':IQRp,'  CV \n[\u00B5gP L\u207B\u00B9]':CVp,
                                   'IQRe':(IQRp-IQRt)/IQRt, 'CVe':(CVp-CVt)/CVp
                                  },

                        'Observed':{
                            '  IQR \n[\u00B5gP L\u207B\u00B9]':IQRt,'  CV \n[\u00B5gP L\u207B\u00B9]':CVt
                                    }

                                    }
                        } #predicted_iqr,predicted_cv,observed_iqr,observed_cv

    ed = []
    for key, value in s.items(): # ['spring', 'summer', 'autumn', 'winter'
        for keym,valuem in s[key].items():#'HWC', 'EPI', 'HYP']
            for keyp,valuesp in s[key][keym]['Metric'].items():#['pred','obs']:
                for keyv,valuesv in s[key][keym]['Metric'][keyp].items():#['E','IQR', 'CV',]
                    v = {
                            'Season': keym,                        
                            'Compartment': ['WcP' if key == 'HWC' else('EpiP' if key =='EPI' else 'HypP')][0],
                            'OP':keyp,
                            'Metric':keyv,
                            'Value': valuesv
                        }
                    ed.append(v)



    ED = pd.DataFrame(ed)#.drop(columns = 'Error',axis=0)
    ind = pd.MultiIndex.from_product([ED.Compartment.unique(),list(ED.Season.unique())])


    cols =  [('Value',  'Observed',  '  CV \n[\u00B5gP L\u207B\u00B9]'),    
    ('Value',  'Observed', '  IQR \n[\u00B5gP L\u207B\u00B9]'),
    ('Value', 'Predicted',  '  CV \n[\u00B5gP L\u207B\u00B9]'),
    ('Value', 'Predicted', '  IQR \n[\u00B5gP L\u207B\u00B9]'),
    ('Value', 'Predicted', 'IQRe'),
    ('Value', 'Predicted', 'CVe'),
    ('Value', 'Predicted', 'MEP'),
    ('Value', 'Predicted', 'RMSE'),
    ('Value', 'Predicted', 'MSE'),
    ('Value', 'Predicted', 'MAE'),
    ('Value', 'Predicted', 'R\u00b2')]

    values = ED.pivot_table(ED,index = ['Compartment','Season'], columns = ['OP','Metric']).reindex(ind).loc[:,cols]#.values
    Table = pd.DataFrame(values.values,index = ind,columns = pd.MultiIndex.from_tuples(cols))

    _table_styles = [
        dict(selector="thead", props=[("border-bottom", "1.5pt solid black")]),
        dict(selector="thead tr:first-child", props=[("display", "None")]),  # Hides first row of header
        dict(selector=".col_heading", props=[("text-align", "center"), ("vertical-align", "middle")]),
        dict(selector='th', props= [('text-align', 'center')]),
        dict(selector='th.col_heading', props= [('white-space', 'pre-wrap')]),
        dict(selector="tbody tr", props=[("background-color", "white")]),
        dict(selector="tbody tr td", props=[("text-align", "center")]),
        dict(selector=".row_heading", props=[("text-align", "left"), ("font-weight", "bold")]),
    ];

    Table0 = Table.style.set_table_styles(_table_styles + make_borders(Table))\
                  .set_table_attributes(
                                        'style="border-collapse: collapse;"'                                 
                                        '<colgroup>'
                                        '<col>'  # First column
                                        '<col style="border-right: 1pt solid black;">'  # Vertical line after first column
                                        '<col>'
                                        '<col style="border-right: 1pt solid black;">'  # Vertical line after fourth column
                                        '<col>'
                                        '<col>'
                                        '<col>'
                                        '<col>'
                                        '<col>'
                                        '<col>'
                                        '<col style="border-right: 1pt solid black;">'  # Vertical line after fourth column
                                        '</colgroup>'
                                        'style="border-top: 1.5pt solid black; border-bottom: 1.5pt solid black;"'

                                        )\
                        .format(
                            {
                                ('Value', 'Observed', '  CV \n[\u00B5gP L\u207B\u00B9]'): '{:.1f}',  # Format for 'CV' under 'Observed' of 'Value'
                                ('Value', 'Observed', '  IQR \n[\u00B5gP L\u207B\u00B9]'): '{:.1f}',  # Format for 'IQR' under 'Observed' of 'Value'
                                # Format for 'CV' under 'Predicted' of 'Value'
                                ('Value', 'Predicted', '  CV \n[\u00B5gP L\u207B\u00B9]'): '{:.1f}',  # Format for 'CV' under 'Observed' of 'Value'
                                ('Value', 'Predicted', '  IQR \n[\u00B5gP L\u207B\u00B9]'): '{:.1f}',
                                ('Value', 'Predicted', 'IQRe'): '{:.0%}',
                                ('Value', 'Predicted', 'CVe'): '{:.0%}',
                                ('Value', 'Predicted', 'RMSE'): '{:.3f}',
                                ('Value', 'Predicted', 'MEP'): '{:.1%}',
                                ('Value', 'Predicted', 'MAE'): '{:.3f}',
                                ('Value', 'Predicted', 'MSE'): '{:.3f}',
                                ('Value', 'Predicted', 'R\u00b2'): '{:.1%}'
                                #('Value', 'Predicted', 'Error'): '{:.2%}'  # Format for 'Error' under 'Predicted' of 'Value'
                            })
    clear_output()
    return Table0


'_____________________'

'''PLGM, RNN, PGRNN'''

'_____________________'


def Bench_OverallTable(plgm):
    #Etrue,Htrue,Epred,Hpred = data(plgm,norm = True, datatype = 'Series')
    names = ['MMpred','RNNpred','HYB0pred']
    fe = lambda t,p: (p-t)/t
    g = {}
    t = plgm.loc[:,'PEPItrue']
    vol =plgm.loc[:,'EpiV[L]']
    for nx,n in enumerate(names):
        p = plgm.loc[:,n].values

        df_t = plgm.loc[:,'PEPItrue'].groupby([plgm.loc[:,'PEPItrue'].index.year,plgm.loc[:,'PEPItrue'].index.month]).agg([np.median]).values.T[0]
        df_p = plgm.loc[:,n].groupby([plgm.loc[:,n].index.year,plgm.loc[:,n].index.month]).agg([np.median]).values.T[0]
        df_v = vol.groupby([vol.index.year,vol.index.month]).agg([np.median]).values.T[0]  
        e = np.median(fe(df_t/df_v,df_p/df_v))#np.nanmedian((df_p-df_t)/df_t)#fe(df_t,df_p)

        q=quartiles((t/vol).values,(p/vol).values);     
        IQRp= q[0];CVp = q[1];  IQRt = q[2];CVt = q[3]

        rr = mean_squared_error(own_norm(t/vol),own_norm(p/vol))**(1/2)
        mse = mean_squared_error(own_norm(t/vol),own_norm(p/vol))
        mae = mean_absolute_error(own_norm(t/vol),own_norm(p/vol))
        r2 = R2S(own_norm(t),own_norm(p))   
        #[predicted_iqr,predicted_cv,observed_iqr,observed_cv]
        g[n] = {'Metric':{'Predicted':{'MEP':e,'MAE':mae,'MSE':mse,
                                       'R\u00b2':r2,'RMSE':rr,
                                       '  IQR \n[\u00B5gP L\u207B\u00B9]':IQRp,'  CV \n[\u00B5gP L\u207B\u00B9]':CVp,
                                       'IQRe':(IQRp-IQRt)/IQRt, 'CVe':(CVp-CVt)/CVp
                                      },

                          'Observed':{'  IQR \n[\u00B5gP L\u207B\u00B9]':IQRt,'  CV \n[\u00B5gP L\u207B\u00B9]':CVt}}} #predicted_iqr,predicted_cv,observed_iqr,observed_cv

    gd = []
    for key, value in g.items(): # ['HWC', 'EPI', 'HYP']
        for keym,valuem in g[key]['Metric'].items():#['Metric']['pred','obs']:
            for keyp,valuesp in g[key]['Metric'][keym].items():#['E','IQR', 'CV',]
                v = {
                'Compartment': ['PLGM' if key == 'MMpred' else ('RNN' if key == 'RNNpred' else 'PGRNN')][0],
                'DF':keym,
                'Metric': keyp,
                'Value': valuesp
                    }
                gd.append(v)
    GD = pd.DataFrame(gd)

    cols = [('Value',  'Observed',  '  CV \n[\u00B5gP L\u207B\u00B9]'),    
            ('Value',  'Observed', '  IQR \n[\u00B5gP L\u207B\u00B9]'),
            ('Value', 'Predicted',  '  CV \n[\u00B5gP L\u207B\u00B9]'),
            ('Value', 'Predicted', '  IQR \n[\u00B5gP L\u207B\u00B9]'),
            # ('Value', 'Predicted', 'IQRe'),
            # ('Value', 'Predicted', 'CVe'),
            ('Value', 'Predicted', 'MEP'),
            ('Value', 'Predicted', 'RMSE'),
            ('Value', 'Predicted', 'MSE'),
            ('Value', 'Predicted', 'MAE'),
            ('Value', 'Predicted', 'R\u00b2')]#['MMpred','RNNpred','HYB0pred']
    # ind = pd.MultiIndex.from_product([['PLGM','RNN','PGRNN']])
    ind = pd.MultiIndex.from_product([['PLGM','RNN','PGRNN']])


    values = GD.pivot_table(GD,index = ['Compartment'], columns = ['DF','Metric']).reindex(['PLGM','RNN','PGRNN']).loc[:,cols]
    Table = pd.DataFrame(values.values,index = ind,columns = pd.MultiIndex.from_tuples(cols))
    _table_styles = [
        dict(selector="thead", props=[("border-bottom", "1.5pt solid black")]),
        dict(selector="thead tr:first-child", props=[("display", "None")]),  # Hides first row of header
        dict(selector=".col_heading", props=[("text-align", "center"), ("vertical-align", "middle")]),
        dict(selector='th', props= [('text-align', 'center')]),
        dict(selector='th.col_heading', props= [('white-space', 'pre-wrap')]),
        dict(selector="tbody tr", props=[("background-color", "white")]),
        dict(selector="tbody tr td", props=[("text-align", "center")]),
        dict(selector=".row_heading", props=[("text-align", "left"), ("font-weight", "bold")]),
    ];
    Table0 = Table.style.set_table_styles(_table_styles + make_borders(Table)).set_table_attributes(
                                        'style="border-collapse: collapse;"'                                 
                                        '<colgroup>'
                                        '<col style="border-right: 1pt solid black;">'  # Vertical line after first column
                                        '<col>'
                                        '<col style="border-right: 1pt solid black;">'  # Vertical line after fourth column
                                        '<col>'
                                        '<col>'
                                        '<col>'
                                        '<col>'
                                        '<col>'
                                        '<col>'   
                                        # '<col>'
                                        # '<col>'
                                        '<col style="border-right: 1pt solid black;">'  # Vertical line after fourth column
                                        '</colgroup>'
                                        'style="border-top: 1.5pt solid black; border-bottom: 1.5pt solid black;"'

    ).format(
        {
            ('Value', 'Observed', '  CV \n[\u00B5gP L\u207B\u00B9]'): '{:.1f}',  # Format for 'CV' under 'Observed' of 'Value'
            ('Value', 'Observed', '  IQR \n[\u00B5gP L\u207B\u00B9]'): '{:.1f}',  # Format for 'IQR' under 'Observed' of 'Value'
            # Format for 'CV' under 'Predicted' of 'Value'
            ('Value', 'Predicted', '  CV \n[\u00B5gP L\u207B\u00B9]'): '{:.1f}',  # Format for 'CV' under 'Observed' of 'Value'
            ('Value', 'Predicted', '  IQR \n[\u00B5gP L\u207B\u00B9]'): '{:.1f}',
            ('Value', 'Predicted', 'IQRe'): '{:.0%}',
            ('Value', 'Predicted', 'CVe'): '{:.0%}',
            ('Value', 'Predicted', 'RMSE'): '{:.3f}',
            ('Value', 'Predicted', 'MEP'): '{:.1%}',
            ('Value', 'Predicted', 'MAE'): '{:.3f}',
            ('Value', 'Predicted', 'MSE'): '{:.3f}',
            ('Value', 'Predicted', 'R\u00b2'): '{:.1%}'
            #('Value', 'Predicted', 'Error'): '{:.2%}'  # Format for 'Error' under 'Predicted' of 'Value'
        }
    )

    clear_output()
    return Table0


def Bench_SeasonalTable(plgm):
    s = {}
    fe = lambda t,p: (p-t)/t
    for ix, i in enumerate(plgm.Season.unique()):
        mask = plgm.Season == i
        t = plgm[mask]['PEPItrue']
        vol = plgm[mask]['EpiV[L]']
        q=quartiles(t/vol)
        CVt = q[1];IQRt = q[0]
        for nx,n in enumerate(['MMpred','RNNpred','HYB0pred']):
            p = plgm[mask][n]
            df_p = p.groupby([p.index.year,p.index.month]).agg([np.median]).values.T[0]
            df_t = t.groupby([t.index.year,t.index.month]).agg([np.median]).values.T[0]
            df_v = vol.groupby([vol.index.year,vol.index.month]).agg([np.median]).values.T[0]
            e = np.median(fe(df_t/df_v,df_p/df_v))#np.nanmedian((df_p-df_t)/df_t)#fe(df_t,df_p)
            q=quartiles(p/vol); 
            IQRp= q[0];CVp = q[1]
            rr = mean_squared_error(own_norm(p/vol),own_norm(t/vol))**(1/2)
            mse = mean_squared_error(own_norm(p/vol),own_norm(t/vol))
            mae = mean_absolute_error(own_norm(p/vol),own_norm(t/vol))
            # r2 = r2_score(own_norm(p),own_norm(t))
            r2 = R2S(own_norm(p),own_norm(t))
            if n not in s:
                s[n] = {i:{'Metric':{'Predicted':{'MEP':e,'MAE':mae,'MSE':mse,
                                               'R\u00b2':r2,'RMSE':rr,
                                               '  IQR \n[\u00B5gP L\u207B\u00B9]':IQRp,'  CV \n[\u00B5gP L\u207B\u00B9]':CVp,
                                               'IQRe':(IQRp-IQRt)/IQRt, 'CVe':(CVp-CVt)/CVp
                                              },

                                  'Observed':{'  IQR \n[\u00B5gP L\u207B\u00B9]':IQRt,'  CV \n[\u00B5gP L\u207B\u00B9]':CVt}}} #predicted_iqr,predicted_cv,observed_iqr,observed_cv
                       }
            else:
                s[n][i] = {'Metric':{
                        'Predicted':{'MEP':e,'MAE':mae,'MSE':mse,
                                   'R\u00b2':r2,'RMSE':rr,
                                   '  IQR \n[\u00B5gP L\u207B\u00B9]':IQRp,'  CV \n[\u00B5gP L\u207B\u00B9]':CVp,
                                   'IQRe':(IQRp-IQRt)/IQRt, 'CVe':(CVp-CVt)/CVp
                                  },

                        'Observed':{
                            '  IQR \n[\u00B5gP L\u207B\u00B9]':IQRt,'  CV \n[\u00B5gP L\u207B\u00B9]':CVt
                                    }

                                    }
                        } #predicted_iqr,predicted_cv,observed_iqr,observed_cv

    ed = []
    for key, value in s.items(): # ['spring', 'summer', 'autumn', 'winter'
        for keym,valuem in s[key].items():#'HWC', 'EPI', 'HYP']
            for keyp,valuesp in s[key][keym]['Metric'].items():#['pred','obs']:
                for keyv,valuesv in s[key][keym]['Metric'][keyp].items():#['E','IQR', 'CV',]
                    v = {
                            'Season': keym,                        
                            'Compartment': ['PGRNN' if key =='HYB0pred' else ('PLGM' if key == 'MMpred' else 'RNN')][0],#key,
                            'OP':keyp,
                            'Metric':keyv,
                            'Value': valuesv
                        }
                    ed.append(v)



    ED = pd.DataFrame(ed)#.drop(columns = 'Error',axis=0)
    ind = pd.MultiIndex.from_product([ED.Compartment.unique(),list(ED.Season.unique())])

    cols = [('Value',  'Observed',  '  CV \n[µgP L⁻¹]'),
                ('Value',  'Observed', '  IQR \n[µgP L⁻¹]'),
                ('Value', 'Predicted',  '  CV \n[µgP L⁻¹]'),
                ('Value', 'Predicted', '  IQR \n[µgP L⁻¹]'),
                ('Value', 'Predicted',               'CVe'),
                ('Value', 'Predicted',              'IQRe'),
                ('Value', 'Predicted',               'MAE'),
                ('Value', 'Predicted',               'MEP'),
                ('Value', 'Predicted',               'MSE'),
                ('Value', 'Predicted',              'RMSE'),
                ('Value', 'Predicted',                'R²')]
    clear_output()
    cc = ['MMpred','RNNpred','HYB0pred']
    cc = ['PLGM','RNN','PGRNN']
    ind = pd.MultiIndex.from_product([cc,list(ED.Season.unique())])
    v = ED.pivot_table(ED,index = ['Compartment','Season'], columns = ['OP','Metric']).reindex(ind).loc[:,cols]
    T = pd.DataFrame(v.values,index = ind,columns = pd.MultiIndex.from_tuples(cols))
    _table_styles = [
        dict(selector="thead", props=[("border-bottom", "1.5pt solid black")]),
        dict(selector="thead tr:first-child", props=[("display", "None")]),  # Hides first row of header
        dict(selector=".col_heading", props=[("text-align", "center"), ("vertical-align", "middle")]),
        dict(selector='th', props= [('text-align', 'center')]),
        dict(selector='th.col_heading', props= [('white-space', 'pre-wrap')]),
        dict(selector="tbody tr", props=[("background-color", "white")]),
        dict(selector="tbody tr td", props=[("text-align", "center")]),
        dict(selector=".row_heading", props=[("text-align", "left"), ("font-weight", "bold")]),
    ];
    T = T.style.set_table_styles(_table_styles + make_borders(T)).set_table_attributes(
                                        'style="border-collapse: collapse;"'                                 
                                        '<colgroup>'
                                        '<col>'
                                        '<col style="border-right: 0.5pt solid black;">'  # Vertical line after first column
                                        '<col>'
                                        '<col style="border-right: 0.5pt solid black;">'
                                        '<col>'
                                        '<col>'
                                        '<col>'#
                                        '<col>'#
                                        '<col>'
                                        '<col>'
                                        '<col style="border-right: 0.5pt solid black;">'  # Vertical line after fourth column
                                        #'<col style="border-right: 1pt solid black;">'  # Vertical line after fourth column
                                        '</colgroup>'
                                        #'style="border-top: 10pt solid black; border-bottom: 10pt solid black;"'

    ).format(
        {
                ('Value',  'Observed',  '  CV \n[µgP L⁻¹]'):'{:.1f}',
                ('Value',  'Observed', '  IQR \n[µgP L⁻¹]'):'{:.1f}',
                ('Value', 'Predicted',  '  CV \n[µgP L⁻¹]'):'{:.1f}',
                ('Value', 'Predicted', '  IQR \n[µgP L⁻¹]'):'{:.1f}',
                ('Value', 'Predicted',               'CVe'):'{:.0%}',
                ('Value', 'Predicted',              'IQRe'):'{:.0%}',
                ('Value', 'Predicted',               'MAE'):'{:.2f}',
                ('Value', 'Predicted',               'MEP'):'{:.0%}',
                ('Value', 'Predicted',               'MSE'):'{:.2f}',
                ('Value', 'Predicted',              'RMSE'):'{:.2f}',
                ('Value', 'Predicted',                'R²'):'{:.0%}'
            #('Value', 'Predicted', 'Error'): '{:.2%}'  # Format for 'Error' under 'Predicted' of 'Value'
        })
    return T