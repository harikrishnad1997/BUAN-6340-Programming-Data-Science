import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats as sps
import scipy.linalg as spl
from itertools import product

def add_intercept(x_vars,y_var,data):
    newdf = data.copy()
    x_vars = ['(intercept)'] + x_vars
    newdf['(intercept)'] = np.ones((data.shape[0],))
    return(x_vars,y_var,newdf)
    
class learning_model(object):
    def __init__(self,x):
        self.x = x
        (self.n_obs,self.n_feat) = self.x.shape
class unsupervised_model(learning_model):
    pass
class supervised_model(learning_model):
    def __init__(self,x,y):
        super(supervised_model,self).__init__(x)
        self.y = y
        
class prediction_model(supervised_model):
    def predict(self,newdata): raise NotImplementedError()
    @property
    def fitted(self): return self.predict(self.x)
    @property
    def residuals(self): return self.y-self.fitted
            
class likelihood_model(learning_model):
    def evaluate_lnL(self,pred): raise NotImplementedError
    @property
    def lnL(self): return self.evaluate_lnL(self.fitted)
    @property
    def aic(self): return 2*self.n_feat-2*self.lnL
    @property
    def bic(self): return np.log(self.n_obs)*self.n_feat-2*self.lnL
    @property
    def deviance(self): return 2*self.lnL-2*self._null_lnL_()
    def _null_lnL_(self): return self.evaluate_lnL(np.ones(self.y.shape)*self.y.mean())
    def _gradient_(self,coef): raise NotImplementedError
    def _hessian_(self,coef): raise NotImplementedError
    def __vcov_params_lnL__(self): return -np.linalg.inv(self._hessian_(self.params))
    def __max_likelihood__(self,init_params):
        return newton(self._gradient_,self._hessian_,init_params)

class linear_model(prediction_model,likelihood_model):
    def __init__(self,x,y,*args,**kwargs):
        super(linear_model,self).__init__(x,y)
        self.params = self.__fit__(x,y,*args,**kwargs)
    def __fit__(self,x,y,*args,**kwargs): return np.linalg.solve(x.T@x,x.T@y)
    def predict(self,newdata): return newdata@self.params
    def evaluate_lnL(self,pred): 
        return -self.n_obs/2*(np.log(2*np.pi*(self.y-pred).var())+1)
    @property
    def r_squared(self):
        return 1-self.residuals.var()/self.y.var()
    @property
    def adjusted_r_squared(self):
        return 1-(1-self.r_squared)*(self.n_obs-1)/(self.n_obs-self.n_feat)
    @property
    def degrees_of_freedom(self):
        return self.n_obs-self.n_feat
    @property
    def ssq(self):
        return self.residuals.var()*(self.n_obs-1)/self.degrees_of_freedom

class least_squares_regressor(linear_model):
    def __init__(self,x,y,white:bool=False,hc:int=3,*args,**kwargs):
        super(least_squares_regressor,self).__init__(x,y,*args,**kwargs)
        self.white = white
        self.hc = hc
    @property
    def vcov_params(self): 
        if self.white: 
            return self._white_(self.hc)
        return np.linalg.inv(self.x.T@self.x)*self.ssq
    def _white_(self,hc):
        e = self.residuals.values.reshape(-1,1) if type(self.residuals)==pd.Series else self.residuals
        e = self.__hc_correction__(e**2,hc)
        meat = np.diagflat(e.flatten())
        bread = np.linalg.inv(self.x.T@self.x)@self.x.T
        return bread@meat@bread.T
    def __hc_correction__(self,ressq,hc):
        mx = 1-np.diagonal(self.x@np.linalg.solve(self.x.T@self.x,self.x.T)).reshape(-1,1)
        p = int(np.round((1-mx).sum()))
        if hc==1: ressq *= self.n_obs/(self.n_obs-self.n_feat)
        elif hc==2: ressq /= mx
        elif hc==3: ressq /= mx**2
        elif hc==4: 
            delta = 4*np.ones((self.n_obs,1))
            delta = np.hstack((delta,self.n_obs*(1-mx)/p))
            delta = delta.min(1).reshape(-1,1)
            ressq /= np.power(mx,delta)
        elif hc==5:
            delta = max(4,self.n_obs*0.7*(1-mx).max()/p)*np.ones((self.n_obs,1))
            delta = np.hstack((delta,self.n_obs*(1-mx)/p))
            delta = delta.min(1).reshape(-1,1)/2
            ressq /= np.power(mx,delta)
        return ressq
    
class broom_model(learning_model):
    @property
    def vcov_params(self)->np.ndarray: raise NotImplementedError()
    def _glance_dict_(self)->dict: raise NotImplementedError()
    @property
    def std_error(self): return np.sqrt(np.diagonal(self.vcov_params))
    @property
    def t(self): return self.params/self.std_error
    @property
    def p(self): return 2*sp.stats.t.cdf(-np.abs(self.t),df=self.n_obs-self.n_feat)
    def conf_int(self,level:float):
        spread = -self.std_error*sp.stats.t.ppf((1-level)/2,df=self.n_obs-self.n_feat)
        return np.vstack([self.params-spread,self.params+spread])
    @property
    def tidy(self):return self.tidyci(ci=False)
    def tidyci(self,level=0.95,ci=True):
        n = len(self.x_vars)
        df = [self.x_vars,self.params[:n],self.std_error[:n],self.t[:n],self.p[:n]]
        cols = ['variable','estimate','std.error','t.statistic','p.value']
        if ci:
            df += [self.conf_int(level)[:,:n]]
            cols += ['ci.lower','ci.upper']
        df = pd.DataFrame(np.vstack(df).T)
        df.columns = cols
        return df
    @property
    def glance(self):
        return pd.DataFrame(self._glance_dict_(),index=[''])
    
class LeastSquaresRegressor(least_squares_regressor,broom_model):
    def __init__(self,x_vars:list,y_var:str,data:pd.DataFrame,*args,**kwargs):
        super(LeastSquaresRegressor,self).__init__(data[x_vars],data[y_var],*args,**kwargs)
        self.x_vars = x_vars
        self.y_var = y_var
        self.data = data
    def _glance_dict_(self):
        return {'r.squared':self.r_squared,
                'adjusted.r.squared':self.adjusted_r_squared,
                'self.df':self.n_feat,
                'resid.df':self.degrees_of_freedom,
                'aic':self.aic,
                'bic':self.bic,
                'log.likelihood':self.lnL,
                'deviance':self.deviance,
                'resid.var':self.ssq}