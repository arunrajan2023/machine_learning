
#Machine learning code
#Written by Arun 
#Programming language: Python

import numpy as np
import scipy as sp
import math
import time
import itertools
from numpy import linalg              
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split          
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.kernel_ridge import KernelRidge
from matplotlib import pyplot as plt
import sys, traceback
import warnings
import random
warnings.simplefilter("error")
np.seterr(divide='ignore', invalid='ignore')   #skips divide by zero , NaN errors
     
start = time.time()			  #timer
file1 = open("mxene.dat",      "r")        #data file
gwfile= open("gw.dat",    "r")       #GW file
ftag  = open("tags.dat",       "r")       #Tags
fout  = open("comp-features",  "w+")

#reading data file
data   = []
for line in file1:
	data.append( [float(n) for n in line.split()])

#reading GW file
egap_gw= []
for line2 in gwfile:
	egap_gw.append(line2.split())

#reading tags file
tags    = []
symbols = []
for _tag in ftag:
	_tag = _tag.split()
	tags.append(_tag[0])
	symbols.append(_tag[1])
print("tags    :", tags)
print("symbols :", symbols)
 
#convert to numpy arrays
data    = np.asarray(data,    dtype=float) 
egap_gw = np.asarray(egap_gw, dtype=float)

print("")
print("====================================================================")
print("                       MACHINE LEARNING                             ")
print("====================================================================")

 
#---------------Normalize the given data------------------
print("Data...")
data_nrow, data_ncol   = data.shape
#print(data.shape)

print("Normalization (Centering and Scaling). . .")
#centering
data = StandardScaler().fit_transform(data)

#scale 0-1
data = MinMaxScaler().fit_transform(data)                       
print("")

print("====================================================================")

#-------------------- primary features--------------------
prim_data  = np.copy(data)
print("Primary features           :", prim_data.shape)
print("")

#------------------- #We dont use feature selection here!
prim_data1 = prim_data
prim_data2 = prim_data
prim_nrow, prim_ncol = prim_data.shape

#---------------- Correlation co-efficient -----------------
print("Highly correlated primary features.......")
corrmax = 0.99
deletep = []
for _prim_ncol in range(prim_ncol):
	column1 = (prim_data[:,[_prim_ncol]]).T
	for _prim_ncol_new in range(prim_ncol):
		column2 = (prim_data[:,[_prim_ncol_new]]).T
		if _prim_ncol < _prim_ncol_new:
			corr    = np.corrcoef(column1, column2)[0, 1]
			if corr > corrmax:
				print(_prim_ncol, " & ", _prim_ncol_new, ": corr=", corr)
				deletep.append(_prim_ncol_new)

				#append symbol
				symbols[_prim_ncol_new]  = 'nil'

deletep = np.unique(deletep)
if len(deletep) > 0:
	print(len(deletep)," indices to be deleted:", deletep)
	prim_data = np.delete(prim_data, deletep, 1)
	print("")
	print("Primary features (modified):", prim_data.shape)
	prim_nrow, prim_ncol = prim_data.shape

else:
	print("No feature to be deleted.")
print("")

print("====================================================================")
#--------------compound features---------------------------------                          
print("Compound features:")

#--------------1D compound features----------
print("1D compound features:")
comp_data1D = np.concatenate((prim_data, 1./(1.+prim_data), 
	np.sqrt(prim_data), 1./np.sqrt(prim_data +1),
	np.log10(prim_data+1), 1.0/(1+np.log10(prim_data+1)), 
	np.cbrt(prim_data), 1./np.cbrt(prim_data +1),
	prim_data * prim_data, 1./((1.+prim_data) * (1.+prim_data)),
	np.exp(prim_data), np.exp(-prim_data)), axis=1)

comp_tag1D  = ['x', 'ix', 'sqrt', 'isqrt', 'log10', 'ilog10', 
	'cbrt', 'icbrt', 'sq', 'isq', 'exp', 'iexp']

comp_tag    = []
xctr        = 0
for _symbols in range(len(symbols)):
	for _comp_tag1D in range(len(comp_tag1D)):
		new_tag = str(comp_tag1D[_comp_tag1D]) + "(" + str(symbols[_symbols]) + ")"
		#print(xctr, _symbols, _comp_tag1D, new_tag)
		comp_tag.append(new_tag)
		xctr += 1
#print("1D comp_tag:", comp_tag)
print("1D comp_tag:", len(comp_tag), " counter=", xctr)
              
comp_nrow, comp_ncol = comp_data1D.shape
print("comp_data1D    :", comp_data1D.shape)
print("Done")
print("")


#--------------2D compound features----------
comp_ncol1     = int(comp_ncol * (1 + comp_ncol)/ 2 )
print("2D compound features:")	
comp_data_1D2D = np.copy(comp_data1D)    
iter_2D        = 0
array2D        = []
del1           = []
corrmax        = 1.00
skip2D         = 0
for _comp_data1D in range(comp_ncol):		#1D
    #column1
    column1 = comp_data1D[:, [_comp_data1D]]
    for _comp_data1Dnew in range(comp_ncol):    #1D
        if _comp_data1Dnew   >= _comp_data1D:
            iter_2D      += 1
            #column2
            column2       = comp_data1D[:, [_comp_data1Dnew]]                            
            column_2D     = column1 * column2
            array2D.append(column_2D)

comp_tag2D    = []
for _comp_tag in range(len(comp_tag)):      
	for _comp_tag1 in range(len(comp_tag)):      
		if _comp_tag <=_comp_tag1:
			new_tag = str(comp_tag[_comp_tag]) + "__X__" + str(comp_tag[_comp_tag1])
			comp_tag2D.append(new_tag)
#print("2D comp_tag:", comp_tag2D)
print("2D comp_tag:", len(comp_tag2D))


array1D2D              = np.concatenate((array2D), axis=1)

if comp_ncol1 != iter_2D:
    print("Something went wrong in data generation!")
    sys.exit()
    
comp_data_1D2D       = np.concatenate((comp_data1D, array1D2D), axis=1)
nrow_1D2D, ncol_1D2D = comp_data_1D2D.shape

#2D = 1D2D - 1D
comp_data2D      = comp_data_1D2D[:, comp_ncol:]
nrow_2D, ncol_2D = comp_data2D.shape

print("comp_data2D   :", comp_data2D.shape)
print("comp_data_1D2D:", comp_data_1D2D.shape)
print("Done")
print("")

#-------------3D compound features----------
print("3D comp. features:") #, comp_ncol1)
array3D        = []                                        # for saving MEMORY and time

iter_3D        = 0
comp_ncol3     = comp_ncol * ncol_2D
for _comp_data1D in range(comp_ncol):                      #1D
    #column1
    column1 = comp_data1D[:, [_comp_data1D]]
    for _comp_data2D in range(ncol_2D):                    #2D
        if _comp_data2D >= _comp_data1D:
            iter_3D     += 1
            #column2
            column2      = comp_data2D[:, [_comp_data2D]]
            column_3D    = column1 * column2               #3D
            array3D.append(column_3D)
            
#if comp_ncol3 != iter_3D:
#    print("Something went wrong in data generation!")
#    sys.exit()            

comp_data3D          = np.concatenate((array3D), axis=1)
comp_data_1D2D3D     = np.concatenate((comp_data_1D2D, comp_data3D), axis=1)
comp_nrow, comp_ncol = comp_data_1D2D3D.shape

print("comp_data3D     :", comp_data3D.shape)
print("comp_data_1D2D3D:", comp_data_1D2D3D.shape)
print("Done")
print("")



print("====================================================================")
print("Copying comp_data_1D2D3D ----------> comp_data")
comp_data    =   np.copy(comp_data_1D2D3D)
print("")

print("Removing duplicate features             :")
def unique_rows(array):
    array        = np.ascontiguousarray(array)
    unique_array = np.unique(array.view([('', array.dtype)]*array.shape[1]))
    return unique_array.view(array.dtype).reshape((unique_array.shape[0], array.shape[1]))

array       =  comp_data.T
comp_data   =  unique_rows(array).T		#transpose gives unique columns
print("comp_data with unique features         :", comp_data.shape)
print("WARNING: features are sorted.")
print("WARNING: fetch array indices for older pattern.")


print(" ")
print("Feature selection (varience threshold) :")
comp_sel   = VarianceThreshold(threshold=0.0)
comp_data  = comp_sel.fit_transform(comp_data)
print("comp_data                              :", comp_data.shape)

print("")
print("Scaling & centering of compound features:")
comp_data = StandardScaler().fit_transform(comp_data)
print("Done")
print("")


print("====================================================================")
print("Copying comp_data ---------> X")
print("Copying GW energy gap -----> y")
X = np.copy(comp_data)
y = np.copy(egap_gw)
print(" ")

print("====================================================================")
print("Boxplot for GW data:")
plt.figure(0)
bplot    = plt.boxplot(y)
print("Medians       :", [item.get_ydata()[0] for item in bplot['medians']])
#fliers  = [item.get_ydata() for item in bplot['fliers']]
fliers   = []
for _fliers in bplot['fliers']:
	_fliers  = _fliers.get_ydata()
	fliers.append(_fliers)

fliers  = np.asarray(fliers)
print("fliers        :", fliers.ravel())
print("")
print("Figure: boxplot_GW_egap.png")
plt.savefig("boxplot_GW_egap.png")
print("")

print("Removing outliers from GW data:")
outlier_index = []
_ycounter     = 0
for _y in y.ravel():
	for _fliers in fliers.ravel():
		if float(_y) == float(_fliers):	     #outlier listed
			outlier_index.append(_ycounter)
	_ycounter += 1

outlier_index = np.unique(outlier_index)
if len(outlier_index) > 0:
	for _outlier_index in outlier_index.ravel():
		print(_outlier_index, "    :   ", y[_outlier_index])

	X 	  = np.delete(X, outlier_index, 0)		#deleting outliers from y (and hence from X)
	y 	  = np.copy(np.delete(y, outlier_index, 0))
	prim_data = np.delete(prim_data, outlier_index, 0)      #remove corresponding samples from prim_data
	comp_data = np.delete(comp_data, outlier_index, 0)

	print("")
	print("X                 :", X.shape)
	print("y                 :", y.shape)

	comp_nrow, comp_ncol 	= comp_data.shape
	print("Updated X, y & comp_data!")
	
else:
	print("Outliers not found!")
print("Done")
print("")

print("====================================================================")
print("Training and test data:")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=27)         # train (2/3rd) and test data (1/3rd) 
print("X_train        :", X_train.shape)
print("y_train        :", y_train.shape)
print("X_test         :", X_test.shape)
print("y_test         :", y_test.shape)
print("")

print("====================================================================")
#LASSO
print("LASSO for Compound features:")

alpha_array          = []
r2_score_lasso_array = []
r2_score_lasso_arrayX= []
mse_lasso_array      = []

#when varying alpha, try alpha = [1e-3, 1e-2, 1e-1, 1]
#lasso                = Lasso(tol=0.001)
#alpha = 0.000442856705806
lasso                = LassoCV(tol=0.001)
print(lasso)
y_pred_lasso         = lasso.fit(X_train, y_train.ravel()).predict(X_test)    #use (X_train, y_train) on X_test to predict y
#y_pred_lasso         = lasso.fit(X_train, y_train.ravel()).predict(X_test)    #use (X_train, y_train) on X_test to predict y
r2_score_lasso       = r2_score(y_test, y_pred_lasso)                                                
lasso_coeff          = lasso.coef_                                    #store lasso coefficients
alpha                = lasso.alpha_

#MSE for test data
mse_lasso            = mean_squared_error(y_test, y_pred_lasso)  

print("alpha         =", alpha)
print("r^2           =", r2_score_lasso)
print("MSE           =", mse_lasso)
print("RMSE=sqrt(MSE)=", np.sqrt(mse_lasso))
print("")

plt.figure(1)
plt.scatter(y_test, y_pred_lasso)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
plt.title("LASSO regression fit.")
plt.xlabel('GW energy gap (test data)')
plt.ylabel('energy gap (predicted)')
print("Figure: lasso_fit.png")
plt.savefig("lasso_fit.png")
print("")

print("====================================================================")
print("Reduced compound features (after LASSO):")

ncoeff        =  0
coeff_index   =  []
for _lasso_coeff in range(len(lasso_coeff)):
    ncoeff    =  ncoeff + 1
    if lasso_coeff[_lasso_coeff] != 0:                                     #store non-zero coeffs
    	coeff_index.append(ncoeff)
print("Number of non-zero LASSO coeff.      :", len(coeff_index))

rcomp_data    =  []
for _coeff_index in range(len(coeff_index)):
	rcomp_data.append((comp_data[:,coeff_index[_coeff_index]]))        #reduced comp_data
    
rcomp_data = np.asarray(rcomp_data).T
#rcomp_data          = np.concatenate((rcomp_data), axis=1)                #reduced comp_data
print("comp_data (new)                       :", rcomp_data.shape)
print("LASSO done!")
print("")


print("====================================================================")

comp_data            = np.copy(rcomp_data)
comp_nrow, comp_ncol = comp_data.shape
corrmax = 0.75
print("Removing highly comp_data with corr > ", corrmax, ".......")
deletec  = []
for _comp_ncol in range(comp_ncol):
        column1 = (comp_data[:,[_comp_ncol]]).T
        for _comp_ncol_new in range(comp_ncol):
                column2 = (comp_data[:,[_comp_ncol_new]]).T
                if _comp_ncol < _comp_ncol_new:
                        corr    = np.corrcoef(column1, column2)[0, 1]
                        if corr > corrmax:
                                #print(_comp_ncol, " & ", _comp_ncol_new, ": corr=", corr)
                                deletec.append(_comp_ncol_new)

deletec = np.unique(deletec)
if len(deletec) > 0:
        print(len(deletec)," indices to be deleted:", deletec)
        comp_data = np.delete(comp_data, deletec, 1)
        print("")
        print("comp_data        (modified):", comp_data.shape)
        comp_nrow, comp_ncol = comp_data.shape
else:
        print("No feature to be deleted.")
print("")


print("====================================================================")
print("Training and test data (new):")
X                     = np.copy(comp_data)	
comp_nrow, comp_ncol  = X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=27)         # train (2/3rd) and test data (1/3rd) 
print("X_train        :", X_train.shape)
print("y_train        :", y_train.shape)
print("X_test         :", X_test.shape)
print("y_test         :", y_test.shape)
print("")

        
print("====================================================================")
print("RandonForestRegressor for reduced compound features:")

#Forest with n_estimators trees.
forest                = RandomForestRegressor(n_estimators=10, random_state=27)                                        
forest.fit(X_train, y_train.ravel())
y_pred_forest         = forest.predict(X_test)
print(forest)
print("")

#Find feature_importances_
importances = forest.feature_importances_
indices     = np.argsort(importances)[::-1]
#sd          = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)	#standard deviation

print("Feature ranking (selected) :")
ncomp_data  = []
for _comp_ncol in range(comp_ncol):
	if _comp_ncol <prim_ncol:		#top features based on feature_importances_
		print("%d. feature %d (%f)" % (_comp_ncol + 1, indices[_comp_ncol], importances[indices[_comp_ncol]]))
		ncomp_data.append(comp_data[:, indices[_comp_ncol]])				#store the new comp_data
comp_data  = np.asarray(ncomp_data).T

print("comp_data  (new)              :", comp_data.shape)
print("prim_data                     :", prim_data.shape)
print("")


plt.figure(2)
plt.title("(Reduced) Compound feature importances")
plt.bar(range(comp_ncol), importances[indices], color="r", align="center")
#plt.bar(range(comp_ncol), importances[indices], color="r", yerr=sd[indices], align="center")
plt.xticks(range(comp_ncol), indices)
plt.xlim([-1, prim_ncol])	 #only prim_ncol number of comp_data shown
plt.xlabel("Indices")
plt.ylabel("feature_importances_")
plt.savefig("comp-feature-importance.png")
print("Figure: comp-feature-importance.png")
print("Done")
print("")

print("====================================================================")
print("Linear regression (LR) and Kernel-Ridge regression (KRR) on prim_data & comp_data:")
print("")

prim_nrow, prim_ncol    = prim_data.shape
rmse_linear_prim    	= []
nfeature_prim           = 0

ncut = 10
###################################################

def linear_regression(X):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=27)  #train & test. 
	linear         	= LinearRegression()							    #Linear regression
	y_linear       	= linear.fit(X_train, y_train).predict(X_test)
	mse_val       	= mean_squared_error(y_test, y_linear)	                                    #MSE
	return mse_val

def kernel_ridge_regression(X):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=27)  #train & test. 
	KRR            	= GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=3, param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3], "gamma": np.logspace(-2, 2, 5)})         #KRR with CV
	y_KRR          	= KRR.fit(X_train, y_train).predict(X_test)
	mse_val       	= mean_squared_error(y_test, y_KRR)	                                    #MSE
	return mse_val

comb_data_set  =  ['prim_data', 'comp_data']
y_test_prim_array   = []            #arrays for multiplots
y_test_comp_array   = []
y_linear_prim_array = []
y_linear_comp_array = []
y_KRR_prim_array    = []
y_KRR_comp_array    = []

for _comb_data_set in range(len(comb_data_set)):
	print("")
	print(comb_data_set[_comb_data_set],":")

	if _comb_data_set == 0:
		comb_data = np.copy(prim_data)
	else:
		comb_data = np.copy(comp_data)
	    
	comb_nrow, comb_ncol = comb_data.shape  
	
	ctr = 0
	nelement  =np.arange(comb_ncol)

	for _nelement in nelement:
		#nCr  (here n & r are constants)                          
		r_cunter = 0
		new_list = list(itertools.combinations(nelement, _nelement+1))
		list_len = _nelement + 1

		#if ctr + 1 > prim_ncol:
		if ctr + 1 > ncut:
			break
		mse_arrayl  = []              
		rmse_arrayl = []              
		mse_arrayk  = []              
		rmse_arrayk = []              
		for _new_list in new_list:
			X 	 = np.copy(comb_data[:, _new_list])		#X
			comb_nrow, comb_ncol = X.shape

			#Linear Regression
			mse_vall  = linear_regression(X)                        #MSE through linear regression
			rmse_vall = np.sqrt(mse_vall) 				#RMSE

			mse_arrayl.append(mse_vall)
			rmse_arrayl.append(rmse_vall)

			#Kernel Ridge Regression
			mse_valk  = kernel_ridge_regression(X)                       #MSE through KRR
			rmse_valk = np.sqrt(mse_valk)                                #RMSE

			mse_arrayk.append(mse_valk)
			rmse_arrayk.append(rmse_valk)

		mse_min_indexl   = list(mse_arrayl).index(min(mse_arrayl))       #min. MSE index
		rmse_min_indexl  = list(rmse_arrayl).index(min(rmse_arrayl))     #min. RMSE index

		mse_min_indexk   = list(mse_arrayk).index(min(mse_arrayk))       #min. MSE index
		rmse_min_indexk  = list(rmse_arrayk).index(min(rmse_arrayk))     #min. RMSE index

		min_mse_vall     = mse_arrayl[mse_min_indexl]			#min. MSE LR
		min_rmse_vall    = rmse_arrayl[rmse_min_indexl]			#min. RMSE LR

		min_mse_valk     = mse_arrayk[mse_min_indexk]			#min. MSE KRR
		min_rmse_valk    = rmse_arrayk[rmse_min_indexk]			#min. RMSE KRR

		#LR--------------------------------------------------------------------------------------------
		print("LR : ", ctr + 1, ". Min. MSE = ", min_mse_vall, " & RMSE = ", min_rmse_vall)
		X 	 = np.copy(comb_data[:, new_list[rmse_min_indexl]])  #X
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=27)  #train & test. 
		linear         	= LinearRegression()          #Linear regression
		y_linear       	= linear.fit(X_train, y_train).predict(X_test)

		if _comb_data_set == 0:
			y_test_prim_array.append(y_test)                 #saving information
			y_linear_prim_array.append(y_linear)             #saving information
			y_test_prim = np.copy(y_test)

		else:
			y_test_comp_array.append(y_test)                 #saving information
			y_linear_comp_array.append(y_linear)             #saving information
			y_test_comp = np.copy(y_test)
        
		X_nrow, X_ncol  = X_test.shape
		#X_new           = np.random.rand(30,X_ncol)  
		#y_new          	= linear.fit(X_test, y_linear).predict(X_new)                #predict y_new for given x_new
		#XX = np.arange(len(y_new))
		print("")

		#KRR-------------------------------------------------------------------------------------------
		print("KRR: ", ctr + 1, ". Min. MSE = ", min_mse_valk, " & RMSE = ", min_rmse_valk)
		X 	 = np.copy(comb_data[:, new_list[rmse_min_indexk]])  #X
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=27)  #train & test. 
		KRR            	= GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=3, 
				param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3], "gamma": np.logspace(-2, 2, 5)})       #KRR with CV
		y_KRR          	= KRR.fit(X_train, y_train).predict(X_test)
        
		if _comb_data_set == 0:
			y_KRR_prim_array.append(y_KRR)             #saving information
		else:
			y_KRR_comp_array.append(y_KRR)             #saving information
		print("")
        
		X_nrow, X_ncol  = X_test.shape
		#y_new          	= KRR.fit(X_test, y_KRR).predict(X_new)                #predict y_new for given x_new
		#XX = np.arange(len(y_new))
		print("")
		ctr += 1


methods = ['r=1', 'r=2', 'r=3', 'r=4', 'r=5', 'r=6', 'r=7', 'r=8', 'r=9', 'r=10', 
           'r=1', 'r=2', 'r=3', 'r=4', 'r=5', 'r=6', 'r=7', 'r=8', 'r=9', 'r=10']
#------------------------------------------------------------------------------

plt.close('all')

#------------------------------------------------------------------------------
print("====================================================================")
print("Plot for prim_data (LR)")
#------------------------------------------------------------------------------
nrow = 2
ncol = 5
fig, axarr = plt.subplots(nrow, ncol, figsize=(10, 4))  
fig.text(0.5, 1.00, "Linear regression: Fits per min. RMSE nCr prim_data", ha="center")
fig.subplots_adjust(hspace=0.3, wspace=0.05)

fig.text(0.5, 0.00, 'egap GW (eV)', ha='center')
fig.text(0.00, 0.5, 'egap (predicted) (eV)', va='center', rotation='vertical')
ctr = 0
for _nrow in range(nrow):
    for _ncol in range(ncol):
        axarr[_nrow, _ncol].set_title(r"${}$".format(methods[ctr]))
        axarr[_nrow, _ncol].scatter(y_test_prim_array[ctr],y_linear_prim_array[ctr], color='g')
    
        ymin1 = y_linear_prim_array[ctr].min()
        ymax1 = y_linear_prim_array[ctr].max()
        axarr[_nrow, _ncol].plot([ymin1, ymax1], [ymin1, ymax1], 'k--', lw=3)
    
        ctr += 1

        if _nrow != nrow-1 and _ncol != 0:
            plt.setp([a.get_xticklabels() for a in axarr[_nrow, :]], visible=False)
            plt.setp([a.get_yticklabels() for a in axarr[:, _ncol]], visible=False)
plt.tight_layout()

file = "LR-prim-nCr-fit.png" 

print("Figure: ", file)
plt.savefig(file)
print("Done")
print("")
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
print("====================================================================")
print("Plot for comp_data (LR)")
#------------------------------------------------------------------------------
nrow = 2
ncol = 5
fig, axarr = plt.subplots(nrow, ncol, figsize=(10, 4))  
fig.text(0.5, 1.00, "Linear regression: Fits per min. RMSE nCr comp_data", ha="center")
fig.subplots_adjust(hspace=0.3, wspace=0.05)

fig.text(0.5, 0.00, 'egap GW (eV)', ha='center')
fig.text(0.00, 0.5, 'egap (predicted) (eV)', va='center', rotation='vertical')
ctr = 0
for _nrow in range(nrow):
    for _ncol in range(ncol):
        axarr[_nrow, _ncol].set_title(r"${}$".format(methods[ctr]))
        axarr[_nrow, _ncol].scatter(y_test_comp_array[ctr],y_linear_comp_array[ctr], color='g')
    
        ymin1 = y_linear_comp_array[ctr].min()
        ymax1 = y_linear_comp_array[ctr].max()
        axarr[_nrow, _ncol].plot([ymin1, ymax1], [ymin1, ymax1], 'k--', lw=3)
    
        ctr += 1

        if _nrow != nrow-1 and _ncol != 0:
            plt.setp([a.get_xticklabels() for a in axarr[_nrow, :]], visible=False)
            plt.setp([a.get_yticklabels() for a in axarr[:, _ncol]], visible=False)
plt.tight_layout()

file = "LR-comp-nCr-fit.png" 

print("Figure: ", file)
plt.savefig(file)
print("Done")
print("")
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
print("====================================================================")
print("Plot for prim_data (KRR)")
#--------------------------------------------------------------------------------
nrow = 2
ncol = 5
fig, axarr = plt.subplots(nrow, ncol, figsize=(10, 4))  
fig.text(0.5, 1.00, "Kernel ridge regression: Fits per min. RMSE nCr prim_data", ha="center")
fig.subplots_adjust(hspace=0.3, wspace=0.05)

fig.text(0.5, 0.00, 'egap GW (eV)', ha='center')
fig.text(0.00, 0.5, 'egap (predicted) (eV)', va='center', rotation='vertical')
ctr = 0
for _nrow in range(nrow):
    for _ncol in range(ncol):
        axarr[_nrow, _ncol].set_title(r"${}$".format(methods[ctr]))
        axarr[_nrow, _ncol].scatter(y_test_prim_array[ctr],y_KRR_prim_array[ctr], color='g')
    
        ymin1 = y_KRR_prim_array[ctr].min()
        ymax1 = y_KRR_prim_array[ctr].max()
        axarr[_nrow, _ncol].plot([ymin1, ymax1], [ymin1, ymax1], 'k--', lw=3)
        ctr += 1

        if _nrow != nrow-1 and _ncol != 0:
            plt.setp([a.get_xticklabels() for a in axarr[_nrow, :]], visible=False)
            plt.setp([a.get_yticklabels() for a in axarr[:, _ncol]], visible=False)
plt.tight_layout()

file = "KRR-prim-nCr-fit.png" 

print("Figure: ", file)
plt.savefig(file)
print("Done")
print("")
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
print("====================================================================")
print("Plot for comp_data (KRR)")
#--------------------------------------------------------------------------------
nrow = 2
ncol = 5
fig, axarr = plt.subplots(nrow, ncol, figsize=(10, 4))  
fig.text(0.5, 1.00, "Kernel ridge regression: Fits per min. RMSE nCr comp_data", ha="center")
fig.subplots_adjust(hspace=0.3, wspace=0.05)

fig.text(0.5, 0.00, 'egap GW (eV)', ha='center')
fig.text(0.00, 0.5, 'egap (predicted) (eV)', va='center', rotation='vertical')
ctr = 0
for _nrow in range(nrow):
    for _ncol in range(ncol):
        axarr[_nrow, _ncol].set_title(r"${}$".format(methods[ctr]))
        axarr[_nrow, _ncol].scatter(y_test_comp_array[ctr],y_KRR_comp_array[ctr], color='g')
    
        ymin1 = y_KRR_comp_array[ctr].min()
        ymax1 = y_KRR_comp_array[ctr].max()
        axarr[_nrow, _ncol].plot([ymin1, ymax1], [ymin1, ymax1], 'k--', lw=3)
        ctr += 1

        if _nrow != nrow-1 and _ncol != 0:
            plt.setp([a.get_xticklabels() for a in axarr[_nrow, :]], visible=False)
            plt.setp([a.get_yticklabels() for a in axarr[:, _ncol]], visible=False)
plt.tight_layout()

file = "KRR-comp-nCr-fit.png" 

print("Figure: ", file)
plt.savefig(file)
print("Done")
#--------------------------------------------------------------------------------

print("")
print("END")
print("====================================================================")




#close the opened files.
file1.close()
gwfile.close()
fout.close()

#run time
end = time.time()
m, s = divmod(end - start, 60)
h, m = divmod(m, 60)
print("Run time: %d:%02d:%02d" % (h, m, s)) 

