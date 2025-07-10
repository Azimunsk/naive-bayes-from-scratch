import pandas as pd
import numpy as np
import sys
import math

def compute_gaussian_probab(x, mean, var):
    res = 1
    for i in range(0,len(x)):
        exponent = math.exp(-((x[i]-mean[i])**2/ (2*var[i])))
        res *= (1/ (math.sqrt(2*math.pi*var[i])))*exponent
    return res

def main():
    df = pd.read_csv('wheat-seeds.csv')
    column_names = ['area', 'perimeter', 'compactness', 'length', 'width', 'asymmetry_coefficient', 'groove_length', 'class']
    df.columns = column_names
    #print(df.head())
    #print(df.shape)
    '''print(df.head())
    print(df.shape)
    print(df.columns)
    print(df.dtypes)
    print(df.isna().sum())'''
    dfrandom = df.sample(frac=1, random_state=1119).reset_index(drop=True)
    #print(dfrandom.head())
    #----seperate data into training and test parts----
    dftrain = dfrandom.iloc[0:190,:]
    #print(dftrain.head())
    #print(dftrain.shape)
    dftest = dfrandom.iloc[100:,:]
    #print(dftest.head())
    #print(dftest.shape)
    #----assemble data by categories----
    dfone = dfrandom[dfrandom['class']==1]
    #print(dfone)
    dftwo = dfrandom[dfrandom['class']==2]
    #print(dftwo)
    dfthree = dfrandom[dfrandom['class']==3]
    #print(dfthree)
    #----find mean of each class----
    mean_one = dfone.iloc[:,0:7].mean(axis = 0)
    #print('mean one\n', mean_one)
    mean_two = dftwo.iloc[:,0:7].mean(axis=0)
    #print('mean two\n', mean_two)
    mean_three = dfthree.iloc[:,0:7].mean(axis=0)
    #print('mean three\n', mean_three)
    #----find variance of each class----
    var_one = dfone.iloc[:,0:7].var(axis=0)
    #print('var on\n', var_one)
    var_two = dftwo.iloc[:,0:7].var(axis=0)
    #print('var two\n', var_two)
    var_three = dfthree.iloc[:,0:7].var(axis=0)
    #print('var three\n', var_three)
    #---- do prediction on test set via Naive Bayes----
    count_correct = 0
    print(len(dftest))
    for i in range(0,len(dftest)):
        x = dftest.iloc[i,0:7].values
        probc1 = compute_gaussian_probab(x,mean_one.values,var_one.values)
        probc2 = compute_gaussian_probab(x,mean_two.values,var_two.values)
        probc3 = compute_gaussian_probab(x,mean_three.values,var_three.values)
        probs = np.array([probc1, probc2, probc3])
        maxindex = probs.argmax(axis=0)

        if dftest.iloc[i,7] == 1:
            index = 0
        if dftest.iloc[i,7] == 2:
            index = 1
        if dftest.iloc[i,7] == 3:
            index = 2
        if maxindex == index:
            count_correct = count_correct+1
        #print(probc1,' ', probc2,' ', probc3,' class=',dftest.iloc[i,7])
    print('classification accuracy =', count_correct/len(dftest)*100)
if __name__ == '__main__':
    main()