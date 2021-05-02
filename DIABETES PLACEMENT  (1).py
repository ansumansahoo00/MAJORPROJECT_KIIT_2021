#!/usr/bin/env python
# coding: utf-8

# In[ ]:


###Diabetes Prediction Using Logistic Regression and KNN Algorithm########
##########################################################################
import pandas as pd
from tkinter import messagebox as m
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
col_names=['pregnant','glucose','bp','skin','insulin','bmi','pedigree','age','label']
pima=pd.read_csv("C:/Users/KIIT/Desktop/pima.csv",names=col_names)
feature=['pregnant','glucose','bp','skin','insulin','bmi','pedigree','age']
######################################################################
#dataset cleaning and transformation#######################################
#calculate the median value for a specific column and substitute
#that value everywhere (in the same column) we have zero or null.

# Calculate the median value for BMI
median_bmi = pima['bmi'].median()
##Substitute it in the bmi column of the # dataset where values are 0
pima['bmi'] = pima['bmi'].replace(to_replace=0, value=median_bmi)
# Calculate the median value for plasma glucose concentration(glucose)
median_glucose = pima['glucose'].median()
# Substitute it in the glucose column of the # dataset where values are 0
pima['glucose'] = pima['glucose'].replace(to_replace=0, value=median_glucose)
####################################################################
#calculate the median value of skin thickness
median_skin=pima['skin'].median()
## Substitute it in the skin column of the # dataset where values are 0
pima['skin'] = pima['skin'].replace(to_replace=0, value=median_skin)
######################################################################
#calculate the median value of insulin
median_insulin=pima['insulin'].median()
## Substitute it in the insulin column of the # dataset where values are 0
pima['insulin'] = pima['insulin'].replace(to_replace=0, value=median_insulin)
print(pima)
####################################################################
#####################################################################
X=pima[feature]
Y=pima.label
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=5)
#################Apply naive Bayes algorithm
naiveaccuracy=0
def naive():
    from sklearn.naive_bayes import BernoulliNB
    global naiveaccuracy
    global NAIVE
    NAIVE=BernoulliNB()
    params = {'alpha': [0.01, 0.1, 0.5, 1.0, 10.0],
         }

    bernoulli_nb_grid = GridSearchCV(NAIVE, param_grid=params, n_jobs=-1, cv=5, verbose=5)
    bernoulli_nb_grid.fit(X_train,Y_train)
    print(bernoulli_nb_grid.best_params_)
    print(" best accuracy of naive bayes is",bernoulli_nb_grid.best_score_)
    Y_pred=bernoulli_nb_grid.predict(X_test)
    
    from sklearn import metrics
    confusion=metrics.confusion_matrix(Y_test,Y_pred)
    print(confusion)
    TP=confusion[1,1]#True Positive
    TN=confusion[0,0]#True Negative
    FP=confusion[0,1]#False Positive
    FN=confusion[1,0]#False Negative
    sensitivity=TP/(TP+FN)#When actual value is positive how many prediction True
    print("sensitivity is",sensitivity)                      
    specificity=TN/(TN+FP)#When actual value is negative how many prediction True
    print("specificity is",specificity)
    
    naiveaccuracy=round(((TP+TN)/(TP+TN+FP+FN)),2)*100
    m.showinfo(title="Accuracy",message="Accuracy is"+str(naiveaccuracy)+"%")
    print(metrics.accuracy_score(Y_test,Y_pred))
    #####Apply Decision Tree Algorithm####    
dtreeaccuracy=0
def dtree():
    from sklearn.tree import DecisionTreeClassifier
    global dtaccuracy
    global DTREE
    DTREE=DecisionTreeClassifier(random_state=1234)
    params = {'max_features': ['auto', 'sqrt', 'log2'],
          'min_samples_split': [2,3,4,5,6,7,8,9,10,11,12,13,14,15], 
          'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10,11],
          'random_state':[123]}
    model1 = GridSearchCV(DTREE, param_grid=params,n_jobs=-1)
    model1.fit(X_train,Y_train)
    #The best hyper parameters set
    print("Best Hyper Parameters:",model1.best_params_)
    print("best accuracy of decision tree is",model1.best_score_)
    Y_pred=model1.predict(X_test)
    from sklearn import metrics
    confusion=metrics.confusion_matrix(Y_test,Y_pred)
    print(confusion)
    TP=confusion[1,1]#True Positive
    TN=confusion[0,0]#True Negative
    FP=confusion[0,1]#False Positive
    FN=confusion[1,0]#False Negative
    sensitivity=TP/(TP+FN)#When actual value is positive how many prediction True
    print("sensitivity is",sensitivity)                      
    specificity=TN/(TN+FP)#When actual value is negative how many prediction True
    print("specificity is",specificity)
    dtaccuracy=round(((TP+TN)/(TP+TN+FP+FN)),2)*100
    m.showinfo(title="Accuracy",message="Accuracy is"+str(dtaccuracy)+"%")
    print(metrics.accuracy_score(Y_pred,Y_test))
    ############################################################################
#####################APPLY KNN ALGORITHM####################################
from sklearn.neighbors import KNeighborsClassifier
knnaccuracy=0###global

KNN=KNeighborsClassifier(n_jobs=-1)

def knn():
    '''Apply KNN algorithm to data set'''
    
    global knnaccuracy
    global KNN
    KNN=KNeighborsClassifier(n_jobs=-1)
    params = {'n_neighbors':[5,6,7,8,9,10],
          'leaf_size':[1,2,3,5],
          'weights':['uniform', 'distance'],
          'algorithm':['auto', 'ball_tree','kd_tree','brute'],
          'n_jobs':[-1]}
    model1 = GridSearchCV(KNN, param_grid=params, n_jobs=1)
    model1.fit(X_train,Y_train)
    print("Best Hyper Parameters:\n",model1.best_params_)
    print("best accuracy of knn is",model1.best_score_)
    prediction=model1.predict(X_test)
    
###############################################################################
                            #confusion matrix
#it is used to check the performance of the algorithm
#find sensitivity,Specificity and Accuracy
##############################################################################
    from sklearn import metrics
    confusion=metrics.confusion_matrix(Y_test,prediction)
    print(confusion)
    TP=confusion[1,1]#True Positive
    TN=confusion[0,0]#True Negative
    FP=confusion[0,1]#False Positive
    FN=confusion[1,0]#False Negative
    sensitivity=TP/(TP+FN)#When actual value is positive how many prediction True
    print("sensitivity is",sensitivity)                      
    specificity=TN/(TN+FP)#When actual value is negative how many prediction True
    print("specificity is",specificity)
    knnaccuracy=round(((TP+TN)/(TP+TN+FP+FN)),2)*100
    m.showinfo(title="Accuracy",message="Accuracy is"+str(knnaccuracy)+"%")
    print(metrics.accuracy_score(prediction,Y_test))
###########################################################################
####################APPLY LOGISTIC REGRESSION#########################
from sklearn.linear_model import LogisticRegression
#LOGREG=LogisticRegression()
logaccuracy=0
def logreg():
    '''Apply Logistic Regression to Data Set'''
    
    global logaccuracy
    global LOGREG
    LOGREG=LogisticRegression(solver='liblinear',multi_class='auto')
    LRparam_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'penalty': ['l2'],
    'max_iter': list(range(100,800,100)),
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
}
    LR_search = GridSearchCV(LOGREG, param_grid=LRparam_grid, refit = True, verbose = 3, cv=None)
    LR_search.fit(X_train,Y_train)
    print("best hyper parameters are", LR_search.best_params_)
    print("best accuracy of logistic regression is",LR_search.best_score_)
    Y_pred=LR_search.predict(X_test)
    from sklearn import metrics
    confusion=metrics.confusion_matrix(Y_test,Y_pred)
    print(confusion)
    TP=confusion[1,1]#True Positive
    TN=confusion[0,0]#True Negative
    FP=confusion[0,1]#False Positive
    FN=confusion[1,0]#False Negative
    sensitivity=TP/(TP+FN)#When actual value is positive how many prediction True
    print("sensitivity is",sensitivity)                      
    specificity=TN/(TN+FP)#When actual value is negative how many prediction True
    print("specificity is",specificity)
    logaccuracy=round(((TP+TN)/(TP+TN+FP+FN)),2)*100
    m.showinfo(title="Accuracy",message="Accuracy is"+str(logaccuracy)+"%")
    
    print(metrics.accuracy_score(Y_test,Y_pred))
from sklearn.svm import SVC
from sklearn import svm
svmaccuracy=0
def SVM():
    '''Apply SVM Classifier to dataset'''
    
    global svmaccuracy
    global SVM
    SVM=svm.SVC()
    params = {'C': [6,7,8,9,10,11,12], 
          'kernel': ['linear','rbf']}
    model1 = GridSearchCV(SVM, param_grid=params, n_jobs=-1)
    model1.fit(X_train,Y_train)
    print("Best Hyper Parameters of SVM are:\n",model1.best_params_)
    print("best accuracy of SVM is",model1.best_score_)
    Y_pred=model1.predict(X_test)
    from sklearn import metrics
    confusion=metrics.confusion_matrix(Y_test,Y_pred)
    TP=confusion[1,1]
    TN=confusion[0,0]
    FP=confusion[0,1]
    FN=confusion[1,0]
    sensitivity=TP/(TP+FN)                  
    specificity=TN/(TN+FP)
    svmaccuracy=round(((TP+TN)/(TP+TN+FP+FN)),2)*100
    m.showinfo(title="Accuracy",message="Accuracy is"+str(svmaccuracy)+"%")



def compare():
    '''compare the 4 algorithm by using bar'''
    result=[knnaccuracy,logaccuracy,dtaccuracy,naiveaccuracy,svmaccuracy]
    clf=['KNN','LOG REG','DTREE','NAIVEBAYES','SVM']
    import matplotlib.pyplot as plt
    plt.bar(clf,result,color=['green','orange','blue','yellow'])
    plt.ylabel("accuracy")
    plt.xlabel("Model")
    plt.title("Model acuracy comparasion")
    plt.show()
    
from tkinter import *
w=Tk()
w.geometry("1200x800")
w.title("Diabetes Prediction System")
w.resizable(0,0)
vpreg=StringVar()
vglucose=StringVar()
vbp=StringVar()
vskin=StringVar()
vinsulin=StringVar()
vbmi=StringVar()
vpedegree=StringVar()
vage=StringVar()
def validate():
    if vpreg.get()=="" or vglucose.get()==""or vbp.get()=="" or vskin.get()==""       or vinsulin.get()=="" or vbmi.get()=="" or vpedegree.get()=="" or vage.get()=="":
        m.showinfo(title="Wrong Input",message="please fill up all details")
        return
def predict():
    ''' It will predict the status of a new patient'''
    global KNN
    global LOGREG
    #global knnaccuracy
    global logaccuracy
    global svmaccuracy
    knn()
    logreg()
    x=0
    while x==0:
    
        
        try:
            a=float(vpreg.get())
            b=float(vglucose.get())
            c=float(vbp.get())
            d=float(vskin.get())
            e=float(vinsulin.get())
            f=float(vbmi.get())
            g=float(vpedegree.get())
            h=float(vage.get())
            x=1
            validate()
           
        except:
        
            reset()
        
            btnpredict.configure(command=predict)
            x=0
            vpreg.set("")
    
        #####we found hoghest accuracy is logistic
        ####regression
    
    l=SVM.predict([[a,b,c,d,e,f,g,h]])
    if l==0:
        m.showinfo(title="Diabetes Prediction",message="You Have No Diabetes")
    else:
        m.showinfo(title="Diabetes Prediction",message="You have Diabetes or may get soon")
            
       
    
   
def reset():
    vpreg.set("")
    vglucose.set("")
    vbp.set("")
    vskin.set("")
    vinsulin.set("")
    vbmi.set("")
    vpedegree.set("")
    vage.set("")
    
img=PhotoImage(file="C:/Users/KIIT/Desktop/IMAG.png")
lblimage=Label(w,image=img)
lblimage.grid(row=1,column=1,rowspan=9)
labeltitle=Label(w,text="Enter Your Details!!!!",fg='red',font=('arial',20,'bold'))
labeltitle.grid(row=1,column=2,columnspan=2)
labelpreg=Label(w,text="Pregnant",font=('arial',20,'bold'))
labelpreg.grid(row=2,column=2)
entrypreg=Entry(w,font=('arial',20,'bold'),textvariable=vpreg)
entrypreg.grid(row=2,column=3)
labelglucose=Label(w,text="Glucose",font=('arial',20,'bold'))
labelglucose.grid(row=3,column=2)
entryglucose=Entry(w,font=('arial',20,'bold'),textvariable=vglucose)
entryglucose.grid(row=3,column=3)
labelbp=Label(w,text="Blood Pressure",font=('arial',20,'bold'))
labelbp.grid(row=4,column=2)
entrybp=Entry(w,font=('arial',20,'bold'),textvariable=vbp)
entrybp.grid(row=4,column=3)
labelskin=Label(w,text="Skin",font=('arial',20,'bold'))
labelskin.grid(row=5,column=2)
entryskin=Entry(w,font=('arial',20,'bold'),textvariable=vskin)
entryskin.grid(row=5,column=3)
labelinsulin=Label(w,text="Insulin",font=('arial',20,'bold'))
labelinsulin.grid(row=6,column=2)
entryinsulin=Entry(w,font=('arial',20,'bold'),textvariable=vinsulin)
entryinsulin.grid(row=6,column=3)
labelbmi=Label(w,text="Body Mass Index",font=('arial',20,'bold'))
labelbmi.grid(row=7,column=2)
entrybmi=Entry(w,font=('arial',20,'bold'),textvariable=vbmi)
entrybmi.grid(row=7,column=3)

labelpedegree=Label(w,text="Pedegree",font=('arial',20,'bold'))
labelpedegree.grid(row=8,column=2)
entrypedegree=Entry(w,font=('arial',20,'bold'),textvariable=vpedegree)
entrypedegree.grid(row=8,column=3)

labelage=Label(w,text="Age",font=('arial',20,'bold'))
labelage.grid(row=9,column=2)
entryage=Entry(w,font=('arial',20,'bold'),textvariable=vage)
entryage.grid(row=9,column=3)
btnpredict=Button(w,text="Predict",bg='yellow',width=10,relief='groove',font=('arial',20,'bold'),fg='green',command=predict)
btnpredict.grid(row=10,column=2)
btnreset=Button(w,text="Reset",bg='yellow',width=10,relief='groove',font=('arial',20,'bold'),fg='green',command=reset)
btnreset.grid(row=10,column=3)
btnknn=Button(w,text="  KNN  ",bg='cyan',font=('arial',20,'bold'),command=knn)
btnknn.grid(row=10,column=1)
btndt=Button(w,text="Decision Tree",bg='cyan',font=('arial',20,'bold'),command=dtree)
btndt.grid(row=11,column=1)
btndt=Button(w,text="Naive Bayes",bg='cyan',font=('arial',20,'bold'),command=naive)
btndt.grid(row=12,column=1)
btnlogreg=Button(w,text="Logistic Regression",bg='cyan',font=('arial',20,'bold'),command=logreg)
btnlogreg.grid(row=13,column=1)
btnlogreg=Button(w,text="SVM",bg='#A7B0B2',font=('arial',20,'bold'),fg='white',command=SVM)
btnlogreg.grid(row=14,column=1)
btncompare=Button(w,text="Compare",bg='crimson',border=10,font=('arial',20,'bold'),command=compare)
btncompare.place(relx = 0.5, rely = 0.9, anchor = CENTER) 
w.mainloop()


# In[ ]:





# In[ ]:




