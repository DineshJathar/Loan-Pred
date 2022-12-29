#!/usr/bin/env python
# coding: utf-8

# # Loan Prediction Analysis

# ## Data Description : 
# ### Getting familer with varibles of data 

# 
# ### 1. Loan_ID                       : Unique Loan ID
# ### 2. Gender                         :Male/Female
# ### 3. Married                         : Applicant Married Status(Y/N)
# ### 4. Dependents                 : Number of Depedents
# ### 5. Education                     : Applicant Education (Graduate/Under Graduate)
# ### 6. Self_Employed            : Y/N
# ### 7. Applicant Income       : Income of Applicant
# ### 8. Coapplicant Income   : Income of Coapplicant
# ### 9. Loan Amount              : Loan Amount In Thousand
# ### 10.Loan_Amount_Term  : Term of Loan Amount In months
# ### 11.Credit_History            :  Credit_History meets guidelines (1/0)
# ### 12.Property_Area            : Urban/Semi Ubran / Rural
# ### 13.Loan_Status               : (Target) Loan Approved or not (Y/N)

# # Objectives:
# ### 1) To find which factors affect the approval of loan.
# ### 2) To predict loan will approved or not.

# # Importing requried libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings
warnings.filterwarnings('ignore')


# In[2]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# # Reading Data

# In[3]:


train_data=pd.read_csv("C:/Users/dines/Downloads/train_ctrUa4K (1).csv")


# In[4]:


train_data.head()


# In[5]:


train_data.shape   # Checking Shape of dataset


# In[6]:


train_data.info()


# In[7]:


train_data.describe()    # For Numerical Columns


# In[8]:


train_data.describe(include="object")


# # Checking missing values and replacing it by appropriate method

# In[9]:


train_data.isnull().sum()


# In[10]:


# Filling missing values for categorical  varibles by appropriate method (Mode)
train_data["Gender"].fillna(train_data["Gender"].mode()[0],inplace=True)
train_data["Married"].fillna(train_data["Married"].mode()[0],inplace=True)
train_data["Credit_History"].fillna(train_data["Credit_History"].mode()[0],inplace=True)
train_data["Self_Employed"].fillna(train_data["Self_Employed"].mode()[0],inplace=True)
train_data["Dependents"].fillna(train_data["Dependents"].mode()[0],inplace=True)


# In[11]:


# Filling missing values of numeric varibles by appropriate method (median)
train_data["LoanAmount"].fillna(train_data["LoanAmount"].median(),inplace=True)
train_data["Loan_Amount_Term"].fillna(train_data["Loan_Amount_Term"].median(),inplace=True)


# # Outlier Detection & Handling

# In[12]:


plt.figure(figsize=(10,5))
plt.subplot(1,3,1)
sns.boxplot(train_data["ApplicantIncome"])

plt.subplot(1,3,2)
sns.boxplot(train_data["CoapplicantIncome"])

plt.subplot(1,3,3)
sns.boxplot(train_data["LoanAmount"])


# ### Here we can see that outliers are present in the dataset for this 3 varibles.so we have to take care of outliers while doing further analysis.
# ### So we remove some outliers from data.

# In[13]:


print("Before removing outliers shape of data is :",train_data.shape)

train_data=train_data[train_data["ApplicantIncome"]<25000]
train_data=train_data[train_data["CoapplicantIncome"]<12000]
train_data=train_data[train_data["LoanAmount"]<400]

print("After removing outliers shape of data is :",train_data.shape)


# # Data Visualization

# In[14]:


fig,axs = plt.subplots(nrows=2, ncols=3, figsize=(16,8))
sns.countplot(data=train_data, x="Gender",ax=axs[0,0])
sns.countplot(data=train_data, x="Married",ax=axs[0,1])
sns.countplot(data=train_data, x="Credit_History",ax=axs[0,2])
sns.countplot(data=train_data, x="Self_Employed",ax=axs[1,0])
sns.countplot(data=train_data, x="Dependents",ax=axs[1,1])
sns.countplot(data=train_data, x="Education",ax=axs[1,2])


# ###  1.  Count of male applicant is more than female.
# ###  2.  Count of married applicant is more than unmarried.
# ###  3.  Credit history is present for many applicants.
# ###  4.  Count of Self employed is less than that of non-self-employed
# ### 5.   Count of Graduate is more than under graduate

# In[15]:


sns.countplot(data=train_data,x="Loan_Status")


# ### Here we can see that more loans are approved Vs rejected

# In[16]:


train_data=train_data.drop(["Loan_ID"],axis=1)


# In[17]:


cat=train_data.select_dtypes("object").columns.tolist()
cat


# In[18]:


for i in cat[:-1]:
    plt.figure(figsize=(15,10))
    plt.subplot(2,3,1)
    sns.countplot(x=i,hue="Loan_Status",data=train_data)
    plt.xlabel(i)


# In[19]:


sns.distplot(train_data['ApplicantIncome'])
plt.title("Distribution plot of Applicant Income")


# ### Here from the distribution plot of applicant income we see that most of applicant have income less than 20000.

# In[20]:


sns.distplot(train_data["CoapplicantIncome"])
plt.xlabel("Coapplicant Income")
plt.title("Distribution plot of Coapplicant Income")


# In[21]:


sns.distplot(train_data["LoanAmount"])
plt.title("Distribution plot of Loan amount")


# # Label Encoding

# In[22]:


train_data["Gender"]=train_data["Gender"].replace(("Male","Female"),(1,0))
train_data["Married"]=train_data["Married"].replace(("Yes","No"),(1,0))
train_data["Education"]=train_data["Education"].replace(("Graduate","Not Graduate"),(1,0))
train_data["Self_Employed"]=train_data["Self_Employed"].replace(("Yes","No"),(1,0))
train_data["Property_Area"]=train_data["Property_Area"].replace(("Urban","Semiurban","Rural"),(0,1,2))
train_data["Loan_Status"]=train_data["Loan_Status"].replace(("Y","N"),(1,0))


# In[23]:


train_data.head()


# # Checking Multicolliearity in the data 

# In[24]:


plt.figure(figsize=(16,8))
sns.heatmap(train_data.corr(),annot=True,fmt='.2f', linewidths=2)


# ### From the above heatmap we see that there is not significane correlation between any indepedent varibles. 
# ### So there is no multicollinearity in the data.

# # Splitting data into dependent and independent varibles 

# In[25]:


x=train_data.drop(["Loan_Status"],axis=1)
y=train_data["Loan_Status"]


# In[26]:


from sklearn.model_selection import train_test_split


# In[27]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)


# In[28]:


x_train.shape


# In[29]:


x_test.shape


# In[30]:


#from sklearn.preprocessing import StandardScaler
#sc=StandardScaler()
#x_train=sc.fit_transform(x_train)
#x_test=sc.fit_transform(x_test)


# # Logistic Regression

# In[31]:


lr=LogisticRegression()


# In[32]:


lr.fit(x_train,y_train)


# In[33]:


y_pred_lr=lr.predict(x_test)


# In[34]:


from sklearn import metrics
from sklearn.metrics import confusion_matrix,precision_score,recall_score,accuracy_score,classification_report


# In[35]:


cm=confusion_matrix(y_test,y_pred_lr)
plt.figure(figsize=(8,4))
plt.title("Confusion Matrix for Logistic Regreesion")
sns.heatmap(cm,annot=True,fmt="d")
plt.xlabel("Actual values")
plt.ylabel("Predicted Values")


# In[36]:


print("Accuarcy Score for Logistic Regression is :",accuracy_score(y_test,y_pred_lr))
print("Recall Score for Logistic Regression  is :",recall_score(y_test,y_pred_lr))
print("Precision Score for Logistic Regression  is :",precision_score(y_test,y_pred_lr))


# ### Here we got 82.43 % accuracy for Logistic Regression Model and aslo recall score and precision is also quite good.
# ### But we will also see other models for best fit.

# # Decision Tree

# In[38]:


DT=DecisionTreeClassifier(criterion="entropy")


# In[39]:


DT.fit(x_train,y_train)


# In[40]:


y_pred_DT=DT.predict(x_test)


# In[41]:


y_pred_DT


# In[42]:


cm=confusion_matrix(y_test,y_pred_DT)
plt.figure(figsize=(8,4))
plt.title("Confusion Matrix for Decision Tree")
sns.heatmap(cm,annot=True,fmt="d")
plt.xlabel("Actual values")
plt.ylabel("Predicted Values")


# In[43]:


print("Accuarcy Score for Decision Tree is :",accuracy_score(y_pred_DT,y_test))
print("Recall Score For Decision Tree  is :",recall_score(y_test,y_pred_DT))
print("Precision Score For Decision Tree  is :",precision_score(y_test,y_pred_DT))


# ### Here our accuarcy score is not good ( which is less than Logistic Regression model) .So we will move forward for another model.
# ### So we will not prefer Decision Tree model for future prediction.

# In[45]:


feature_imp=pd.Series(DT.feature_importances_).sort_values(ascending=False)


# In[46]:


sns.barplot(x=feature_imp,y=x.columns)
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Visulizing Important Features")
plt.show()


# In[47]:


from sklearn import tree


# In[48]:


feature_name=x.columns
class_name=["Loan Approved","Loan Not Apporved"]


# In[49]:


fig=plt.figure(figsize=(25,20))
tree.plot_tree(DT,feature_names=x.columns,class_names=class_name,filled=True,max_depth=5)
plt.show()


# # Random Forest

# In[50]:


RF=RandomForestClassifier(criterion="entropy",random_state=10)


# In[51]:


RF.fit(x_train,y_train)


# In[52]:


y_pred_RF=RF.predict(x_test)


# In[53]:


cm=confusion_matrix(y_test,y_pred_RF)
plt.figure(figsize=(8,4))
sns.heatmap(cm,annot=True,fmt="d")
plt.xlabel("Actual Values")
plt.ylabel("Prediced Values")
plt.title("Confusion Matrix for Random Forest")


# In[54]:


print("Accuracy Score for Random Forest Model is :",accuracy_score(y_test,y_pred_RF))
print("Recall Score For Random Forest Model  is :",recall_score(y_test,y_pred_RF))
print("Precision Score For Random Forest Model  is :",precision_score(y_test,y_pred_RF))


# ### Here accuracy score is 80.40 (which is better than decision tree but not better than Logistic Regression.)

# ### Here we tried to get important features from Random Forest Model then using that features bulid the new Random forest Model . 

# In[55]:


feature_imp=pd.Series(RF.feature_importances_).sort_values(ascending=False)


# In[56]:


sns.barplot(x=feature_imp,y=x.columns)
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Visulizing Important Features")
plt.show()


# In[57]:


from sklearn.feature_selection import SelectFromModel


# In[58]:


feat_sel=SelectFromModel(RF)


# In[59]:


feat_sel.fit(x_train,y_train)


# In[60]:


x_imp_train=feat_sel.transform(x_train)
x_imp_test=feat_sel.transform(x_test)


# In[61]:


RF_new=RandomForestClassifier(criterion="entropy")


# In[62]:


RF_new.fit(x_imp_train,y_train)


# In[63]:


y_pred_new_RF=RF_new.predict(x_imp_test)
y_pred_new_RF


# In[64]:


confusion_matrix(y_test,y_pred_new_RF)


# In[65]:


print("Accuarcy Score for new random forest is ",accuracy_score(y_test,y_pred_new_RF))
print("Recall Score for  new random forest is :",recall_score(y_test,y_pred_new_RF))
print("Precision Score for new random forest is :",precision_score(y_test,y_pred_new_RF))


# ### Our Accuarcy score remains same but recall score decreases ( by around 5 %) and precision increases (by around 3 %) . 

# # Supoort Vector

# In[66]:


from sklearn.svm import SVC    # Support Vector Classifier


# In[67]:


svc=SVC(kernel="linear",random_state=0)


# In[68]:


svc.fit(x_train,y_train)


# In[69]:


y_pred_SVC=svc.predict(x_test)
y_pred_SVC


# In[70]:


cm=confusion_matrix(y_test,y_pred_SVC)
sns.heatmap(cm,annot=True,fmt="d")
plt.title("Comfusion Matrix For SVC")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.show()


# In[71]:


print("Accuarcy Score For Supoort Vector Classifier is :",accuracy_score(y_test,y_pred_SVC))
print("Recall Score For Supoort Vector Classifier is :",recall_score(y_test,y_pred_SVC))
print("Precision Score For Supoort Vector Classifier is :",precision_score(y_test,y_pred_SVC))


# ### Here we got slightly higher accuracy than Logistic Regression model ( 82.43 %).
# ### Also recall is is also higher.
# ### So we can say that SVC may be good fit .

# # Naive Bayes Classifier 

# In[72]:


from sklearn.naive_bayes import GaussianNB


# In[73]:


gnb=GaussianNB()


# In[74]:


gnb.fit(x_train,y_train)


# In[75]:


y_pred_NB=gnb.predict(x_test)
y_pred_NB


# In[76]:


cm=confusion_matrix(y_test,y_pred_NB)
sns.heatmap(cm,annot=True,fmt="d")
plt.title("Comfusion Matrix for Naive Bayes")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.show()


# In[77]:


print("Accuarcy Score For Naive Bayes Classifier is :",accuracy_score(y_test,y_pred_NB))
print("Recall Score For Naive Bayes Classifier  is :",recall_score(y_test,y_pred_NB))
print("Precision Score For Naive Bayes Classifier  is :",precision_score(y_test,y_pred_NB))


# ### Here also we got lower accuracy as compared to SVC.
# ### So finally we can conclude that  SVC can be use for futher prediction.

# ### So we use SVC model for prediction.

# ### Checking in random inputs. 

# In[80]:


Gender=int(input("Enter Gender of applicant :"))    # for Male=1 and female=0
Married=int(input("Enter marrital Status of applicant :"))  # Yes=1 and no=0
Dependents=int(input("Enter no of Dependents of applicant :")) # 0,1,2,3
Education=int(input("Enter Education Status of applicant :"))  # Graduate=1 and Not Graduate=0
Self_Employed=int(input("Enter wethere applicant is Self_Employed or not :")) # Yes=1 and no=0
ApplicantIncome=float(input("Enter income of applicant :" ))  
CoapplicantIncome=float(input("Enter Income of coapplicant :")) 
LoanAmount=float(input("Enter how much loan applicant want :"))
Loan_Amount_Term=int(input("Enter in how many months applicant wishes to repay loan :"))  
Credit_History=int(input("Enter Credit_History of applicant :"))  # 0 or 1
Property_Area=int(input("Enter Property_Area of applicant :" ))    # Urban=0 , Semiurban=1 , rural=2

prediction=gnb.predict([[Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area]])

print(prediction)


# In[81]:


if prediction==1:
    print("Applicant loan will be approved")
else:
    print("Applicant loan will be rejected")


# ##           THANK YOU ! 
