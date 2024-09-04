<H3>ENTER YOUR NAME: PAWAN CHARAN</H3>
<H3>ENTER YOUR REGISTER NO: 212223220074</H3>
<H3>EX. NO.1</H3>
<H3>DATE: 04/09/2024</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
#importing libraries
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#Reading the dataset
df=pd.read_csv("Churn_Modelling.csv", index_col="RowNumber")
df

#Dropping the unwanted Columns
df.drop(['CustomerId'],axis=1,inplace=True)
df.drop(['Surname'],axis=1,inplace=True)
df.drop('Age',axis=1,inplace=True)
df.drop('Geography',axis=1,inplace=True)
df.drop('Gender',axis=1,inplace=True)
df

#Checking for null values
df.isnull().sum()

#Checking for duplicate values
df.duplicated()

#Describing the dataset
df.describe()

#Scaling the dataset
scaler=StandardScaler()
df1=pd.DataFrame(scaler.fit_transform(df))
df1

#Allocating X and Y attributes
x=df1.iloc[:,:-1].values
x
y=df1.iloc[:,-1].values
y

#Splitting the data into training and testing dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(x_train)
print(len(x_train))
print(x_test)
print(len(x_test))
```



## OUTPUT:
## Dataset:
![NN ex 01 img 01](https://github.com/user-attachments/assets/fd94d10e-765f-43eb-a63c-26ff53b84200)


## Dropping The Unwanted Dataset:

![NN ex 01 img 02](https://github.com/user-attachments/assets/8bcfcc11-c590-4df4-82f3-afe08d90ad8e)

## Checking Null Values:

![NN ex 01 img 03](https://github.com/user-attachments/assets/c40f34a8-585c-44be-a759-9b0a949ca549)

## Checking For Duplication:

![NN ex 01 img 04](https://github.com/user-attachments/assets/290d709b-d4cc-450b-aeac-f410e8b87084)

## Describing The DataSet:
![NN ex 01 img 05](https://github.com/user-attachments/assets/b6add3e5-b0d0-47ba-a999-5e2793e98b12)


## Scaling The DataSet:

![NN ex 01 img 06](https://github.com/user-attachments/assets/8777f2b8-1246-40e9-af92-13ef561c17eb)

## X Features:

![NN ex 01 img 07](https://github.com/user-attachments/assets/a4e7375d-4c4b-44ed-a735-6708033c63fb)

## Y Features:

![NN ex 01 img 08](https://github.com/user-attachments/assets/51da4f85-5ab5-4883-9c7d-ffbd2a9558f0)

## Splitting The Training And Testing DataSet:
![NN ex 01 img 09](https://github.com/user-attachments/assets/d813cffd-9173-4c7d-8484-e0552abb7537)


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


