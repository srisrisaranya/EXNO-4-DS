# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
## FEATURE SCALING:
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")
df.head()
```

![image](https://github.com/user-attachments/assets/6838b633-3d30-4558-a715-ed00e7283f3f)

```
df.dropna()
```

![image](https://github.com/user-attachments/assets/f9a610cd-afb4-4f4a-b29f-7bab813932dc)

```
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
```

![image](https://github.com/user-attachments/assets/7bdea6a9-5997-425b-b75f-cdf0001d88ab)

```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```

![image](https://github.com/user-attachments/assets/e478d692-ff9e-48f5-9797-61faaf89758a)

```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```

![image](https://github.com/user-attachments/assets/d68d992b-65f5-4133-adab-ec3eb906b25d)

```
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```

![image](https://github.com/user-attachments/assets/d0a8f2fa-db2e-4820-897f-bb1217c45a63)

```
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```

![image](https://github.com/user-attachments/assets/fd3a2d89-0b68-48ef-9c5f-3e41dff7a8ba)

```
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head()
```

![image](https://github.com/user-attachments/assets/366961e4-49a6-44eb-b449-4be50a845000)

## FEATURE SELECTION

```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
data = pd.read_csv('/content/income(1) (1).csv',na_values=[" ?"])
data
```

![image](https://github.com/user-attachments/assets/a52d6200-2c4f-48b5-b401-38337a02a5e1)

```
data.isnull().sum()
```

![image](https://github.com/user-attachments/assets/1f38b14e-dea1-4e3e-a43c-f68133fa7f1c)

```
missing = data[data.isnull().any(axis=1)]
missing
```

![image](https://github.com/user-attachments/assets/4311092c-79d3-4bbb-82c0-d71dc81924be)

```
data2 = data.dropna(axis=0)
data2
```

![image](https://github.com/user-attachments/assets/6ebb1e49-c1ac-41c7-afe6-28dcb2825b67)

```
sal=data['SalStat']
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```

![image](https://github.com/user-attachments/assets/1e3ccf5e-9b30-4337-8586-6a046a38b4c8)

```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```

![image](https://github.com/user-attachments/assets/8213059c-0b7a-47c2-a6a0-b7aacf67a6db)

```
data2
```

![image](https://github.com/user-attachments/assets/7130573b-ab90-44a3-8a17-287ce78bcbce)

```
new_data=pd.get_dummies(data2,drop_first=True)
new_data
```

![image](https://github.com/user-attachments/assets/c5ee05de-4e2b-4f3d-9d9c-69c2368488a0)

```
y=new_data['SalStat'].values
print(y)
```

![image](https://github.com/user-attachments/assets/4bee8b68-0b0c-4099-806c-311bf69d2fc8)

```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
x = new_data.drop('SalStat', axis=1)  
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=0)
KNN_classifier = KNeighborsClassifier(n_neighbors=5)
KNN_classifier.fit(train_x, train_y)
```

![image](https://github.com/user-attachments/assets/f2609bbe-be7c-4a20-b659-a696393107f4)

```
prediction=KNN_classifier.predict(test_x)
confusionmatrix=confusion_matrix(test_y,prediction)
print(confusionmatrix)
```

![image](https://github.com/user-attachments/assets/d8e2cdc2-25e2-4f7a-8fcc-d8f5d82276e5)

```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```

![image](https://github.com/user-attachments/assets/4ebe3682-0471-4ac1-8d2e-29df0a4e461a)

```
tips.time.unique()
```

![image](https://github.com/user-attachments/assets/c467ef6d-64ae-468f-a60b-b7930c191ea6)

```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```

![image](https://github.com/user-attachments/assets/b650fa60-46f5-43c2-b5aa-45ba05f7d670)

```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```

![image](https://github.com/user-attachments/assets/d33b1c55-5a43-417a-a9a9-e2556e5bc4fe)

# RESULT:
Thus,the Feature selection and Feature scaling has been used on the given dataset is executed successfully.
