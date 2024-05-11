## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
  ```
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```
![image](https://github.com/SanjithaBolisetti/EXNO-3-DS/assets/119393633/7c5c7b41-ca06-4c35-a325-c55746908950)

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/SanjithaBolisetti/EXNO-3-DS/assets/119393633/84505d9e-25d8-41fb-ac61-d919371e8447)

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/SanjithaBolisetti/EXNO-3-DS/assets/119393633/16dec4a2-5429-4709-8a4b-830c42d6cf88)

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/SanjithaBolisetti/EXNO-3-DS/assets/119393633/384c48bf-a191-47a2-9e50-d28fe33a34cc)

```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
```
![image](https://github.com/SanjithaBolisetti/EXNO-3-DS/assets/119393633/d74a1536-ddbb-4a83-8a3c-a1e88d3abf9b)

```
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/SanjithaBolisetti/EXNO-3-DS/assets/119393633/6684ca0d-c3b9-4f65-ad37-5bed3f027d09)

```
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/SanjithaBolisetti/EXNO-3-DS/assets/119393633/5776741a-8ec0-4b90-b26e-ad2332d6d5da)

```
pip install --upgrade category_encoders
```
![image](https://github.com/SanjithaBolisetti/EXNO-3-DS/assets/119393633/56b25041-b4db-4f56-8613-ef36001cdd20)

```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```
![image](https://github.com/SanjithaBolisetti/EXNO-3-DS/assets/119393633/377aa32b-8193-43d8-b619-dd17463dc37f)

```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
![image](https://github.com/SanjithaBolisetti/EXNO-3-DS/assets/119393633/50e87ab7-5452-442c-84c6-8c150bcbd70c)

```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
![image](https://github.com/SanjithaBolisetti/EXNO-3-DS/assets/119393633/bb5e9e61-855c-4b0a-91e6-17ce6c9f0a1d)

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
![image](https://github.com/SanjithaBolisetti/EXNO-3-DS/assets/119393633/288b5a55-3464-42e6-9e37-72577c1ba921)

```
df.skew()
```
![image](https://github.com/SanjithaBolisetti/EXNO-3-DS/assets/119393633/ba6e4ee4-2336-40ff-938a-be512d1ce196)

```
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/SanjithaBolisetti/EXNO-3-DS/assets/119393633/e0ef4334-2437-4b76-843c-b65ce00b7f7f)

```
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/SanjithaBolisetti/EXNO-3-DS/assets/119393633/a6741de7-111c-4932-8076-ff78f47ea3b6)

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/SanjithaBolisetti/EXNO-3-DS/assets/119393633/48218bb8-c4ce-4259-8404-a5a22a9713b7)

```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
```
![image](https://github.com/SanjithaBolisetti/EXNO-3-DS/assets/119393633/fd914687-20ca-40c1-8eec-28e542dbdd1e)

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/SanjithaBolisetti/EXNO-3-DS/assets/119393633/5971fa2f-f2b5-4cbb-9358-811c4eab2f3c)

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew_1"]),line='45')
plt.show()
```

![image](https://github.com/SanjithaBolisetti/EXNO-3-DS/assets/119393633/3642a54c-8b42-4647-abd1-9c71e75b201f)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/SanjithaBolisetti/EXNO-3-DS/assets/119393633/1204af19-ecc7-4dec-a95e-8fd6412b1b32)

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/SanjithaBolisetti/EXNO-3-DS/assets/119393633/fd9fccd0-cfa3-414b-9d31-53db84413de5)

```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```

![image](https://github.com/SanjithaBolisetti/EXNO-3-DS/assets/119393633/472ee5a5-4590-4867-aaa9-823dc5ad5060)

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```

![image](https://github.com/SanjithaBolisetti/EXNO-3-DS/assets/119393633/0187857c-5a38-43e1-ad73-3b0281102dfc)

# RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.

       
