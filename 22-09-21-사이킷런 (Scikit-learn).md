#### ML_01_수업자료.html


```python
# ML 관련 모듈 임포트 

import sklearn
```

# 사이킷런 (Scikit-learn)


```python
sklearn.__version__
```




    '1.0.2'



# XOR

- 입력 피처 2개 
- 서로 같은 값이 입력되면 0 이 반환
- 서로 다른 값이 입력되면 1 이 반환

>- 1) 학습 데이터 생성
>- 2) 모델 생성
>- 3) fit() →  모델에 학습데이터를 입력하여 학습 시킴 
>- 4) 예측 predict() → 학습시킨 모델에 테스트 데이터 입력
>- 5) score() → 평가 

![image.png](attachment:image.png)


```python
# 경고 메세지 숨기기 
import warnings
warnings.filterwarnings(action='ignore')

# 라이브러리 임포트 
import numpy as np
import pandas as pd
```


```python
# 모델 = 학습기 = 분류기 임포트 는????

# knn
from sklearn.neighbors import KNeighborsClassifier
# decisionTree
from sklearn.tree import DecisionTreeClassifier
# randomForest
from sklearn.ensemble import RandomForestClassifier
# svm : support vector machine
from sklearn.svm import SVC
```


```python
# 1) 학습 데이타 생성 - 리스트 생성 

xor_data =  [
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
]

# 입력 데이타와 결과 데이타 분리 
# 입력 데이타 X
# 정답지 = 레이블 = 타겟  y 

# X : 0, 1번째 열
X = [ [0, 0], [0, 1], [1,0], [1, 1] ]
# y : 2번째 열
y = [0, 1, 1, 0]
```


```python
# 2) 모델 생성
# 모델명 : model_dt = model_분류기() = 모델생성함수()

# decisionTree 
model_dt = DecisionTreeClassifier()
print(model_dt)

# 생성된 모델의 옵션 정보 확인 : 모델명.get_params() 모델의 스팩
model_dt.get_params()
```

    DecisionTreeClassifier()
    




    {'ccp_alpha': 0.0,
     'class_weight': None,
     'criterion': 'gini',
     'max_depth': None,
     'max_features': None,
     'max_leaf_nodes': None,
     'min_impurity_decrease': 0.0,
     'min_samples_leaf': 1,
     'min_samples_split': 2,
     'min_weight_fraction_leaf': 0.0,
     'random_state': None,
     'splitter': 'best'}




```python
# 3) fit() → 모델에 학습데이터를 입력하여 학습 시킴 
# 모델명.fit(입력데이터, 결과데이터)

model_dt.fit( X , y)
```




    DecisionTreeClassifier()




```python
# 4) 예측 predict() → 학습시킨 모델에 테스트 데이터 입력
# 학습시킨 모델명.predict( 입력데이터 )

model_dt.predict(X)
```




    array([0, 1, 1, 0])




```python
# 개별 확인
# model_dt.predict([1, 1]), model_dt.predict([0, 1]) # ValueError: Expected 2D array, got 1D array instead: 차원에러

# 개별 확인 - 2차원으로
model_dt.predict([[1, 1]]), model_dt.predict([[0, 1]])
```




    (array([0]), array([1]))




```python
# 학습되지 않은 미지의 데이터의 경우 → 예측 불가

model_dt.predict([[2, 3]])
```




    array([0])




```python
# 5) score() → 평가
# 몇개 맞았는가?
# 학습시킨 모델명.score( 테스트 데이터, 테스트 데이터에 대한 y 정답 )

model_dt.score( X, model_dt.predict(X))
```




    1.0




```python
X
```




    [[0, 0], [0, 1], [1, 0], [1, 1]]




```python
model_dt.predict(X)
```




    array([0, 1, 1, 0])




```python
# 정답률, 에러률 임포트

from sklearn.metrics import accuracy_score, mean_squared_error
```


```python
# accuracy_score( y_true, y_pred )
# mean_squared_error( y_true, y_pred )  
# shift + tab 키
    
print('정답률 → ' , accuracy_score( y , model_dt.predict(X)))
print('에러율 → ' , mean_squared_error(y, model_dt.predict(X)))                           
```

    정답률 →  1.0
    에러율 →  0.0
    

### 데이터가 데이터프레임인 경우


```python
# 1) 학습 데이터 생성 - 데이터프레임 생성

xor_data =  [
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
]

df = pd.DataFrame(xor_data)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 1) 입력 데이터와 결과 데이터 분리

X = df[[0, 1]]
y = df[2]
```


```python
X  # 2개이상 데이터 프레임
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
y # 1개는 넘파이
```




    0    0
    1    1
    2    1
    3    0
    Name: 2, dtype: int64




```python
# 2) 모델 생성
# 학습기 : SVM
# 모델명 : model_dt = model_분류기() = 모델생성함수()

model_svc = SVC()
print(model_svc)

# 생성된 모델의 옵션 정보 확인 : 모델명.get_params() 모델의 스팩
model_svc.get_params()
```

    SVC()
    




    {'C': 1.0,
     'break_ties': False,
     'cache_size': 200,
     'class_weight': None,
     'coef0': 0.0,
     'decision_function_shape': 'ovr',
     'degree': 3,
     'gamma': 'scale',
     'kernel': 'rbf',
     'max_iter': -1,
     'probability': False,
     'random_state': None,
     'shrinking': True,
     'tol': 0.001,
     'verbose': False}




```python
# 3) fit() → 모델에 학습데이터를 입력하여 학습 시킴 
# 학습된 모델명.fit(입력데이터, 결과데이터)

model_svc.fit( X , y)
```




    SVC()




```python
# 4) 예측 predict() → 학습시킨 모델에 테스트 데이터 입력
# 학습된 모델명.predict( 입력데이터 )

model_svc.predict(X)
```




    array([0, 1, 1, 0], dtype=int64)




```python
# 5) score() → 평가
# 몇개 맞았는가?
# 학습된 모델명.score( 테스트 데이터, 테스트 데이터에 대한 y 정답 )

model_svc.score( X, model_svc.predict(X))
print('정답률 →' , accuracy_score(y, model_svc.predict(X)))
print('에러율 →' , mean_squared_error(y, model_svc.predict(X)))
```

    정답률 → 1.0
    에러율 → 0.0
    
