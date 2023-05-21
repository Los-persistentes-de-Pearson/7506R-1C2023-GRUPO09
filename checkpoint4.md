---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.5
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Importamos

```python
try:
  import google.colab
  IN_COLAB = True
except:
  IN_COLAB = False

import pandas as pd 
import numpy as np
import sklearn as sk
import seaborn as sns
import pydotplus
from six import StringIO
from IPython.display import Image  
from matplotlib import pyplot as plt
#TODO creo q no va:
from dict_paises import COUNTRY_ALPHA3_TO_COUNTRY_ALPHA2, COUNTRY_ALPHA2_TO_CONTINENT
from joblib import dump, load
from os.path import exists

from sklearn.model_selection import StratifiedKFold, KFold,RandomizedSearchCV, train_test_split, cross_validate
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import confusion_matrix, classification_report , f1_score, make_scorer, precision_score, recall_score, accuracy_score,f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn import tree

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score



#Si estamos  en colab tenemos que instalar la libreria "dtreeviz" aparte. 
if IN_COLAB == True:
    !pip install 'dtreeviz'
import dtreeviz as dtreeviz

#Para eliminar los warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

from advertencia import ADVERTENCIA #Borrar cuando entreguemos

```

Constantes

```python
# Constantes
JOBS=-2
SEED=9
```

importo el dataframe original para tener los ids

```python
hotelsdf_pruebasOriginal = pd.read_csv("./hotels_test.csv")
```

Importamos los dataframes ya filtrados que generamos en el checkpoint 1. De esta forma evitamos tener todo el codigo de homologacion de dataframes de testeo y train aqui de nuevo

```python
hotelsdf_train_filtrado = pd.read_csv("hotels_filtrado_train.csv")
hotelsdf_testeo_filtrado = pd.read_csv("hotels_filtrado_test.csv")
```

```python
set_test = set(hotelsdf_testeo_filtrado.columns)
set_modelo = set(hotelsdf_train_filtrado.columns)

missing = list(sorted(set_test - set_modelo))
added = list(sorted(set_modelo - set_test))

print('Faltan en arbol:', missing)
print('Sobran en arbol:', added)
```

Guardo datos antes de separacion train test


# Generacion de datos para el entrenamiento de los modelos

Se genera un dataset con los datos necesarios para predecir la cancelacion y creamos un dataset conteniendo el target, para luego, generar conjuntos de test y train

```python
hotelsdf_modelo_x=hotelsdf_train_filtrado.drop(['is_canceled'], axis='columns', inplace=False)

hotelsdf_modelo_y = hotelsdf_train_filtrado['is_canceled'].copy()

x_train, x_test, y_train, y_test = train_test_split(hotelsdf_modelo_x,
                                                    hotelsdf_modelo_y, 
                                                    test_size=0.3,  #proporcion 70/30
                                                    random_state=SEED) #Semilla 9, como el Equipo !!
```

```python
x_test
```

Imports para armar la red

```python
import tensorflow as tf
from tensorflow import keras
from keras.utils.vis_utils import plot_model
# import visualkeras

np.random.seed(9)
tf.random.set_seed(9) 

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
```

```python
# from sklearn.datasets import load_iris
# from sklearn.preprocessing import OneHotEncoder, StandardScaler

# iris = load_iris()

# x = iris['data']
# y = iris['target']
# names = iris['target_names']
# feature_names = iris['feature_names']

# # Split the data set into training and testing
# x_train, x_test, y_train, y_test = train_test_split(x, y, 
#                                                     test_size=0.5,
#                                                     stratify = y,
#                                                     random_state=22)

# # One hot encoding
# enc = OneHotEncoder()

y_train_tensor = tf.convert_to_tensor(y_train)
# y_train_encoder = y_train.to_list()
y_test_tensor = tf.convert_to_tensor(y_test)
# y_test_encoder = y_test.to_list()


# # scaler = StandardScaler()
x_train_tensor = tf.convert_to_tensor(x_train)
# x_train_scaled = x_train.to_list()
x_test_tensor = tf.convert_to_tensor(x_test)
# x_test_scaled = x_test.to_list()
```

```python
# calcula la cantidad de clases
#cant_clases=len(np.unique(y))
cant_clases = 1

d_in=len(x_train.columns)

modelo_hotels_1 = keras.Sequential([
    # input_shape solo en la primer capa
    keras.layers.Dense(8,input_shape=(d_in,),activation ='relu'),
    keras.layers.Dense(16,input_shape=(d_in,),activation ='relu'),
    keras.layers.Dense(32,input_shape=(d_in,),activation ='relu'),
    keras.layers.Dense(64,input_shape=(d_in,),activation ='relu'),
    keras.layers.Dense(cant_clases, activation='sigmoid'),])


modelo_hotels_1.summary()
```

```python
modelo_hotels_1.compile(
  optimizer=keras.optimizers.SGD(learning_rate=0.1), 
  loss='binary_crossentropy', 
  # metricas para ir calculando en cada iteracion o batch 
  metrics=['AUC'], 
)

cant_epochs=10

historia_modelo_iris_1=modelo_hotels_1.fit(x_train,y_train,epochs=cant_epochs,batch_size=16,verbose=False)
```

```python
y_pred = modelo_hotels_1.predict(x_test)
y_predic_cat_ej1 = np.where(y_pred>0.7,1,0)

ds_validacion=pd.DataFrame(y_predic_cat_ej1,y_test).reset_index()
ds_validacion.columns=['y_pred','y_real']

tabla=pd.crosstab(ds_validacion.y_pred, ds_validacion.y_real)
grf=sns.heatmap(tabla,annot=True, cmap = 'Blues')
plt.show()
```

```python
y_pred
```

Predecimos sobre el de testeo

```python
# y_pred_testeo = modelo_hotels_1.predict(hotelsdf_testeo_filtrado)
```

```python
# y_pred_testeo
```

```python
# df_submission = pd.DataFrame({'id': hotelsdf_pruebasOriginal['id'], 'is_canceled': y_pred_testeo.tolist()})
# df_submission.to_csv('submissions/red_1.csv', index=False)
```
