---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.5
  kernelspec:
    display_name: Python 3 (ipykernel)
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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

import tensorflow as tf
from tensorflow import keras
from keras.utils.vis_utils import plot_model

from keras.wrappers.scikit_learn import KerasClassifier #Libreria para usarg Grid Search con kerss
```

Constantes

```python
# Constantes
JOBS=-2
SEED=9

np.random.seed(SEED)
tf.random.set_seed(SEED) 
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

Buscamos los valores que no fueron generados por el one hot encoder

```python
x_train.columns
```

```python
valoresNoBinarios = ['lead_time', 'arrival_year', 'arrival_week_number', 'arrival_month_day',
       'weekend_nights_num', 'week_nights_num', 'adult_num', 'children_num',
       'babies_num', 'previous_cancellations_num',
       'booking_changes_num', 'days_in_waiting_list',
       'average_daily_rate', 'required_car_parking_spaces_num',
       'special_requests_num', 'dias_totales']
```

Imports para armar la red


Tenemos que escalar todos los valores de nuestro data set (excepto los valores producidos por el one hot encoding

```python
sScaler = StandardScaler()
sScaler.fit(pd.DataFrame(x_train[valoresNoBinarios]))
```

```python
x_train_transform_1=sScaler.transform(pd.DataFrame(x_train[valoresNoBinarios]))
x_test_transform_1=sScaler.transform(pd.DataFrame(x_test[valoresNoBinarios]))
```

```python
x_train_transform_1
```

```python
#Creamos un nuevo dataframe con los valores escalados
x_train_escalado = x_train.copy()
x_test_escalado = x_test.copy()
```

```python
#Le asignamos los nuevos valores escalados y mantenemos los valores del one hot encoding
for i in range(len(valoresNoBinarios)):
    x_train_escalado[valoresNoBinarios[i]]=x_train_transform_1[:,i]
    x_test_escalado[valoresNoBinarios[i]]=x_test_transform_1[:,i]
```

```python
x_train_escalado
```

```python
x_test_escalado
```

```python
cant_clases = 1


#d_in=len(y_train)
d_in=len(x_train_escalado.columns)

modelo_hotels_1 = keras.Sequential([
    keras.layers.Dense(8,input_shape=(d_in,),activation ='relu'),
    keras.layers.Dense(cant_clases, activation='sigmoid'),
])


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

historia_modelo_hotel_1=modelo_hotels_1.fit(x_train_escalado,y_train,epochs=cant_epochs,batch_size=16,verbose=False)
```

```python
y_pred = modelo_hotels_1.predict(x_test_escalado)
y_predic_cat_ej1 = np.where(y_pred>0.7,1,0)

ds_validacion=pd.DataFrame(y_predic_cat_ej1,y_test).reset_index()
ds_validacion.columns=['y_pred','y_real']

tabla=pd.crosstab(ds_validacion.y_pred, ds_validacion.y_real)
grf=sns.heatmap(tabla,annot=True, cmap = 'Blues', fmt='g')
plt.show()
```

```python
if not exists('submissions/red_neuronal_basica.csv'):
    y_pred_testeo = modelo_hotels_1.predict(hotelsdf_testeo_filtrado)
    y_pred_testeo_cat = np.where(y_pred_testeo>0.70,1,0)
    df_resultados_pred = pd.DataFrame.from_records(y_pred_testeo_cat,columns = ["resultado"])
    df_submission = pd.DataFrame({'id': hotelsdf_pruebasOriginal['id'], 'is_canceled': df_resultados_pred["resultado"]})
    df_submission.to_csv('submissions/red_neuronal_basica.csv', index=False)
```

# Validacion cruzada


En esta etapa vamos a realizar unas series de validaciones cruzadas en la busqueda de encontrar el mejor resultado


Para usar la libreria Keras Classifier necesitamos crear una funcion que cree una modelo.

```python
loss='binary_crossentropy'
metrics=['AUC']
optimizer="adam"

def creador_modelo(learning_rate = 0.1, 
                   activation = 'sigmoid', 
                   output = 2, 
                  hidden_layers = 2
                  ):
    model = keras.Sequential()
    model.add(keras.layers.Dense(5, activation=activation, input_shape=(d_in,)))
    
    for i in range(hidden_layers):
        model.add(keras.layers.Dense(output, activation=activation))

    model.add(keras.layers.Dense(1, activation=activation))
    
    model.compile(
      optimizer=optimizer,
      loss=loss, 
      metrics=metrics, 
    )
    return model
```

Vamos a empezar con una baja cantidad de epochs y batch_size

```python
model = KerasClassifier(build_fn=creador_modelo, 
                        verbose=1)
```

```python
param_grid = { 
                  "hidden_layers" : [1, 5, 10, 15, 20], 
                    "output" : [1, 2, 4, 8, 32, 64], 
                    "batch_size" : [5, 10, 20],
                    "epochs" : [50, 100, 150],
                   "activation": ["sigmoid", "relu", "softmax", "softplus", "elu", ]
             } 
```

```python
rs = RandomizedSearchCV(estimator=model, param_distributions=param_grid,n_jobs=JOBS, cv=3,n_iter=10)
```

```python
if exists('modelos/rs_fit.joblib') == False:
    rs_fit = rs.fit(X = x_train_escalado, y = y_train)
    dump(rs_fit, 'modelos/rs_fit.joblib')
else:
    rs_fit = load('modelos/rs_fit.joblib')
```

```python
rs_fit.best_params_
```

```python
modelo_rs = keras.Sequential()
modelo_rs.add(keras.layers.Dense(5, activation=rs_fit.best_params_["activation"], input_shape=(d_in,)))
    
for i in range(rs_fit.best_params_["hidden_layers"]):
        # Add one hidden layer
    modelo_rs.add(keras.layers.Dense(rs_fit.best_params_["output"], activation=rs_fit.best_params_["activation"]))

modelo_rs.add(keras.layers.Dense(1, activation="sigmoid"))

    
modelo_rs.compile(
#     optimizer=keras.optimizers.SGD(learning_rate=rs_fit.best_params_["learning_rate"]), 
    optimizer=optimizer,
    loss=loss, 
    metrics=metrics, 
    )
```

```python
# cant_epochs=10
cant_epochs=rs_fit.best_params_["epochs"]
batch_size=rs_fit.best_params_["batch_size"]

historia_modelo_hotel_2=modelo_rs.fit(x_train_escalado,y_train,
                                      epochs=cant_epochs,
                                      batch_size=batch_size,verbose=False)

```

```python
y_pred = modelo_rs.predict(x_test_escalado)
y_predic_cat_ej1 = np.where(y_pred>0.5,1,0)

ds_validacion=pd.DataFrame(y_predic_cat_ej1,y_test).reset_index()
ds_validacion.columns=['y_pred','y_real']

tabla=pd.crosstab(ds_validacion.y_pred, ds_validacion.y_real)
grf=sns.heatmap(tabla,annot=True, cmap = 'Blues', fmt='g')
#plt.ticklabel_format(style='plain', axis='both')
plt.show()
```

```python

```
