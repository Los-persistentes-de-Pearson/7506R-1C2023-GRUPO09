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

from advertencia import ADVERTENCIA #Borrar cuando entreguemos

import tensorflow as tf
from tensorflow import keras
from keras.utils.vis_utils import plot_model
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

```python
cuantitativas = [
"adult_num",
"arrival_month_day",
"arrival_week_number",
"arrival_year",
"average_daily_rate",
"babies_num",
"booking_changes_num",
"children_num",
"days_in_waiting_list",
"lead_time",
"previous_cancellations_num",
"required_car_parking_spaces_num",
"special_requests_num",
"weekend_nights_num",
"week_nights_num",
]
```

```python
len(cuantitativas)
```

Imports para armar la red

```python
# from sklearn.datasets import load_iris

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

#y_train_tensor = tf.convert_to_tensor(y_train)
# y_train_encoder = y_train.to_list()
#y_test_tensor = tf.convert_to_tensor(y_test)
# y_test_encoder = y_test.to_list()



#x_train_tensor = tf.convert_to_tensor(x_train)
# x_train_scaled = x_train.to_list()
#x_test_tensor = tf.convert_to_tensor(x_test)
# x_test_scaled = x_test.to_list()
```

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
#     x_train_escalado[valoresNoBinarios[i]]=x_train_transform_1[:,0]
#     x_test_escalado[valoresNoBinarios[i]]=x_test_transform_1[:,0]
```

```python
x_train_escalado
```

```python
x_test_escalado
```

```python
len(np.unique(y_train))
```

```python
# calcula la cantidad de clases
#cant_clases=len(np.unique(y))
#cant_clases = len(np.unique(y_train))
cant_clases = 1


#d_in=len(y_train)
d_in=len(x_train_escalado.columns)

modelo_hotels_1 = keras.Sequential([
    # input_shape solo en la primer capa
    keras.layers.Dense(2,input_shape=(d_in,),activation ='relu'),
#     keras.layers.Dense(8,input_shape=(d_in,),activation ='relu'),
#     keras.layers.Dense(16,input_shape=(d_in,),activation ='relu'),
#     keras.layers.Dense(32,input_shape=(d_in,),activation ='relu'),
#     keras.layers.Dense(64,input_shape=(d_in,),activation ='relu'),
    keras.layers.Dense(cant_clases, activation='sigmoid'),
#     keras.layers.Dense(1,input_shape=(d_in,)),
#     keras.layers.Dense(1, activation='sigmoid')
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
#plt.ticklabel_format(style='plain', axis='both')
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

```python
input()
```

## Validacion cruzada

```python
df_submission = pd.DataFrame({'id': hotelsdf_pruebasOriginal['id'], 'is_canceled': resultados})
df_submission.to_csv('submissions/red_1.csv', index=False)
```

```python
y_pred
```

Predecimos sobre el de testeo

```python
y_pred_testeo = modelo_hotels_1.predict(hotelsdf_testeo_filtrado)
```

```python
y_predic_cat_ej1 = np.where(y_pred_testeo>0.3,1,0)
```

```python
resultados = pd.DataFrame(y_predic_cat_ej1)[0]
```

```python

```

# Amongus

```python colab={"base_uri": "https://localhost:8080/"} id="60da0c2b" outputId="c3379a9e-f818-40ee-b033-2a56c767619b" vscode={"languageId": "python"}
cant_clases = 1

d_in=len(x_train.columns)

modelo_hotels_1 = keras.Sequential([
    keras.layers.Dense(8,input_shape=(d_in,),activation ='relu'),
    keras.layers.Dense(16,input_shape=(d_in,),activation ='relu'),
    keras.layers.Dense(32,input_shape=(d_in,),activation ='relu'),
    keras.layers.Dense(cant_clases, activation='sigmoid'),])


modelo_hotels_1.summary()
```

```python id="e0599e98" vscode={"languageId": "python"}
modelo_hotels_1.compile(
  optimizer=keras.optimizers.SGD(learning_rate=0.01), 
  loss='binary_crossentropy', 
  # metricas para ir calculando en cada iteracion o batch 
  metrics=['AUC'], 
)

cant_epochs=10

historia_modelo_hotels_1=modelo_hotels_1.fit(x_train,y_train,epochs=cant_epochs,batch_size=16,verbose=False)
```

```python vscode={"languageId": "python"}
epochs = range(cant_epochs)

plt.plot(epochs, historia_modelo_hotels_1.history['auc'], color='orange', label='AUC')
plt.xlabel("epochs")
plt.ylabel("AUC")
plt.legend()
```

Mostramos los resultados de este primer modelo

```python vscode={"languageId": "python"}
y_pred_modelo_1 = modelo_hotels_1.predict(x_test)
y_predic_cat_modelo_1 = np.where(y_pred_modelo_1>0.50,1,0)

ds_validacion=pd.DataFrame(y_predic_cat_modelo_1,y_test).reset_index()
ds_validacion.columns=['y_pred','y_real']

tabla=pd.crosstab(ds_validacion.y_pred, ds_validacion.y_real)
grf=sns.heatmap(tabla,annot=True, cmap = 'Blues', fmt='g')

#plt.ticklabel_format(style='plain', axis='both')
plt.show()
```

Si bien los resultados son relativamente buenos y no se ve que el modelo este muy sesgado, vemos q a partir de ???? NO SE QUE JUSTIFICAR

```python vscode={"languageId": "python"}
cant_clases = 1

d_in=len(x_train.columns)

modelo_hotels_2= keras.Sequential([
    keras.layers.Dense(8,input_shape=(d_in,),activation ='relu'),
    keras.layers.Dense(16,input_shape=(d_in,),activation ='relu'),
    keras.layers.Dense(32,input_shape=(d_in,),activation ='relu'),
    keras.layers.Dense(cant_clases, activation='sigmoid'),])

modelo_hotels_2.compile(
  optimizer=keras.optimizers.SGD(learning_rate=0.01), 
  loss='binary_crossentropy', 
  # metricas para ir calculando en cada iteracion o batch 
  metrics=['AUC'], 
)

cant_epochs=30

historia_modelo_hotels_2=modelo_hotels_1.fit(x_train,y_train,epochs=cant_epochs,batch_size=16,verbose=False)
```

```python vscode={"languageId": "python"}
epochs = range(cant_epochs)

plt.plot(epochs, historia_modelo_hotels_2.history['auc'], color='orange', label='AUC')
plt.xlabel("epochs")
plt.ylabel("auc")
plt.legend()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="4ab27167" outputId="c6e52f3c-6c96-47c1-ac4b-d14b378ff8d9" vscode={"languageId": "python"}
y_pred_modelo_2 = modelo_hotels_2.predict(x_test)
```

<!-- #region id="7d618a5e" -->
Predecimos sobre el de testeo
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="6eec6f7e" outputId="202daf1d-1b24-466e-99e2-9714509fbfff" vscode={"languageId": "python"}
y_pred_modelo_2
```

Graficamos

```python vscode={"languageId": "python"}
y_predic_cat_modelo_2 = np.where(y_pred_modelo_2>0.50,1,0)
```

```python vscode={"languageId": "python"}
ds_validacion=pd.DataFrame(y_predic_cat_modelo_2,y_test).reset_index()
ds_validacion.columns=['y_pred','y_real']

tabla=pd.crosstab(ds_validacion.y_pred, ds_validacion.y_real)
grf=sns.heatmap(tabla,annot=True, cmap = 'Blues')

#plt.ticklabel_format(style='plain', axis='both')
plt.show()
```

```python vscode={"languageId": "python"}
print(classification_report(y_test,y_predic_cat_modelo_2))
print('F1-Score: {}'.format(f1_score(y_test, y_predic_cat_modelo_2, average='binary'))) 
cm = confusion_matrix(y_test,y_predic_cat_modelo_2)
sns.heatmap(cm, cmap='Blues',annot=True,fmt='g')
plt.xlabel('predecido')
plt.ylabel('verdadero')
```

```python vscode={"languageId": "python"}
dump(modelo_hotels_2, 'modelos/una_red_zafable_2.joblib')

```

```python id="a82b6519" vscode={"languageId": "python"}
y_pred_testeo = modelo_hotels_1.predict(hotelsdf_testeo_filtrado)
```

```python id="2fd7ed7e" vscode={"languageId": "python"}
y_pred_testeo
```

```python id="7498d2e6" vscode={"languageId": "python"}
y_pred_testeo_cat = np.where(y_pred_testeo>0.50,1,0)
```

```python vscode={"languageId": "python"}
df_resultados_pred = pd.DataFrame.from_records(y_pred_testeo_cat,columns = ["resultado"])
df_resultados_pred
```

```python id="5cb7fc52" vscode={"languageId": "python"}
df_submission = pd.DataFrame({'id': hotelsdf_pruebasOriginal['id'], 'is_canceled': df_resultados_pred["resultado"]})
df_submission.to_csv('submissions/red_zafable_2.csv', index=False)
df_submission
df_submission.head()
```

```python

```
