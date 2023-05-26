---
jupyter:
  jupytext:
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

<!-- #region id="05308741" -->
# Importamos
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="b7ee581c" outputId="f7d763e8-f7f9-4db8-e9e0-5f368145297c" vscode={"languageId": "python"}
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

<!-- #region id="dcf41670" -->
Constantes
<!-- #endregion -->

```python id="8c79a19a" vscode={"languageId": "python"}
# Constantes
JOBS=-2
SEED=9

np.random.seed(SEED)
tf.random.set_seed(SEED) 
```

<!-- #region id="18257c34" -->
importo el dataframe original para tener los ids
<!-- #endregion -->

```python id="565f7f5d" vscode={"languageId": "python"}
hotelsdf_pruebasOriginal = pd.read_csv("./hotels_test.csv")
```

<!-- #region id="4642d273" -->
Importamos los dataframes ya filtrados que generamos en el checkpoint 1. De esta forma evitamos tener todo el codigo de homologacion de dataframes de testeo y train aqui de nuevo
<!-- #endregion -->

```python id="50364276" vscode={"languageId": "python"}
hotelsdf_train_filtrado = pd.read_csv("hotels_filtrado_train.csv")
hotelsdf_testeo_filtrado = pd.read_csv("hotels_filtrado_test.csv")
```

```python colab={"base_uri": "https://localhost:8080/"} id="da58be57" outputId="7ae3c0fc-ab75-474d-ea96-0c983bf5d3ce" vscode={"languageId": "python"}
set_test = set(hotelsdf_testeo_filtrado.columns)
set_modelo = set(hotelsdf_train_filtrado.columns)

missing = list(sorted(set_test - set_modelo))
added = list(sorted(set_modelo - set_test))

print('Faltan en arbol:', missing)
print('Sobran en arbol:', added)
```

<!-- #region id="0529db16" -->
Guardo datos antes de separacion train test
<!-- #endregion -->

```python id="KzIl_G3Nkhrq" vscode={"languageId": "python"}
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
"previous_bookings_not_canceled_num",
"previous_cancellations_num",
"required_car_parking_spaces_num",
"special_requests_num",
"weekend_nights_num",
"week_nights_num",
]
```

<!-- #region id="01a3e171" -->
## Generacion de datos para el entrenamiento de los modelos

Se genera un dataset con los datos necesarios para predecir la cancelacion y creamos un dataset conteniendo el target, para luego, generar conjuntos de test y train
<!-- #endregion -->

<!-- #region id="doFPZfUvixS9" -->
Considerando que previamente se vienen modificando los datos y aplicandoles un encoding del tipo One Hot, solo restaria realizar un escalado sobre las variables cuantitativas. No habra problema al aplicarlo sobre ningunda de estas debido a que todas fueron previamente analizadas en el checkpoint 1 y eliminamos outliers que pueden representar un problema en el analisis
<!-- #endregion -->

```python id="a1d3a8b9" vscode={"languageId": "python"}
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
## Escalado de los datasets de test y train


#Tenemos que escalar todos los valores de nuestro data set (excepto los valores producidos por el one hot encoding

sScaler = StandardScaler()
sScaler.fit(pd.DataFrame(x_train[valoresNoBinarios]))

```

```python
x_train_transform_1=sScaler.transform(pd.DataFrame(x_train[valoresNoBinarios]))
x_test_transform_1=sScaler.transform(pd.DataFrame(x_test[valoresNoBinarios]))
hotelsdf_testeo_transform=sScaler.transform(pd.DataFrame(hotelsdf_testeo_filtrado[valoresNoBinarios]))
```

```python
#Creamos un nuevo dataframe con los valores escalados
x_train_escalado = x_train.copy()
x_test_escalado = x_test.copy()
hotelsdf_testeo_escalado = hotelsdf_testeo_filtrado.copy()
```

```python
#Le asignamos los nuevos valores escalados y mantenemos los valores del one hot encoding
for i in range(len(valoresNoBinarios)):
    x_train_escalado[valoresNoBinarios[i]]=x_train_transform_1[:,i]
    x_test_escalado[valoresNoBinarios[i]]=x_test_transform_1[:,i]
    hotelsdf_testeo_filtrado[valoresNoBinarios[i]] = hotelsdf_testeo_transform[:,i]
```

```python
# cuantitativas = [
# "adult_num",
# "arrival_month_day",
# "arrival_week_number",
# "arrival_year",
# "average_daily_rate",
# "babies_num",
# "booking_changes_num",
# "children_num",
# "days_in_waiting_list",
# "lead_time",
# "previous_cancellations_num",
# "required_car_parking_spaces_num",
# "special_requests_num",
# "weekend_nights_num",
# "week_nights_num",
# ]


# from sklearn.preprocessing import StandardScaler
# sscaler=StandardScaler()
# sscaler.fit(pd.DataFrame(x_train[cuantitativas])
```

```python id="IIm5Jlngl37a" vscode={"languageId": "python"}
# x_train_transform_1=sscaler.transform(pd.DataFrame(x_train[cuantitativas]))
# x_test_transform_1=sscaler.transform(pd.DataFrame(x_test[cuantitativas]))
# hotelsdf_testeo_transform=sscaler.transform(pd.DataFrame(hotelsdf_testeo_filtrado[cuantitativas]))



# for cuantitativa in cuantitativas:
#   x_train[cuantitativa] = x_train_transform_1[:,0]
#   x_test[cuantitativa] = x_test_transform_1[:,0]
#   hotelsdf_testeo_filtrado[cuantitativa] = hotelsdf_testeo_transform[:,0]
```

<!-- #region id="f076865e" -->
Imports para armar la red
<!-- #endregion -->

```python id="22bb34c5" vscode={"languageId": "python"}
# import tensorflow as tf
# from tensorflow import keras
# from keras.utils.vis_utils import plot_model
# # import visualkeras

# np.random.seed(9)
# tf.random.set_seed(9) 

# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning) 
```

### RED 1


Creamos una red muy simple con una unica capa oculta y funcion de activacion relu

```python colab={"base_uri": "https://localhost:8080/"} id="60da0c2b" outputId="c3379a9e-f818-40ee-b033-2a56c767619b" vscode={"languageId": "python"}
cant_clases = 1


#d_in=len(y_train)
d_in=len(x_train_escalado.columns)

modelo_hotels_1 = keras.Sequential([
    keras.layers.Dense(8,input_shape=(d_in,),activation ='relu'),
    keras.layers.Dense(cant_clases, activation='sigmoid'),
])


modelo_hotels_1.summary()
```

La entrenamos con el optimizador Nadam puesto que vimos en clase que es el mejor. 

```python id="e0599e98" vscode={"languageId": "python"}
modelo_hotels_1.compile(
  optimizer=keras.optimizers.Nadam(learning_rate=0.01), 
  loss='binary_crossentropy', 
  # metricas para ir calculando en cada iteracion o batch 
  metrics=['AUC'], 
)

cant_epochs=80

historia_modelo_hotel_1=modelo_hotels_1.fit(x_train_escalado,y_train,epochs=cant_epochs,batch_size=16,verbose=False)
```

```python
y_pred = modelo_hotels_1.predict(x_test_escalado)
y_predic_cat_ej1 = np.where(y_pred>0.7,1,0)
historia_modelo_hotels_1=modelo_hotels_1.fit(x_train,y_train,epochs=cant_epochs,batch_size=16,verbose=False)
```

Miramos la metrica AUC en funcion de la cantidad de epocas para enocntrar un valor aceptable

```python vscode={"languageId": "python"}
epochs = range(cant_epochs)

plt.plot(epochs, historia_modelo_hotels_1.history['auc'], color='orange', label='AUC')
plt.xlabel("epochs")
plt.ylabel("AUC")
plt.legend()
```

Predecimos

```python vscode={"languageId": "python"}
y_pred_modelo_1 = modelo_hotels_1.predict(x_test)
y_predic_cat_modelo_1 = np.where(y_pred_modelo_1>0.50,1,0)
```

Mostramos los resultados de este primer modelo

```python vscode={"languageId": "python"}
print(classification_report(y_test,y_predic_cat_modelo_1))
print('F1-Score: {}'.format(f1_score(y_test, y_predic_cat_modelo_1, average='binary'))) 
cm = confusion_matrix(y_test,y_predic_cat_modelo_1)
sns.heatmap(cm, cmap='Blues',annot=True,fmt='g')
plt.xlabel('predecido')
plt.ylabel('verdadero')
```

Los resultados son relativamente buenos y no se ve que el modelo este muy sesgado. Aumentamos la cantidad de nueronas para intentar mejorar el f1_score


### RED 2

```python vscode={"languageId": "python"}
cant_clases = 1

d_in=len(x_train.columns)

modelo_hotels_2= keras.Sequential([
    keras.layers.Dense(8,input_shape=(d_in,),activation ='relu'),
    keras.layers.Dense(16,input_shape=(d_in,),activation ='relu'),
    keras.layers.Dense(32,input_shape=(d_in,),activation ='relu'),
    keras.layers.Dense(64,input_shape=(d_in,),activation ='relu'),
    keras.layers.Dense(cant_clases, activation='sigmoid'),])

modelo_hotels_2.compile(
  optimizer=keras.optimizers.Nadam(learning_rate=0.01), 
  loss='binary_crossentropy', 
  # metricas para ir calculando en cada iteracion o batch 
  metrics=['AUC'], 
)

cant_epochs=80

historia_modelo_hotels_2=modelo_hotels_2.fit(x_train,y_train,epochs=cant_epochs,batch_size=16,verbose=False)
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
y_pred_modelo_2
```

Graficamos

```python vscode={"languageId": "python"}
y_predic_cat_modelo_2 = np.where(y_pred_modelo_2>0.50,1,0)
```

```python vscode={"languageId": "python"}
print(classification_report(y_test,y_predic_cat_modelo_2))
print('F1-Score: {}'.format(f1_score(y_test, y_predic_cat_modelo_2, average='binary'))) 
cm = confusion_matrix(y_test,y_predic_cat_modelo_2)
sns.heatmap(cm, cmap='Blues',annot=True,fmt='g')
plt.xlabel('predecido')
plt.ylabel('verdadero')
```

El f1_score practicamente no mejoro con el aumnento de neruronas y capas. Sin embargo en ambos modelos (modelo_hotels_1 y 2) vemos que a partir de las 40 epocas el AUC practicamente no varia. Entonces bajamos la cantidad de epocas hasta ese valor para obtener una red mas optima Y tambien quito algunas capas para que la red sea mas simple porque no tiene sentido dejar una red mas compleja que no mejora el f1_score


### RED 3

```python vscode={"languageId": "python"}
cant_clases = 1

d_in=len(x_train.columns)

modelo_hotels_3= keras.Sequential([
    keras.layers.Dense(8,input_shape=(d_in,),activation ='relu'),
    keras.layers.Dense(16,input_shape=(d_in,),activation ='relu'),
    keras.layers.Dense(cant_clases, activation='sigmoid'),])

modelo_hotels_3.compile(
  optimizer=keras.optimizers.Nadam(learning_rate=0.01), 
  loss='binary_crossentropy', 
  # metricas para ir calculando en cada iteracion o batch 
  metrics=['AUC'], 
)

cant_epochs=40

historia_modelo_hotels_3=modelo_hotels_3.fit(x_train,y_train,epochs=cant_epochs,batch_size=16,verbose=False)
```

Vemos el AUC contra la cantidad de epocas

```python vscode={"languageId": "python"}
epochs = range(cant_epochs)

plt.plot(epochs, historia_modelo_hotels_3.history['auc'], color='orange', label='AUC')
plt.xlabel("epochs")
plt.ylabel("auc")
plt.legend()
```

```python vscode={"languageId": "python"}
y_pred_modelo_3 = modelo_hotels_3.predict(x_test)
y_pred_modelo_3
```

```python vscode={"languageId": "python"}
y_predic_cat_modelo_3 = np.where(y_pred_modelo_3>0.50,1,0)
```

Graficamos

```python vscode={"languageId": "python"}
print(classification_report(y_test,y_predic_cat_modelo_3))
print('F1-Score: {}'.format(f1_score(y_test, y_predic_cat_modelo_3, average='binary'))) 
cm = confusion_matrix(y_test,y_predic_cat_modelo_3)
sns.heatmap(cm, cmap='Blues',annot=True,fmt='g')
plt.xlabel('predecido')
plt.ylabel('verdadero')
```

Con la mitad de epocas y menor cantidad de capas y neuronas el valor del f1_score es practicamente el mismo. Nos quedamos con esta ultima red


### Prediccion mejor red

```python vscode={"languageId": "python"}
dump(modelo_hotels_3, 'modelos/red_neuronal_3.joblib')
```

```python id="a82b6519" vscode={"languageId": "python"}
y_pred_testeo = modelo_hotels_2.predict(hotelsdf_testeo_filtrado)
y_pred_testeo
```

```python id="7498d2e6" vscode={"languageId": "python"}
y_pred_testeo_cat = np.where(y_pred_testeo>0.4,1,0)
```

```python vscode={"languageId": "python"}
df_resultados_pred = pd.DataFrame.from_records(y_pred_testeo_cat,columns = ["resultado"])
df_resultados_pred
```

```python id="5cb7fc52" vscode={"languageId": "python"}
df_submission = pd.DataFrame({'id': hotelsdf_pruebasOriginal['id'], 'is_canceled': df_resultados_pred["resultado"]})
df_submission.to_csv('submissions/red_neuronal_3.csv', index=False)
df_submission
df_submission.head()
```

# Validacion Cruzada

```python
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
