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
from dict_paises import COUNTRY_ALPHA3_TO_COUNTRY_ALPHA2, COUNTRY_ALPHA2_TO_CONTINENT
from joblib import dump, load
from os.path import exists

from sklearn.model_selection import StratifiedKFold, KFold,RandomizedSearchCV, train_test_split, cross_validate
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import confusion_matrix, classification_report , f1_score, make_scorer, precision_score, recall_score, accuracy_score,f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.fixes import loguniform
from sklearn.neighbors import KNeighborsClassifier

import xgboost as xgb

#Si estamos  en colab tenemos que instalar la libreria "dtreeviz" aparte. 
if IN_COLAB == True:
    !pip install 'dtreeviz'
import dtreeviz as dtreeviz

#Para eliminar los warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

```

```python
# Constantes
JOBS=-2
SEED=9
```

## Cargamos el dataframe de testeo

```python
hotelsdf_pruebasOriginal = pd.read_csv("./hotels_test.csv")
hotelsdf_pruebas = hotelsdf_pruebasOriginal.copy()
```

# Adaptación de los datos al modelo

Ajustamos los datos encontrados en los datasets de manera conveniente para el analisis posterior con los modelos de ensambles


## Cargamos nuestro dataframe previamente analizado

Vamos a crear una copia de nuestro dataframe

```python
hotelsdf_analizado = pd.read_csv("./dataframeCheckpoint1.csv")
hotelsdf_modelo = hotelsdf_analizado.copy()
print("El data frame esta compuesto por "f"{hotelsdf_modelo.shape[0]}"" filas y "f"{hotelsdf_modelo.shape[1]}"" columnas")
```

Un vistazo básico a la información contenida en el dataframe:

```python
pd.concat([hotelsdf_modelo.head(2), hotelsdf_modelo.sample(5), hotelsdf_modelo.tail(2)])
```

Vemos que tenemos una columa extra "Unnamed: 0". Esta hace referencia la columna de origen del registro. Procedemos a borrarla

```python
hotelsdf_modelo.drop("Unnamed: 0", axis=1, inplace=True)
hotelsdf_modelo.reset_index(drop=True)
print()
```

## Transformacion de las columnas

Para que los datos sean compatibles tomamos todas las columnas correspondientes a las variables cualitativas y procedemos a identificarlas:

```python
valoresAConvertir = hotelsdf_modelo.dtypes[(hotelsdf_modelo.dtypes !='int64') & (hotelsdf_modelo.dtypes !='float64')].index
valoresAConvertir = valoresAConvertir.to_list()
valoresAConvertir
```

Sin embargo, no todas estas columnas nos van a servir para nuestro analisis.

### Booking ID

Vamos a empezar removiendo booking\_id visto en como no la necesitamos para analisis

```python
hotelsdf_modelo.drop("booking_id", axis=1, inplace=True)
hotelsdf_modelo.reset_index(drop=True)
valoresAConvertir.remove('booking_id')
```

### Reservation Status & Reservation status date

Reservation Status nos dice el estado de la reservacion, si fue cancelada o no y reservation status date nos marca la fecha en la que cambio el estado. 
Estas dos columnas nos son redundantes

```python
hotelsdf_modelo.drop("reservation_status", axis=1, inplace=True)
hotelsdf_modelo.reset_index(drop=True)
valoresAConvertir.remove('reservation_status')
```

```python
hotelsdf_modelo.drop("reservation_status_date", axis=1, inplace=True)
hotelsdf_modelo.reset_index(drop=True)
valoresAConvertir.remove('reservation_status_date')
```

### Country

Ajustamos los posibles valores que pueda tomar la variable country haciendo usos de diccionarios externos, con el proposito, de trabajar con el contienente de cada entrada

```python
hotelsdf_modelo["continente"] = hotelsdf_modelo["country"].replace(COUNTRY_ALPHA3_TO_COUNTRY_ALPHA2)
hotelsdf_modelo["continente"] = hotelsdf_modelo["continente"].replace(COUNTRY_ALPHA2_TO_CONTINENT)
```

```python
country = hotelsdf_modelo['country'].unique().tolist()
print(country) 
```

```python
country = hotelsdf_modelo['continente'].unique().tolist()
print(country) 
```

Observando el tratado de los datos identificamos nuevos outliers

"ATA" refiere al **continente** de Antartida. Al ser un valor tan fuera de lo comun y tener una sola ocurrencia decidimos eliminarlo del dataframe

```python
hotelsdf_modelo.drop((hotelsdf_modelo[hotelsdf_modelo["country"] == "ATA"].index.values),inplace=True)
hotelsdf_modelo.reset_index(drop=True)
print()
```

"UMI" hace referenca a unas islas cerca de Hawaii. Al ser un unico caso y tener una poblacion de 300 habitantes, decidimos considerarlo como Estados Unidos, es decir America del Norte

Fuentes:
- https://www.iso.org/obp/ui/#iso:code:3166:UM
- https://en.wikipedia.org/wiki/United_States_Minor_Outlying_Islands

```python
hotelsdf_modelo.loc[hotelsdf_modelo['country'] == "UMI", 'country'] = 'North America'
hotelsdf_modelo.loc[hotelsdf_modelo['continente'] == "UMI", 'continente'] = 'North America'
```

Con estos nuevos cambios, la columna continente toma los siguientes valores

```python
continente = hotelsdf_modelo['continente'].unique().tolist()
print(continente) 
```

Procedemos a dropear la columna de country

```python
hotelsdf_modelo=hotelsdf_modelo.drop(['country'], axis='columns', inplace=False)
valoresAConvertir.remove('country')
valoresAConvertir.append('continente')
hotelsdf_modelo.reset_index(drop=True)
```

```python
valoresAConvertir
```

## One hot encoding

Aplicamos la tecnica de one hot encodig para hacer el dataset compatible con los modelos

```python
hotelsdf_modelo = pd.get_dummies(hotelsdf_modelo, columns=valoresAConvertir, drop_first=True)
```

Vamos a observar como nos quedo el dataframe despues del one hot encoding

```python
hotelsdf_modelo.head()
```

Observamos que hay una **gran** cantidad de columnas


### Ajuste del dataset de pruebas

Tratamos de la misma manera al dataset de pruebas para hacerlo compatible con el modelo

Empezamos cambiando el nombre de las columnas para que coincida con el de nuestro dataframe

```python
nuevas_columnas = {
    'adr':'average_daily_rate',
    'adults':'adult_num',
    'agent':'agent_id',
    'arrival_date_day_of_month':'arrival_month_day',
    'arrival_date_month':'arrival_month',
    'arrival_date_week_number':'arrival_week_number',
    'arrival_date_year':'arrival_year',
    'assigned_room_type':'assigned_room_type',
    'babies':'babies_num',
    'booking_changes':'booking_changes_num',
    'children':'children_num',
    'company':'company_id',
    'country':'country',
    'customer_type':'customer_type',
    'days_in_waiting_list':'days_in_waiting_list',
    'deposit_type':'deposit_type',
    'distribution_channel':'distribution_channel',
    'hotel':'hotel_name',
    'id':'booking_id',
    'is_repeated_guest':'is_repeated_guest',
    'lead_time':'lead_time',
    'market_segment':'market_segment_type',
    'meal':'meal_type',
    'previous_bookings_not_canceled':'previous_bookings_not_canceled_num',
    'previous_cancellations':'previous_cancellations_num',
    'required_car_parking_spaces':'required_car_parking_spaces_num',
    'reserved_room_type':'reserved_room_type',
    'stays_in_weekend_nights':'weekend_nights_num',
    'stays_in_week_nights':'week_nights_num',
    'total_of_special_requests':'special_requests_num',
}

hotelsdf_pruebas.rename(columns = nuevas_columnas, inplace = True)
```

Procesamos datos faltantes

#### Dias Totales

Añadimos la columna agregada en el analisis exploratorio previo

```python
hotelsdf_pruebas["dias_totales"] = hotelsdf_pruebas["week_nights_num"] + hotelsdf_pruebas["weekend_nights_num"]
```

#### Datos faltantes

```python
hotelsdf_pruebas.isnull().sum()
```

```python
print("Vemos que 'company id' tiene un " + str( (hotelsdf_pruebas["company_id"].isnull().sum() * 100) / len(hotelsdf_pruebas)  ) + "% de datos faltantes.")
print("Por esto decidimos eliminar la columna (tanto en el dataset de testeo como en el de entrenamiento)")
```

```python
hotelsdf_pruebas.drop("company_id", axis=1, inplace=True)
hotelsdf_pruebas.reset_index(drop=True)
```

### Valores a convertir

De ser posible aplicamos el cirterio anterior

```python
valores_a_convertir_pruebas = hotelsdf_pruebas.dtypes[(hotelsdf_pruebas.dtypes !='int64') & (hotelsdf_pruebas.dtypes !='float64')].index
valores_a_convertir_pruebas = valores_a_convertir_pruebas.to_list()
valores_a_convertir_pruebas
```

#### Booking ID

```python
hotelsdf_pruebas.drop("booking_id", axis=1, inplace=True)
hotelsdf_pruebas.reset_index(drop=True)
valores_a_convertir_pruebas.remove('booking_id')
```

#### Agent ID

Tomamos el mismo criterio que el checkpoint 1. Transformamos a 0

```python
hotelsdf_pruebas.loc[hotelsdf_pruebas['agent_id'].isnull(), 'agent_id'] = 0
```

#### Reservation Status & Reservation status date

Dropeamos estas columnas debido a que no nos dan ninguna informacion adicional

```python
hotelsdf_pruebas.drop("reservation_status_date", axis=1, inplace=True)
hotelsdf_pruebas.reset_index(drop=True)
valores_a_convertir_pruebas.remove('reservation_status_date')
```

#### Country y Continents

Para los valores faltantes de la columna country aplicamos el criterio del analisis exploratorio y los cambiamos por Portugal

```python
hotelsdf_pruebas.loc[hotelsdf_pruebas['country'].isnull(), 'country'] = 'PRT'
```

```python
hotelsdf_pruebas["continente"] = hotelsdf_pruebas["country"].replace(COUNTRY_ALPHA3_TO_COUNTRY_ALPHA2)
hotelsdf_pruebas["continente"] = hotelsdf_pruebas["continente"].replace(COUNTRY_ALPHA2_TO_CONTINENT)
```

```python
country = hotelsdf_pruebas['country'].unique().tolist()
valores_a_convertir_pruebas.append("continente")
print(country) 
```

```python
continentes = hotelsdf_pruebas['continente'].unique().tolist()
print(continentes) 
```

Tal como ocurrio con el dataset de Train, observamos que hay algunos continente (y por tanto sus paises y registros asociados) que parecen ser outliers.
Los estudiamos

```python
hotelsdf_pruebas[ hotelsdf_pruebas['continente'] =="ATA"]
```

Hay un registro correspondiente a "Antartida". como no podemos dropearlo, le ponemos de continente "north america".\
Le asignamos el valor de America del norte debido a que estados unidos es el pais con mas bases en la antartica

```python
hotelsdf_pruebas.loc[hotelsdf_pruebas['continente'] == "ATA", 'continente'] = "North America"
```

```python
hotelsdf_pruebas[ hotelsdf_pruebas['continente'] =="ATF"]
```

"ATF", que es la sigla de Fr. So. Ant. Tr (French southern and antartic lands).
Ponemos su contienente en Europa. 

```python
hotelsdf_pruebas.loc[hotelsdf_pruebas['continente'] == "ATF", 'continente'] = "Europe"
```

```python
hotelsdf_pruebas[hotelsdf_pruebas['continente'] =="ATF"]
```

#### Analisis de valores faltantes de continente

```python
hotelsdf_pruebas[hotelsdf_pruebas['continente'].isna()]
```

Vemos que hay una serie de registros que no tienen el dato del pais. Sin embargo, no son muchos. Debido a esto, vamos a asignarle estos registros el valor de aquel contiente que tenga la mayor cantidad de registros

```python
sns.countplot(data = hotelsdf_pruebas, x = 'continente', palette= 'Set2')
plt.title('Cantidad de registros por continente')
plt.xlabel('Continente')
plt.ylabel('Cantidad de registros')
```

Vemos que el continente con mayor cantidad de registros es europa, asique lo asignamos a ese valor

```python
hotelsdf_pruebas.loc[hotelsdf_pruebas['continente'].isnull(), 'country'] = 'Europe'
```

Miro q se hayan cambiado bien todos los continentes y no haya valores raros

```python
continentes = hotelsdf_pruebas['continente'].unique().tolist()
print(continentes)
```

Como hicimos con el dataset de train, y ya habiendo procesado la columna continente, dropeamos la columna country

```python
hotelsdf_pruebas=hotelsdf_pruebas.drop(['country'], axis='columns', inplace=False)
hotelsdf_pruebas.reset_index(drop=True)
valores_a_convertir_pruebas.remove('country')
```

#### previous bookings not cancelled

Al igual q en el train, dropeamos esta col

```python
hotelsdf_pruebas=hotelsdf_pruebas.drop(['previous_bookings_not_canceled_num'], axis='columns', inplace=False)
hotelsdf_pruebas.reset_index(drop=True)
```

```python
hotelsdf_pruebas.isnull().sum()
```

### One hot encoding del testeo

De la misma manera al dataset de pruebas aplicamos one hot encoding sobre las columnas de variables cualitativas

```python
hotelsdf_pruebas = pd.get_dummies(hotelsdf_pruebas, columns=valores_a_convertir_pruebas, drop_first=True)
hotelsdf_pruebas.head()
```

### Corroboracion de columnas

Despues de todas estas transformaciones vamos a corrobar que los dataframes tengan la misma cantidad de columnas.

```python
set_test = set(hotelsdf_pruebas.columns)
set_modelo = set(hotelsdf_modelo.columns)

missing = list(sorted(set_test - set_modelo))
added = list(sorted(set_modelo - set_test))

print('Faltan en arbol:', missing)
print('Sobran en arbol:', added)
```

Vemos que en el dataframe del arbol nos sobra la columna "is canceled", cosa que hace sentido ya que esa es la columna con la que vamos a entrenar al dataset. Sin embargo, vemos que tambien hay 3 columnas que faltan en el dataset de arbol. 

Vamos a reasignar los valores de las columnas de test para que coincidan.

El siguiente codigo nos calcula cuantas personas tiene cada tipo de cuarto

```python
cant_cuartos = {}
cant_casos_sumado = 0

cant_cuartos["A"] = 0 #Arrancamos asignado 0 a los cuartos de A. Estos fueron removidos por el one hot. Lo vamos a calcular al final.
for letra in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']:
    tipo_cuarto = 'reserved_room_type_' + letra
    cant_casos_sumado += 1
    if tipo_cuarto not in hotelsdf_pruebas.columns:
        continue
    hotelsdf_pruebas[tipo_cuarto]
    resultado = hotelsdf_pruebas[hotelsdf_pruebas[tipo_cuarto] == 1][tipo_cuarto].sum()
    cant_cuartos[letra] = resultado

cuartosA = len(hotelsdf_pruebas) - cant_casos_sumado
cant_cuartos["A"] = cuartosA


cant_cuartos
```

Vemos que L y P tienen una extremadamente pequena cantidad de apariciones. \
Lo vamos a anadir al roomtype A al ser el que tiene la mayor cantidad de apariciones.

Para anadirlos a la columna a, simplemente tenemos que eliminar las columnas L y P (ya que la columna A es la eliminada por el one hot)

```python
hotelsdf_pruebas.drop("reserved_room_type_L", axis=1, inplace=True)
hotelsdf_pruebas.drop("reserved_room_type_P", axis=1, inplace=True)
hotelsdf_pruebas.reset_index(drop=True)
print()
```

Vamos a aplicar el mismo criterio a assigned room type

```python
cant_cuartos = {}
cant_casos_sumado = 0

cant_cuartos["A"] = 0 #Arrancamos asignado 0 a los cuartos de A. Estos fueron removidos por el one hot. Lo vamos a calcular al final.
for letra in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']:
    tipo_cuarto = 'assigned_room_type_' + letra
    cant_casos_sumado += 1
    if tipo_cuarto not in hotelsdf_pruebas.columns:
        continue
    hotelsdf_pruebas[tipo_cuarto]
    resultado = hotelsdf_pruebas[hotelsdf_pruebas[tipo_cuarto] == 1][tipo_cuarto].sum()
    cant_cuartos[letra] = resultado

cuartosA = len(hotelsdf_pruebas) - cant_casos_sumado
cant_cuartos["A"] = cuartosA


cant_cuartos
```

Aca tambien vemos que P tiene muy pocas aparciones. Asique aplicamos el mismo criterio de antes

```python
hotelsdf_pruebas.drop("assigned_room_type_P", axis=1, inplace=True)
hotelsdf_pruebas.reset_index(drop=True)
print()
```

Vemos ahora que nuestras columnas coinciden

```python
set_test = set(hotelsdf_pruebas.columns)
set_modelo = set(hotelsdf_modelo.columns)

missing = list(sorted(set_test - set_modelo))
added = list(sorted(set_modelo - set_test))

print('Faltan en arbol:', missing)
print('Sobran en arbol:', added)
```

# Generacion de datos para el entrenamiento de los modelos

Se genera un dataset con los datos necesarios para predecir la cancelacion y creamos un dataset conteniendo el target, para luego, generar conjuntos de test y train

```python
hotelsdf_modelo_x=hotelsdf_modelo.drop(['is_canceled'], axis='columns', inplace=False)

hotelsdf_modelo_y = hotelsdf_modelo['is_canceled'].copy()

x_train, x_test, y_train, y_test = train_test_split(hotelsdf_modelo_x,
                                                    hotelsdf_modelo_y, 
                                                    test_size=0.3,  #proporcion 70/30
                                                    random_state=SEED) #Semilla 9, como el Equipo !!
```

# KNN

Entrenamos un primer modelo de KNN usando los datos previamente tratados

## KNN base

En primera instancia entrenamos un modelo sin optimizar hiperparametros, de manera que, se obtiene una medida de la predicción base que tiene el modelo.

Creamos el modelo y lo entrenamos:

```python
if not exists('modelos/knn_base.joblib'):
    knn_base = KNeighborsClassifier()
    knn_base.get_params()
    knn_base.fit(x_train, y_train)
    dump(knn_base, 'modelos/knn_base.joblib')
else:
    knn_base = load('modelos/knn_base.joblib')
```

```python
y_pred = knn_base.predict(x_test)
```

Observamos el comportamiento del modelo base

```python
print('correctas: ', np.sum(y_test == y_pred))
print('total: ', len(y_test))
```

Observamos mediante la matriz de confusion el comportamiento del modelo base con los datos de prueba 

```python
print(classification_report(y_test,y_pred))

confusion_knn_base = confusion_matrix(y_test,y_pred)
sns.heatmap(confusion_knn_base, cmap='Blues',annot=True,fmt='g')
plt.xlabel('Predicho')
plt.ylabel('Verdadero')
```

Basado en el grafico, es posible observar que el modelo base ha obtenido un desempeño moderado en los datos de prueba a pesar de no haber recibido ningun tipo de optimización 

Generamos una prediccón para kaggle con el modelo base

```python
y_pred = knn_base.predict(hotelsdf_pruebas)
y_pred
df_submission = pd.DataFrame({'id': hotelsdf_pruebasOriginal['id'], 'is_canceled': y_pred})

if not exists('submissions/knn_base.csv'):
    df_submission.to_csv('submissions/knn_base.csv', index=False)
```

## Busqueda de hiperparametros

### Modificar los k vecinos

Realizamos una busqueda de cuales son los valores de k para los cuales el modelo tiene un mejor desempeño 

```python
metricas = []
cant_vecinos = range(1, 30) 

if not exists('modelos/metricas.joblib'):
    for n in cant_vecinos:
        knn = KNeighborsClassifier(n_neighbors = n, n_jobs=JOBS)
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)
        metricas.append( (n, (y_test == y_pred).sum())) 
else:
    metricas = load('modelos/metricas.joblib')
```

De la prueba anterior observamos el comportamiento que tiene 

```python
plt.figure(figsize = (6,5))

df_metrics = pd.DataFrame(metricas, columns=['cant_vecinos', 'correctos'])

ax = df_metrics.plot( x='cant_vecinos', 
                      y='correctos',
                      title='Aciertos vs Cantidad de Vecinos'
                    )

ax.set_ylabel("Cantidad de aciertos")
ax.set_xlabel("Cantidad de Vecinos")
plt.show()
```

Por otro lado observamos el comportamiento de la presicion 

```python
knn_metricas = []

if not exists('modelos/knn_metricas.joblib'):
    for n in cant_vecinos:
      knn = KNeighborsClassifier(n_neighbors = n)
      scores=cross_val_score(knn,x_train,y_train,cv=10,scoring='accuracy')
      knn_metricas.append(scores.mean())
    dump(knn_metricas, 'modelos/knn_metricas.joblib')
else:
    knn_metricas = load('modelos/knn_metricas.joblib')
```

```python
plt.plot(cant_vecinos, knn_metricas)
plt.xlabel('Cantidad de Vecinos')
plt.ylabel('Cross Validation Accuracy')
plt.title('Accuracy vs Cantidad de Vecinos')
plt.show()
```

Podemos concluir de las graficas anteriores que el modelo tiende a empeorar a medida que aumentan la cantidad de vecinos, de todos modos la variación de la presión es considerablemente pequeña. Por lo tanto consideraremos el mismo rango de valores a la hora de realizar el cross validation search

### Random search cross validation

Buscamos la mejor combinación de hiperparametros con la intención de mejorar las metricas del modelo y a su vez mejorar la performance del mismo

```python
if not exists('modelos/RCV_knn.joblib'):

    params_grid={ 'n_neighbors':range(1,15), 
                  'weights':['distance','uniform'],
                  'algorithm':['ball_tree', 'kd_tree'],
                  'metric':['euclidean','manhattan']
                 }


    knn_optimizado = KNeighborsClassifier()
    combinaciones = 10
    k_folds = 10
    metrica_fn = make_scorer(sk.metrics.f1_score)

    parametros = RandomizedSearchCV(
                estimator=knn_optimizado, 
                param_distributions = params_grid, 
                cv=k_folds, 
                scoring=metrica_fn, 
                n_iter=combinaciones, 
                random_state=9)

    parametros.fit(x_train, y_train)
    parametros.cv_results_['mean_test_score']

    dump(parametros, 'modelos/RCV_knn.joblib')
else:
    parametros = load('modelos/RCV_knn.joblib')
```

Observamos el comportamiento de los mejores hiperparamtros encontrados 

```python
print(parametros)
print(parametros.best_params_)
print(parametros.best_score_)
```

Creamos y entrenamos el modelo con los mejores imperparametros 

```python
if not exists('modelos/knn_optimizado.joblib'):
    knn_optimizado = KNeighborsClassifier(**parametros.best_params_)
    knn_optimizado.fit(x_train, y_train)
else:
    knn_optimizado = load('modelos/knn_optimizado.joblib')
```

## Cross validation

Verificamos la eficacia del modelo y sus hiperparametros mediante la validación cruzada

```python
if not exists('modelos/knn_optimizado.joblib'):

    kfoldcv =StratifiedKFold(n_splits=k_folds) 
    resultados = cross_validate(knn_optimizado,x_train, y_train, cv=kfoldcv,scoring=metrica_fn,return_estimator=True)
    metricas_knn = resultados['test_score']
    knn_optimizado = resultados['estimator'][np.where(metricas_knn==max(metricas_knn))[0][0]]
    dump(knn_optimizado, 'modelos/knn_optimizado.joblib')

```

Observamos la distribucion de la metrica f1 a lo largo de los entrenamientos

```python
metric_labelsCV = ['F1 Score']*len(metricas_knn) 
sns.set_context('talk')
sns.set_style("darkgrid")
plt.figure(figsize=(8,8))
sns.boxplot(metricas_knn)
plt.title("Modelo entrenado con 10 folds")
```

Mostramos la matriz de confusión del modelo y observamos su desempeño global

```python
y_pred= knn_optimizado.predict(x_test)
print(classification_report(y_test,y_pred))
print('F1-Score: {}'.format(f1_score(y_test, y_pred, average='binary'))) 
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm, cmap='Blues',annot=True,fmt='g')
plt.xlabel('predecido')
plt.ylabel('verdadero')
```

Una vez entrenado y guardado el modelo generamos una predicción para kaggle.

```python
y_pred = knn_optimizado.predict(hotelsdf_pruebas)
df_submission = pd.DataFrame({'id': hotelsdf_pruebasOriginal['id'], 'is_canceled': y_pred})

if not exists('submissions/knn_optimizado.csv'):
    df_submission.to_csv('submissions/knn_optimizado.csv', index=False)
```

Si comparamos el desempeño del modelo tanto en ejecución del analisis como en las predicciones de kaggle podemos observar que el modelo optimizado presenta una mejoria de al menos 0.05 puntos a comparación del modelo base. Por otro lado, el modelo knn no presenta una mejora global al analisis hecho con un árbol de decisión

# SVM 






### Librerias y Funciones

```python
def metricas(y_pred,y_test):

  print(classification_report(y_test,y_pred))
  
  cm = confusion_matrix(y_test,y_pred)
  sns.heatmap(cm, cmap='Blues',annot=True,fmt='g')
  plt.xlabel('Predicted')
  plt.ylabel('True')
```

Hacemos un GridSeacrh para ver cual es el mejor kernel a utilizar.
OJO, TOMA 32 MIN aprox correrlo

```python
if not exists('modelos/gridcv_svm_kernel.joblib'):
    parametros = { 'kernel': ["linear", "poly","rbf"]}


    clf_tres = SVC(random_state=9)

    scorer_fn = make_scorer(sk.metrics.f1_score)

    gridcv_svm_tres = GridSearchCV(estimator=clf_tres,
                                  param_grid= parametros,
                                  scoring=scorer_fn,
                                  n_jobs=JOBS #-1
                                  ) 

    #lo entreno sobre los datos
    gridcv_svm_tres.fit(x_train, y_train)

    print("Mostramos los mejores resultados: ")
    print(gridcv_svm_tres.best_params_)
    print()
    print("Mostramos el mejor resultado obtenido de busqueda aleatoria: ")
    print("f1_score: ",gridcv_svm_tres.best_score_)
    
    dump(gridcv_svm_tres, 'modelos/gridcv_svm_kernel.joblib')
```

```python
gridcv_svm_kernel = load('modelos/gridcv_svm_kernel.joblib')
```

Obtenemos que el mejor Kernel es el linel con un f1_score de 0,75.


A continuacion, probamos igualmente los 3 Kernels para ver si podemos obtener alguna optimizacion o mejora del valor anterior


### Modifico Kernels para ver cual se adapta mejor


### Lineal 

```python
if not exists('modelos/svm_lineal_mejor_performance.joblib'):
    #Creo un clasificador con kernel lineal y lo entreno
    svm_lineal_mejor_performance = SVC(kernel='linear', C=5, random_state=9)
    svm_lineal_mejor_performance.fit(x_train, y_train)

    #Hago la predicción y calculo las métricas
    y_pred_lin=svm_lineal_mejor_performance.predict(x_test)
    metricas(y_pred_lin,y_test)

    dump(svm_lineal_mejor_performance, 'modelos/svm_lineal_mejor_performance.joblib')

else:
    svm_lineal_mejor_performance = load('modelos/svm_lineal_mejor_performance.joblib')
    y_pred_lin=svm_lineal_mejor_performance.predict(x_test)
    metricas(y_pred_lin,y_test)

```

Con el kernel lineal, obtebemos un f1_score relativamente bueno (casi -por no decir exactamente igual- al obtenido con el GridSearch antes) aunq no mejor que el obtenido con el decission tree en la entrega anterior.Intentamos cambiar su parametro C para ver si conseguimos alguna leve mejora (no lo elevamos demasiado porque ya sabemos que overfittea). Este proceso fue hecho a "fuerza bruta" ya que no encontramos la manera de correr un Random/Grid search para variar solo el parametro C. Se probo con valores de C = [1, 5, 7, 10, 15, 100] y con todos se obtuvo un f1_score muy similar. Optamos por el valor de 5 ya que lo esperado es que un valor de C mas bajo nos entregue un modelo mas generalizable.


Hago un entrenamiento con cross validation para ver que el modelo sea generalizable\

**ATENCION: 10 MIN con core i5 + 16Gb RAM (sin archivos de joblib)**

```python
if not exists('modelos/svm_lineal_mejor_performance.joblib'):

    folds=5

    kfoldcv = StratifiedKFold(n_splits=folds)
    scorer_fn = make_scorer(sk.metrics.f1_score)
    resultados = cross_validate(svm_lineal_mejor_performance,x_train, y_train, cv=kfoldcv,scoring=scorer_fn,return_estimator=True)

    dump(resultados, 'modelos/resultados_cv_kernel_lineal.joblib')
else:
    resultados = load('modelos/resultados_cv_kernel_lineal.joblib')

```

```python
metricsCV=resultados['test_score']

if not exists('modelos/svm_lineal_mejor_performance.joblib'):
    svm_lineal_mejor_performance=resultados['estimator'][np.where(metricsCV==max(metricsCV))[0][0]]
    dump(svm_lineal_mejor_performance, 'modelos/svm_lineal_mejor_performance.joblib')

metricsCV
```

```python
metric_labelsCV = ['F1 Score']*len(metricsCV) 
sns.set_context('talk')
sns.set_style("darkgrid")
plt.figure(figsize=(8,8))
sns.boxplot(metricsCV)
plt.title("Modelo entrenado con 5 folds")
```

Se puede ver que no hay mucha variacion en los valores obtenidos por lo cual podemos concluir que es un modelo bueno para generalizar. A continuacion los resultados de probar con el dataset de testeo.

```python
svm_lineal_mejor_performance = load('modelos/svm_lineal_mejor_performance.joblib')
```

```python
y_pred= svm_lineal_mejor_performance.predict(x_test)
print(classification_report(y_test,y_pred))
print('F1-Score: {}'.format(f1_score(y_test, y_pred, average='binary'))) 
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm, cmap='Blues',annot=True,fmt='g')
plt.xlabel('Predicted')
plt.ylabel('True')
```

Se puede ver que si bien los resultados no son excelentes, son relativamente buenos (f1_score = 0,75). Lo esperado es que, segun lo estudiado en clase, recien al hacer los ensambles con varios estimadores mediocres -muy buena palabra- (un KNN, un SVM y un RF) obtendremos una mejora en el f1_score.


A continuacion deberiamos exportar el csv para submission a Kaggle. Puesto que no representaninguna mejora del score obtenido anteriormente no lo hacemos

```python
# df_submission = pd.DataFrame({'id': hotelsdf_pruebasOriginal['id'], 'is_canceled': y_pred})
# df_submission.to_csv('submissions/svm_lineal_mejor_performance.csv', index=False)

```

### Polinomico y Radial


El codigo a continuacion para ambos kernels se encuentra comentado en muchas partes debido al gran tiempo que demora entrenar SVM's con tantos datos (no sabemos cuanto exactamente cuanto ya que nunca pudimos terminar de correrlo). Esto se debe a que al utilizar Kernels Radial y Polinomico los algoritmos crean matrices de NXN demandando mucha RAM y CPU. Dejamos los snippets de codigo como prueba de ello.


### Polinomico
Creamos un SVM con Kernel polynomico con parametros por default (sin parametros)

**ATENCION: 3 MIN con core i5 + 16Gb RAM (sin modelos Joblib) **

```python
if not exists('modelos/clf_poly_no_optimizado.joblib'):

    #Creo un clasificador con kernel polinomico y lo entreno sobre los datos
    clf_poly_no_optimizado = SVC(kernel='poly', random_state=9)
    clf_poly_no_optimizado.fit(x_train, y_train)
    
    dump(clf_poly_no_optimizado, 'modelos/clf_poly_no_optimizado.joblib')

else:
    clf_poly_no_optimizado = load('modelos/clf_poly_no_optimizado.joblib')

#Hago la predicción y calculo las métricas
y_pred_pol=clf_poly_no_optimizado.predict(x_test)
metricas(y_pred_pol,y_test)
```

Con el kernel polinomico sin parametros obtenemos un f1_score bastante malo (0,6). Intentamos optimizarlo a continuacion.


#### Intento mejorar hiperparametros (da error)
A continuacion se deja el codigo que se intento utlizar para optimizar los hiperparametros con RandomizedSearchCV del SVM con Kernel polinomico. No se pudo obtener un resultado en tiempo razonable por ello se lo deja comentado

```python
# #vario hiperparaemtros en kernel polinomico
# clf_poly = SVC(kernel='poly')

# parametros = [ {'C': [0,75, 9, 1, 10, 100], 
#                 'gamma': [10, 0.001, 0.0001], 
#                 'kernel': ['poly']},
#  ]


# combinaciones=1 #2,3 se puso 1 para ver si tardaba menos. :(

# scorer_fn = make_scorer(sk.metrics.f1_score)


# Randomcv_svm = RandomizedSearchCV(estimator=clf_poly,
#                               #param_grid= parametros,
#                               param_distributions = parametros,
#                               scoring=scorer_fn,
#                               #cv=kfoldcv,
#                               n_iter=combinaciones,
#                               ) 

# #lo entreno sobre los datos
# Randomcv_svm.fit(x_train, y_train)

# #Hago la predicción y calculo las métricas
# Randomcv_svm.predict(x_test)
# metricas(Randomcv_svm,y_test)
```

Intentamos optimizarlo a mano. :))) (?

```python
if not exists ('modelos/svm_poly_mejor_performance.joblib'):
    svm_poly_mejor_performance = SVC(kernel='poly', C=5, degree=1, gamma=1, coef0=1, random_state=9)
    svm_poly_mejor_performance.fit(x_train, y_train)
    
    dump(svm_poly_mejor_performance, 'modelos/svm_poly_mejor_performance.joblib')
else:
    svm_poly_mejor_performance = load('modelos/svm_poly_mejor_performance.joblib')
    
#Hago la predicción y calculo las métricas
y_pred_pol=svm_poly_mejor_performance.predict(x_test)
metricas(y_pred_pol,y_test)
```

Obtebemos que con valores de c=5, degree = 1, gamma =1 y coef01, el f1_score es de aproximadamente 0,75, parecido al lineal


Hacemos cross validation para ver que el modelo sea generalizable


#### ATENCION tarda 12 Min (sin archivos de joblib)

```python
if not exists('modelos/svm_poly_mejor_performance.joblib'):
    folds=5

    kfoldcv = StratifiedKFold(n_splits=folds)
    scorer_fn = make_scorer(sk.metrics.f1_score)
    resultados = cross_validate(svm_poly_mejor_performance,x_train, y_train, cv=kfoldcv,scoring=scorer_fn,return_estimator=True)

    dump(resultados, 'modelos/resultados_cv_kernel_poly.joblib')
else:
    resultados = load('modelos/resultados_cv_kernel_poly.joblib')

```

```python
metricsCV=resultados['test_score']
if not exists('modelos/svm_lineal_mejor_performance.joblib'):
    svm_poly_mejor_performance=resultados['estimator'][np.where(metricsCV==max(metricsCV))[0][0]]
    dump(svm_poly_mejor_performance, 'modelos/svm_poly_mejor_performance.joblib')

metricsCV
```

```python
metric_labelsCV = ['F1 Score']*len(metricsCV) 
sns.set_context('talk')
sns.set_style("darkgrid")
plt.figure(figsize=(8,8))
sns.boxplot(metricsCV)
plt.title("Modelo entrenado con 5 folds")
```

```python
svm_poly_mejor_performance = load('modelos/svm_poly_mejor_performance.joblib')
```

```python
y_pred= svm_poly_mejor_performance.predict(x_test)
print(classification_report(y_test,y_pred))
print('F1-Score: {}'.format(f1_score(y_test, y_pred, average='binary'))) 
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm, cmap='Blues',annot=True,fmt='g')
plt.xlabel('Predicted')
plt.ylabel('True')
```

Obtuvimos resultados bastante buenos (f1_score = 0,749)


Como conclusion del kernel polinomico podemos decir que es relativamente bueno ya que se obtienen buenos valores de f1_socre (). Sin embargo consideramos que apelar a un Kernerl con esta complejidad para obtener resultados muy parecidos al lineal no es rentable. 
#### Mantenemos al kernel lineal como el mejor hasta el momento


### Kernel radial


**ATENCION: 5 MIN con core i5 + 16Gb RAM (sin archivos de joblib)**

```python
if not exists('modelos/clf_radial_no_optimizado.joblib'):

    #Creo un clasificador con kernel radial y lo entreno
    clf_radial_no_optimizado = SVC(kernel='rbf', random_state=9)
    clf_radial_no_optimizado.fit(x_train, y_train)

    dump(clf_radial_no_optimizado, 'modelos/clf_radial_no_optimizado.joblib')

else:
    clf_radial_no_optimizado = load('modelos/clf_poly_no_optimizado.joblib')

#Hago la predicción y calculo las métricas
y_pred_rad=clf_radial_no_optimizado.predict(x_test)
metricas(y_pred_rad,y_test)
```

Obtenemos resultados bastante malos de f1_score (0,6)


#### Intentamos mejorar los parametros del SVM con kernel raidal haciendo una busquedo con GridSearch.


**ATENCION: 2 MIN con core i5 + 16Gb RAM (sin archivos de joblib)**

```python
if not exists('modelos/gridcv_svm_kernel_radial.joblib'):

    parametros = {'C': [1, 9, 10, 100],
    'gamma': [0, 10, 100],
    'class_weight':['balanced', None]}

    #vario hiperparaemtros en kernel polinomico
    clf_rbf = SVC(kernel ="rbf", cache_size=900, max_iter=100, random_state=9)
    #SVC(kernel='poly', C=5, degree=10, gamma=10, coef0=10)
    #clf.fit(x_train, y_train)
    combinaciones=10

    scorer_fn = make_scorer(sk.metrics.f1_score)

    #svmReg = svm.SVR(cache_size=900, max_iter=1000) # El cache es para agilizar el procesado
    # Ademas se limita a 1000 max iter dado que de otra forma el procesamiento tarda demasiado.


    gridcv_svm_kernel_radial = GridSearchCV(estimator=clf_rbf,
                                param_grid= parametros,
                                #param_distributions = parametros,
                                scoring=scorer_fn,
                                #cv=kfoldcv,
                                #n_iter=combinaciones
                                ) 

    #lo entreno sobre los datos
    gridcv_svm_kernel_radial.fit(x_train, y_train)

    dump(gridcv_svm_kernel_radial, 'modelos/gridcv_svm_kernel_radial.joblib')

else:
    gridcv_svm_kernel_radial = load('modelos/gridcv_svm_kernel.joblib')

#con o sin el archivo joblib...
print("Mostramos los mejores resultados: ")
print(gridcv_svm_kernel_radial.best_params_)
print()
print("Mostramos el mejor resultado obtenido de busqueda aleatoria: ")
print("f1_score: ",gridcv_svm_kernel_radial.best_score_)
```

Se oibtuvieron resultados de f1_score apenas mejores q en el caso anterior (0.67). No representan una mejora respecto al SVM creado por default.


A continuacion, creamos el SVM con ""mejores"" parametros y realizamos la prediccion. <br /> 

**ATENCION: 15 MIN con core i5 + 16Gb RAM (sin archivos de joblib)**<br />


```python
if not exists('modelos/svm_kernel_radial_mejor.joblib'):

    mejor_svm_rbf = SVC().set_params(**gridcv_svm_kernel_radial.best_params_)
    mejor_svm_rbf.fit(x_train, y_train)
    dump(svm_lineal_mejor_performance, 'modelos/svm_kernel_radial_mejor.joblib')

else:
    mejor_svm_rbf = load('modelos/svm_kernel_radial_mejor.joblib')


```

**2 minutos toma lo que sigue**

```python
y_pred_rad_mejorado=mejor_svm_rbf.predict(x_test)
metricas(y_pred_rad_mejorado,y_test)
```

Podemos ver que las metricas mejoraron sustancialmente pero no supera el f1_score de 0,75 alcanzado con el kernels lineal/ polinomico. Ademas existe cierto sesgo ya que el modelo tiene mayor capacidad para predecir la clase 1 sobre la clase 0.


A pesar de ello...


##### Hacemos Cross validation con el svm radial con mejores paraemetros encontrado.
**El siguiente codigo se encuentra comentado ya que no pudimos terminar de correrlo debido al tiempo que toma (minimo 35 minutos con core i5 y 16Gb RAM)**


```python
# if not exists('moswloa/svm_kernel_radial_mejor.joblib'):

#     folds=5

#     kfoldcv = StratifiedKFold(n_splits=folds)
#     scorer_fn = make_scorer(sk.metrics.f1_score)
#     resultados = cross_validate(mejor_svm_rbf,x_train, y_train, cv=kfoldcv,scoring=scorer_fn,return_estimator=True)

#     metricsCV=resultados['test_score']

#     clf_poly_no_optimizado=resultados['estimator'][np.where(metricsCV==max(metricsCV))[0][0]]
    

#     metricsCV
```

Al no poder terminar de correr cross validation no podemos afirmar que el modelo con Kernel radial sea generalizable. En caso de que lo fuere, posee un score bajo (f1_score =0,73) respecto a los otros kernels y por si fuera poco se encuentra sesgado como se dijo mas arriba.<br />
**Se descarta para su utilizacion en el ensmable**.


## Conclusion SVM
Con lo visto en clase, las pruebas hechas durante la realizacion del tp, lo googleado, lo Chatgetepeado Y lo BARDeado (AI de google en prueba) se concluye que al trabjar con una cantidad tan grande de datos de testeo lo mejor es utilzar un Kernel lineal(ver primera seccion de SVM). Este sera el utilizado para el ensable en su correspondiente seccion.


# Random Forest 


Para empezar con el random forest, vamos a crear un modelo con valores totalmente aleatorios.
Usando https://www.random.org/, con valor maximo 50 y valor minimo 1, obtuvimos:
- 33
- 15
- 40
- 36

(Criterion fue dejado como entropy)

```python
#Creamos un clasificador con hiperparámetros arbitrarios
rfc = RandomForestClassifier(max_features='auto', 
                             n_jobs=JOBS,
                             criterion="entropy", 
                             random_state=SEED, 
                             min_samples_leaf=15,
                             min_samples_split=40,
                             n_estimators=36 )
#Entrenamos el modelo
model = rfc.fit(X = x_train, y = y_train)
```

```python
#Nos guardamos este modelo para poder cargarlo en todas las corridas posteriores
#dump(model, 'modelos/randomForest.joblib')
model = load('modelos/randomForest.joblib')
```

```python
#Realizamos una predicción sobre el set de test
y_pred = model.predict(x_test)
#Valores Predichos
y_pred
```

La matriz de confusion es la siguiente:

```python
#Creamos la matriz de confusión
tabla=confusion_matrix(y_test, y_pred)

#Grafico la matriz de confusión
sns.heatmap(tabla,cmap='GnBu',annot=True,fmt='g')
plt.xlabel('Predicted')
plt.ylabel('True')
```

Vemos que obtuvimos una alta cantidad de falsos positivos


Sin ningun tipo de optimizacion obtuvimos los siguientes scores 

```python
accuracy=accuracy_score(y_test,y_pred)
recall=recall_score(y_test,y_pred)
f1=f1_score(y_test,y_pred)

print("Accuracy: "+str(accuracy))
print("Recall: "+str(recall))
print("f1 score: "+str(f1))
```

Ademas, segun este modelo; las 10 columnas mas relevantes son:

```python
p = sorted(list(zip(hotelsdf_modelo_x.columns.to_list(), model.feature_importances_)), key=lambda x: -x[1])
for i in range(10):
    print(p[i])
```

Vamos a hacer un submission de nuestro random forest aleatorio:

```python
y_pred = model.predict(hotelsdf_pruebas)
```

```python
df_submission = pd.DataFrame({'id': hotelsdf_pruebasOriginal['id'], 'is_canceled': y_pred})
df_submission.head()
```

```python
df_submission.to_csv('submissions/random_forest_random.csv', index=False)
```

Este modelo tuvo el siguiente resultado en Kaggle
![randoForest](informe/images/randomForest_random.png)


## Cross validation


Ahora vamos a buscar mejorar esos resultados; optimizando los hiperparametros usando validacion cruzada

```python
if exists('modelos/randomForestCV.joblib') == False:
    rf_cv = RandomForestClassifier(oob_score=False, random_state=9, n_jobs=JOBS)
    #rf_cv = RandomForestClassifier(max_features='sqrt', oob_score=True, random_state=1, n_jobs=-1)
    param_grid = { "criterion" : ["gini", "entropy"], 
                   "min_samples_leaf" : [1, 5, 10, 15, 20], #Vamos a hacer muchas combinaciones ya que solo vamos
                   "min_samples_split" : [2, 8, 16, 32, 64],#a correr este modelo 1 sola vez; ya que lo vamos a 
                   "n_estimators": [10, 20, 30, 40, 50, 60, 70] } #guardar   

    #Probamos entrenando sólo con 1 métrica
    gs = GridSearchCV(estimator=rf_cv, param_grid=param_grid, scoring="f1", cv=5, n_jobs=JOBS) #Optimizamos f1_score
    gs_fit = gs.fit(X = x_train, y = y_train)
    dump(gs_fit, 'modelos/randomForestCV.joblib')
```

```python
gs_fit = load('modelos/randomForestCV.joblib')
```

```python
gs_fit.best_params_
```

```python
#Obtenemos el mejor modelo
rf_cv_best=gs_fit.best_estimator_

#Predicción
y_pred_rf_cv_best = rf_cv_best.predict(x_test)
y_pred_rf_cv_best
```

Con esta validacion, obtenemos la siguiente matriz de confusion

```python
#Creo matriz de confusión
tabla=confusion_matrix(y_test,y_pred_rf_cv_best)

#Grafico matriz de confusión
sns.heatmap(tabla, cmap='Blues',annot=True,fmt='g')
plt.xlabel('Predicted')
plt.ylabel('True')

#Reporte
print(classification_report(y_test,y_pred_rf_cv_best))
```

A priori, se ven menos falsos positivos

```python
#Evaluo la performance en el conjunto de evaluación
accuracyCV=accuracy_score(y_test,y_pred_rf_cv_best)
recallCV=recall_score(y_test,y_pred_rf_cv_best)
f1CV=f1_score(y_test,y_pred_rf_cv_best)

print("Accuracy: "+str(accuracyCV))
print("Recall: "+str(recallCV))
print("f1 score: "+str(f1CV))
```

Con este nuevo modelo, obtuvimos las siguientes mejoras:

```python
print(str("Accuracy = ") + str(accuracyCV - accuracy)[3:4] + "%")
print(str("Recall = ") + str(recallCV - recall)[3:4] + "%")
print(str("f1 score = ") + str(f1CV - f1)[3:4] + "%")
```

Vemos que optimizando por el f1 score, obtuvimos una mejora del 2% nada mas; pero una mejora del 4% en recall


Vamos a realizar una submission de este modelo

```python
y_pred_model_rfcv = rf_cv_best.predict(hotelsdf_pruebas)
```

```python
df_submission = pd.DataFrame({'id': hotelsdf_pruebasOriginal['id'], 'is_canceled': y_pred_model_rfcv})
df_submission.head()
```

```python
df_submission.to_csv('submissions/random_forestCV.csv', index=False)
```

Este modelo tuvo el siguiente resultado en Kaggle
![randoForestCVMM](informe/images/randomForestCV.png)


## Cross validation multiples metricas


Ahora vamos a realizar un random forest pero tratando de optimizar distintas metricas a la vez. \
Luego vamos a elegir la que optimice mejor todas las metricas

```python
#Metricas que vamos a analizar:
metricas=['accuracy','f1','roc_auc' ,'recall', 'precision'] 

if exists('modelos/randomForestCVMM.joblib') == False:
    rf_cv = RandomForestClassifier(oob_score=False, random_state=1, n_jobs=JOBS)

    param_grid = { "criterion" : ["gini", "entropy"], 
                    "min_samples_leaf" : [1, 5, 10, 15, 20], #Vamos a hacer muchas combinaciones ya que solo vamos
                    "min_samples_split" : [2, 8, 16, 32, 64],#a correr este modelo 1 sola vez; ya que lo vamos a 
                    "n_estimators": [10, 20, 30, 40, 50, 60, 70] } #guardar   


    gs_multimetrica = GridSearchCV(estimator=rf_cv, 
                                   param_grid=param_grid, 
                                   scoring=metricas, 
                                   refit=False, 
                                   cv=5, 
                                   n_jobs=JOBS)
    #Entrenamiento
    gs_multimetrica_fit = gs_multimetrica.fit(X = x_train, y = y_train)
    dump(gs_multimetrica_fit, 'modelos/randomForestCVMM.joblib')
```

```python
gs_multimetrica_fit = load('modelos/randomForestCVMM.joblib')
```

Vamos a graficar todos los resultados de las metricas que medimos

```python
labels=[ key for key in gs_multimetrica_fit.cv_results_.keys() if("mean_test" in key)]

for k in labels:
    plt.plot(gs_multimetrica_fit.cv_results_[k],linestyle='--' , linewidth=0.8,marker='o',markersize=2)     
    x_linea=np.argmax(gs_multimetrica_fit.cv_results_[k])
    plt.axvline(x_linea,linestyle='--' ,linewidth=0.8,color='grey')
        
plt.xlabel("modelo", fontsize=10)
plt.ylabel("métrica", fontsize=10)
plt.legend(labels)
plt.show()
```

Del grafico se observa que hay un modelo que parece optimizar todas las metricas. A ojo parece ser el ~180\
Vamos a corroborarlo:

```python
for metrica in metricas:
    params_analizar=gs_multimetrica_fit.cv_results_['params'][np.argmax(gs_multimetrica_fit.cv_results_['mean_test_' + metrica])]
    print(
"Metrica " + metrica + ": " + str(params_analizar))
```

Vemos que son todos muy similares pero con cierta variazon. Vamos a elegir a f1 score para tener cierto tipo de balance

```python
params_elegidos=gs_multimetrica_fit.cv_results_['params'][np.argmax(gs_multimetrica_fit.cv_results_['mean_test_f1'])]

#Creamos un clasificador RF
rfc_multimetrica = RandomForestClassifier(criterion= params_elegidos['criterion'], 
                                          min_samples_leaf= params_elegidos['min_samples_leaf'], 
                                          min_samples_split= params_elegidos['min_samples_split'], 
                                          n_estimators=params_elegidos['n_estimators'], 
                                          oob_score=True, random_state=2, n_jobs=JOBS)
#Entrenamos un modelo
model_rfc_multimetrica = rfc_multimetrica.fit(X = x_train, y = y_train)

#Hacemos una predicción con el dataset de train
y_pred_model_rfc_multimetrica = model_rfc_multimetrica.predict(x_test)
```

Vamos a visualizar uno de los estimadores de este random forest resultante:

```python
plt.figure(figsize=(12,12))

tree_plot=tree.plot_tree(rfc_multimetrica.estimators_[56],
                         max_depth=2,
                         feature_names=hotelsdf_modelo_x.columns.to_list(),
                         filled=True,
                         rounded=True,
                         class_names=True)

plt.show(tree_plot)
```

Vision completa:

```python
#plt.figure(figsize=(100,100))

#tree_plot_completo=tree.plot_tree(rfc_multimetrica.estimators_[56],
#                                 feature_names=hotelsdf_modelo_x.columns.to_list(),
#                                 filled=True,
#                                 rounded=True,)
#                                 #class_names=['Not Survived','Survived']) #model.classes_
#plt.show(tree_plot_completo)
```

Calculamos la matriz de confusion

```python
#Matriz de Confusión
cm = confusion_matrix(y_test,y_pred_model_rfc_multimetrica)
sns.heatmap(cm, cmap='Blues',annot=True,fmt='g')
plt.xlabel('Predicted')
plt.ylabel('True')

#Reporte
print(classification_report(y_test,y_pred_model_rfc_multimetrica))

```

A pesar de todas nuestra busqueda, no se observan cambios significativos

```python
#Evaluo la performance en el conjunto de evaluación
accuracyCVMM=accuracy_score(y_test,y_pred_model_rfc_multimetrica)
recallCVMM=recall_score(y_test,y_pred_model_rfc_multimetrica)
f1CVMM=f1_score(y_test,y_pred_model_rfc_multimetrica)

print("Accuracy: "+str(accuracyCVMM))
print("Recall: "+str(recallCVMM))
print("f1 score: "+str(f1CVMM))
```

Sorprendentemente, no tuvimos mejoras significativas comparado con el modelo que no consideraba todas las metricas

```python
print(str("Accuracy = ") + str(accuracyCVMM - accuracyCV)[3:4] + "%")
print(str("Recall = ") + str(recallCVMM - recallCV)[3:4] + "%")
print(str("f1 score = ") + str(f1CVMM - f1CV)[3:4] + "%")
```

Realizamos la prediccion sobre el dataset de testeo

```python
y_pred_model_rfc_multimetrica = model_rfc_multimetrica.predict(hotelsdf_pruebas)
```

```python
df_submission = pd.DataFrame({'id': hotelsdf_pruebasOriginal['id'], 'is_canceled': y_pred_model_rfc_multimetrica})
df_submission.head()
```

```python
df_submission.to_csv('submissions/random_forestCVMM.csv', index=False)
```

Este modelo tuvo el siguiente resultado en Kaggle
![randoForestCVMM](informe/images/randomForestCVMM.png)


Vemos que a pesar de todas nuestras mejoras, solo obtuvimos una mejora del 0.2%


# XGBoost 

## Modelo base

Generamos un modelo XGBoost base, con los hiperparametros por defecto, de manera que se pueda realizar una comparacion posterior a entrenar un modelo con sus hiperparametros optimmizados

```python

if not exists('modelos/xgb_base.joblib'):
    xgb_base = xgb.XGBClassifier(random_state=9, n_estimators=100) 
    xgb_base.fit(x_train, y_train)
    dump(xgb_base, 'xgb_base.joblib')
else:
    xgb_base = load('modelos/xgb_base.joblib')
```

Vemos el comportamiento del modelo base y mostramos las metricas obtenidas en el procesp

```python
y_pred=xgb_base.predict(x_test)

print(classification_report(y_test,y_pred))
print('F1-Score: {}'.format(f1_score(y_test, y_pred, average='binary'))) 
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm, cmap='Blues',annot=True,fmt='g')
plt.xlabel('Predecido')
plt.ylabel('Verdadero')
```

Realizamos una prediccion para kaggle y almacenamos el modelo generado en una primera instancia 

```python
y_pred = xgb_base.predict(hotelsdf_pruebas)
y_pred
df_submission = pd.DataFrame({'id': hotelsdf_pruebasOriginal['id'], 'is_canceled': y_pred})

if not exists('submissions/xgb_base.csv'):
    df_submission.to_csv('xgb_base.csv', index=False)
```

Destacamos que este modelo sin recibir ninguna optimización tiene la presición mas alta de todos los modelos entrenados

## Busqueda de hiperparametros

Realizamos una busqueda para encontrar los mejores hiperparametros del XGBoost y a su vez optimizar el modelo *Puede tomar tiempo, alrededor de 50 min*

```python
if not exists('modelos/RCV_xgb.joblib'):

    estimadores = [90, 100, 110, 150]
    profundidad_max = [7, 8, 9, 10, 15]
    learning_rate = [0.01, 0.05, 0.1, 0.2]

    params = {
        'max_depth': profundidad_max,
        'n_estimators': estimadores,
        'learning_rate': learning_rate,
            }


    xgb_entrenamiento = xgb.XGBClassifier()
    combinaciones = 10
    k_folds = 10
    metrica_fn = make_scorer(sk.metrics.f1_score)

    parametros = RandomizedSearchCV(
                estimator=xgb_entrenamiento, 
                param_distributions = params, 
                cv=k_folds, 
                scoring=metrica_fn, 
                n_iter=combinaciones, 
                random_state=9)

    parametros.fit(x_train, y_train)
    parametros.cv_results_['mean_test_score']

else:
    parametros = load('modelos/RCV_xgb.joblib')
```

Mostramos las metricas y los mejores hiperparametros conseguidos en el analisis 

```python
print("Mostramos los mejores resultados: ")
print(parametros.best_params_)
print()
print("Mostramos el mejor resultado obtenido de busqueda aleatoria: ")
print("f1_score = ",parametros.best_score_)
```

Entrenamos el modelo con sus hiperparametros

```python
if not exists('modelos/xgb_optimizado.joblib'):
    xgb_optimizado = xgb.XGBClassifier(**parametros.best_params_)
    xgb_optimizado.fit(x_train, y_train)
else:
    xgb_optimizado = load('modelos/xgb_optimizado.joblib')
```

Realizamos la validación cruzada del modelo para verificar que no caiga en overfitting o underfitting 

```python
if not exists('modelos/xgb_optimizado.joblib'):
    kfoldcv =StratifiedKFold(n_splits=k_folds) 
    resultados = cross_validate(xgb_optimizado,x_train, y_train, cv=kfoldcv,scoring=metrica_fn,return_estimator=True)
    metricsCV = resultados['test_score']
    xgb_optimizado = resultados['estimator'][np.where(metricsCV==max(metricsCV))[0][0]]
    dump(xgb_optimizado, 'xgb_optimizado.joblib')

else:
    xgb_optimizado = load('modelos/xgb_optimizado.joblib')
```

Observamos el comportamiento del modelo a lo largo de la validacón cruzada 

```python
metric_labelsCV = ['F1 Score']*len(metricsCV) 
sns.set_context('talk')
sns.set_style("darkgrid")
plt.figure()
sns.boxplot(metricsCV)
plt.title("Modelo entrenado con 10 folds")
```

Observamos la matriz de confusión del modelo y concluimos que es el modelo con el mejor F1 score que se ha podido entrenar en el analisis sobre las reservas

```python
y_pred= xgb_optimizado.predict(x_test)
print(classification_report(y_test,y_pred))
print('F1-Score: {}'.format(f1_score(y_test, y_pred, average='binary'))) 
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm, cmap='Blues',annot=True,fmt='g')
plt.xlabel('predecido')
plt.ylabel('verdadero')
```

Realizamos la predicción de kaggle 

```python
y_pred = xgb_optimizado.predict(hotelsdf_pruebas)
y_pred
if not exists('submissions/xgb_optimizado.joblib'):
    df_submission = pd.DataFrame({'id': hotelsdf_pruebasOriginal['id'], 'is_canceled': y_pred})
    df_submission.to_csv('xgb_optimizado.csv', index=False)
```

El ensamble XGBoost representa el modelo más preciso de todos los modelos entrenados hasta esta sección del analisis

# Modelo Voting


# Modelo Stacking 


# Conclusiones 


1. KNN: 
2. SVM:
3. Random Forest:
4. XGBoost:
5. Voting:
6. Stacking:
7. General
