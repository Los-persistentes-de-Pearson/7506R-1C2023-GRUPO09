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

from sklearn.model_selection import StratifiedKFold, KFold,RandomizedSearchCV, train_test_split, cross_validate
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import confusion_matrix, classification_report , f1_score, make_scorer, precision_score, recall_score, accuracy_score,f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn import tree 

#Si estamos  en colab tenemos que instalar la libreria "dtreeviz" aparte. 
if IN_COLAB == True:
    !pip install 'dtreeviz'
import dtreeviz.trees as dtreeviz

#Para eliminar los warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
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
                                                    random_state=9) #Semilla 9, como el Equipo !!
```

# KNN

Entrenamos un primer modelo de KNN usando los datos previamente tratados

## KNN sin busqueda de hiperparametros

En primera instancia entrenamos un modelo sin optimizar hiperparametros 

```python
from sklearn.neighbors import KNeighborsClassifier

knn_base = KNeighborsClassifier()
knn_base.get_params()

knn_base.fit(x_train, y_train)
y_pred = knn_base.predict(x_test)
```

```python
print('correctas: ', np.sum(y_test == y_pred))
print('total: ', len(y_test))
```

```python
accuracy_score(y_test,y_pred)
```

```python
y_pred = knn_base.predict(hotelsdf_pruebas)
y_pred
df_submission = pd.DataFrame({'id': hotelsdf_pruebasOriginal['id'], 'is_canceled': y_pred})
df_submission.to_csv('knn_base.csv', index=False)
dump(knn_base, 'knn_base.joblib')
```

# SVM 






### Librerias y Funciones

```python
import pandas as pd
import sklearn as sk
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, classification_report
```

```python
def metricas(y_pred,y_test):

  print(classification_report(y_test,y_pred))
  
  cm = confusion_matrix(y_test,y_pred)
  sns.heatmap(cm, cmap='Blues',annot=True,fmt='g')
  plt.xlabel('Predicted')
  plt.ylabel('True')
```

```python
#Construyo un modelo SVM  
#por default kernel rgb radial
svm = SVC()

#Lo entreno con los datos
svm.fit(x_train, y_train)

#Hago la prediccion y calculo las métricas
y_pred=svm.predict(x_test)
metricas(y_pred,y_test)
```

### Modifico Kernels para ver cual se adapta mejor


### Lineal

```python
#Creo un clasificador con kernel lineal y lo entreno sobre los datos escalados min-max
clf = SVC(kernel='linear', C=7)
clf.fit(x_train, y_train)

#Hago la predicción y calculo las métricas
y_pred_lin=clf.predict(x_test)
metricas(y_pred_lin,y_test)
```

### vario valor de C en kernel lineal para encontrar el mejor


NO SE PUIEDE HACERE CON RANDOMIZE SEARCH. SI O SI A MANO

```python
# #vario hiperparaemtros en kernel lineal
# clf = SVC(kernel='linear', cache_size=900, max_iter=100)
# #SVC(kernel='poly', C=5, degree=10, gamma=10, coef0=10)
# #clf.fit(x_train, y_train)

# #parametros = [ {'C': [3, 5, 9, 10, 100] }]
# parametros = [ {'C': [5] }]

# combinaciones=5

# scorer_fn = make_scorer(sk.metrics.f1_score)


# gridcv_svm = RandomizedSearchCV(estimator=clf,
#                               #param_grid= parametros,
#                               param_distributions = parametros,
#                               ) 
# #lo entreno sobre los datos
# gridcv_svm.fit(x_train, y_train)

# #Hago la predicción y calculo las métricas
# gridcv_svm.predict(x_test)
# metricas(gridcv_svm,y_test)
```

Hago un entrenamiento con cross validation para ver que el modelo sea generalizable

```python

#kfoldcv = StratifiedKFold(n_splits=folds)
#scorer_fn = make_scorer(sk.metrics.f1_score)
#resultados = cross_validate(clf,x_train, y_train, cv=kfoldcv,scoring=scorer_fn,return_estimator=True)

```

Exporto el csv para submission

```python
# df_submission = pd.DataFrame({'id': hotelsdf_pruebasOriginal['id'], 'is_canceled': y_pred})
# df_submission.to_csv('submissions/arbol_decisiones_ineficiente.csv', index=False)

```

#### Polinomico y radial


El codigo a continuacion para ambos kernels se encuentra comentado debido al gran tiempo que demora entrenar SVM's con tantos datos (no sabemos cuanto exactamente ua que nunca pudimos terminar de correrlo). Esto se debe a que en estos casos el algoritmos crea una matriz de NXN demandando mucha RAM y CPU. Con lo visto en clase, las pruebas hechas durante la realizacion del tp (ver a continuacion), lo googleado y lo Chatgetepeado se concluye que al trabjar con una cantidad tan grande de datos de testeo lo mejor es utilzar un Kernel lineal(ver seccion anterior)


### Polinomico

```python
#Creo un clasificador con kernel polinomico y lo entreno sobre los datos escalados min-max
# clf = SVC(kernel='poly')
# clf.fit(x_train, y_train)

# #Hago la predicción y calculo las métricas
# y_pred_pol=clf.predict(x_test)
# metricas(y_pred_pol,y_test)

```

#### intento mejorar hiperparametros (da error)

```python
# #vario hiperparaemtros en kernel polinomico
# clf = SVC(cache_size=900, max_iter=100)
# #SVC(kernel='poly', C=5, degree=10, gamma=10, coef0=10)
# #clf.fit(x_train, y_train)

# # parametros = [ {'C': [0,75, 9, 1, 10, 100], 
# #                 'gamma': [0.001, 0.0001], 
# #                 'kernel': ['poly']},
# #  ]


# combinaciones=5

# scorer_fn = make_scorer(sk.metrics.f1_score)

# parametros = {
#                 'kernel': ['poly'],
#                 'degree': [6 ,7, 8, 9] ,
#                 'gamma': ['scale', 'auto'],
#                 #'coef0': [10],
#                 'C': [1, 5, 8],
# }

# #svmReg = svm.SVR(cache_size=900, max_iter=1000) # El cache es para agilizar el procesado
# # Ademas se limita a 1000 max iter dado que de otra forma el procesamiento tarda demasiado.


# gridcv_svm = RandomizedSearchCV(estimator=clf,
#                               #param_grid= parametros,
#                               param_distributions = parametros
#                               #scoring=scorer_fn,
#                               #cv=kfoldcv,
#                               #n_iter=combinaciones
#                               ) 

# #lo entreno sobre los datos
# gridcv_svm.fit(x_train, y_train)

# #Hago la predicción y calculo las métricas
# gridcv_svm.predict(x_test)
# metricas(gridcv_svm,y_test)
```

### Kernel radial

```python
#Creo un clasificador con kernel radial y lo entreno sobre los datos escalados min-max
#clf = SVC(kernel='rbf', C=5, gamma=10)
# clf = SVC(kernel='rbf')
# clf.fit(x_train, y_train)

# #Hago la predicción y calculo las métricas
# y_pred_rad=clf.predict(x_test)
# metricas(y_pred_rad,y_test)
```

Intentamos mejorar los parametros del SVM con kernel raidal haciendo una busquedo con GridSearch.

```python
from sklearn.utils.fixes import loguniform

parametros = {'C': [1, 9, 10, 100],
 'gamma': [0, 10, 100],
 'kernel': ['rbf'],
 'class_weight':['balanced', None]}

#vario hiperparaemtros en kernel polinomico
clf = SVC(cache_size=900, max_iter=100)
#SVC(kernel='poly', C=5, degree=10, gamma=10, coef0=10)
#clf.fit(x_train, y_train)
combinaciones=10

scorer_fn = make_scorer(sk.metrics.f1_score)

#svmReg = svm.SVR(cache_size=900, max_iter=1000) # El cache es para agilizar el procesado
# Ademas se limita a 1000 max iter dado que de otra forma el procesamiento tarda demasiado.


gridcv_svm = GridSearchCV(estimator=clf,
                              param_grid= parametros,
                              #param_distributions = parametros,
                              scoring=scorer_fn,
                              #cv=kfoldcv,
                              #n_iter=combinaciones
                              ) 

#lo entreno sobre los datos
gridcv_svm.fit(x_train, y_train)

print("Mostramos los mejores resultados: ")
print(gridcv_svm.best_params_)
print()
print("Mostramos el mejor resultado obtenido de busqueda aleatoria: ")
print("f1_score: ",gridcv_svm.best_score_)

```

obtenemos el SVM con ""mejores"" parametros y realizamos la prediccion. Se comenta el codigo debido a que toma mucho tiempo entrtenarlo

```python
# mejor_svm = SVC().set_params(**gridcv_svm.best_params_)
# mejor_svm.fit(x_train, y_train)

# y_pred_rad_mejorado=mejor_svm.predict(x_test)
# metricas(y_pred_rad_mejorado,y_test)
```

```python
y_pred_rad=gridcv_svm.predict(x_test)
metricas(y_pred_rad,y_test)
```

El f1 score da muy mal. da altisimo el accuracy pero muy bajo el


# Random Forest 





# XGBoost 




# Modelo Voting





# Modelo Stacking 






# Conclusiones 
