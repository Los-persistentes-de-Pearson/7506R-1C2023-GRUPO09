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
from dict_paises import COUNTRY_ALPHA3_TO_COUNTRY_ALPHA2, COUNTRY_ALPHA2_TO_CONTINENT

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
hotelsdfTesteoOriginal = pd.read_csv("./hotels_test.csv")
hotelsdfTesteo = hotelsdfTesteoOriginal.copy()
```

# Arbol de decisiones sin optimizacion


Vamos a comenzar creando un arbol de decisiones que tenga en cuenta todas las columnas. \
Luego, vamos a realizar una optimizacion y vamos a optimizar dicho arbol para luego comparar resultados.


## Cargamos nuestro dataframe del checkpoint pasado

Vamos a crear una copia de nuestro dataframe para la creacion del arbol

```python
hotelsdfCheckpoint1 = pd.read_csv("./dataframeCheckpoint1.csv")
hotelsdfArbol = hotelsdfCheckpoint1.copy()
print("El data frame esta compuesto por "f"{hotelsdfArbol.shape[0]}"" filas y "f"{hotelsdfArbol.shape[1]}"" columnas")
```

Un vistazo básico a la información contenida en el dataframe:

```python
pd.concat([hotelsdfArbol.head(2), hotelsdfArbol.sample(5), hotelsdfArbol.tail(2)])
```

Vemos que tenemos una columa extra "Unnamed: 0". Esta hace referencia la columna de origen del registro. Procedemos a borrarla

```python
hotelsdfArbol.drop("Unnamed: 0", axis=1, inplace=True)
hotelsdfArbol.reset_index(drop=True)
print()
```

## Transformacion de las columnas para la creacion del arbol


Para poder usar el arbol de sklearn, tenemos que transformar todas nuestras columnas no numericas a valores numericos. \
Dichas columnas son las siguientes:

```python
valoresAConvertir = hotelsdfArbol.dtypes[(hotelsdfArbol.dtypes !='int64') & (hotelsdfArbol.dtypes !='float64')].index
valoresAConvertir = valoresAConvertir.to_list()
valoresAConvertir
```

Sin embargo, no todas estas columnas nos van a servir para nuestro analisis.



### Booking ID


Vamos a empezar removiendo booking\_id visto en como no la necesitamos para analisis

```python
hotelsdfArbol.drop("booking_id", axis=1, inplace=True)
hotelsdfArbol.reset_index(drop=True)
valoresAConvertir.remove('booking_id')
```

### Reservation Status & Reservation status date


Reservation Status nos dice el estado de la reservacion, si fue cancelada o no y reservation status date nos marca la fecha en la que cambio el estado. 
Estas dos columnas nos son redundantes

```python
hotelsdfArbol.drop("reservation_status", axis=1, inplace=True)
hotelsdfArbol.reset_index(drop=True)
valoresAConvertir.remove('reservation_status')
```

```python
hotelsdfArbol.drop("reservation_status_date", axis=1, inplace=True)
hotelsdfArbol.reset_index(drop=True)
valoresAConvertir.remove('reservation_status_date')
```

### Country
Country toma una amplia cantidad de valores como vimos en el analisis univariado. Asique decidimos agrupar los paises por continente para poder usar la variable

```python
hotelsdfArbol["continente"] = hotelsdfArbol["country"].replace(COUNTRY_ALPHA3_TO_COUNTRY_ALPHA2)
hotelsdfArbol["continente"] = hotelsdfArbol["continente"].replace(COUNTRY_ALPHA2_TO_CONTINENT)
```

```python
country = hotelsdfArbol['country'].unique().tolist()
print(country) 
```

```python
country = hotelsdfArbol['continente'].unique().tolist()
print(country) 
```

Viendo estos resultados vemos que hay dos outliers que no logramos identificar en el analisis univariado.


"ATA" refiere al **continente** de Antartida. Al ser un valor tan fuera de lo comun y tener una sola ocurrencia decidimos eliminarlo del dataframe

```python
hotelsdfArbol.drop((hotelsdfArbol[hotelsdfArbol["country"] == "ATA"].index.values),inplace=True)
hotelsdfArbol.reset_index(drop=True)
print()
```

"UMI" hace referenca a unas islas cerca de Hawaii. Al ser un unico caso y tener una poblacion de 300 habitantes, decidimos considerarlo como Estados Unidos, es decir America del Norte

Fuentes:
- https://www.iso.org/obp/ui/#iso:code:3166:UM
- https://en.wikipedia.org/wiki/United_States_Minor_Outlying_Islands

```python
hotelsdfArbol.loc[hotelsdfArbol['country'] == "UMI", 'country'] = 'North America'
hotelsdfArbol.loc[hotelsdfArbol['continente'] == "UMI", 'continente'] = 'North America'
```

Con estos nuevos cambios, la columna continente toma los siguientes valores

```python
continente = hotelsdfArbol['continente'].unique().tolist()
print(continente) 
```

Procedemos a dropear la columna de country

```python
hotelsdfArbol=hotelsdfArbol.drop(['country'], axis='columns', inplace=False)
valoresAConvertir.remove('country')
valoresAConvertir.append('continente')
hotelsdfArbol.reset_index(drop=True)
```

```python
valoresAConvertir
```

### One hot encoding


Vamos a transformar dichas variables categoricas con la tecnica de one hot encoding. \
Esto va a crear una serie de nuevas columnas con todos los posibles de la variable categorica. En cada columna va a haber un 1 o un 0 para indicar el valor del registro de esa variable. \
Una de las columnas (en este caso la primera) es eliminada ya que, si todas las otras columnas son falsas, significa que la variable toma el valor de la columna eliminada. \
Esto lo podemos hacer gracias a que eliminamos todos nuestros valores faltantes en las secciones anteriores.

```python
hotelsdfArbol = pd.get_dummies(hotelsdfArbol, columns=valoresAConvertir, drop_first=True)
```

Vamos a observar como nos quedo el dataframe despues del one hot encoding

```python
hotelsdfArbol.head()
```

Observamos que hay una **gran** cantidad de columnas


### Aplicamos mismas modificaciones al dataset de testeo


Ahora vamos a aplicar las mismas modificaciones y encodings al dataframe de testeo para poder aplicarle el modelo


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

hotelsdfTesteo.rename(columns = nuevas_columnas, inplace = True)
```

Antes de nada, vamos a procesar todos los datos faltantes del dataframe.


#### Dias Totales


Anadimos la columna que creamos en el checkpoint 1

```python
hotelsdfTesteo["dias_totales"] = hotelsdfTesteo["week_nights_num"] + hotelsdfTesteo["weekend_nights_num"]
```

#### Datos faltantes

```python
hotelsdfTesteo.isnull().sum()
```

```python
print("Vemos que 'company id' tiene un " + str( (hotelsdfTesteo["company_id"].isnull().sum() * 100) / len(hotelsdfTesteo)  ) + "% de datos faltantes.")
print("Por esto decidimos eliminar la columna (tanto en el dataset de testeo como en el de entrenamiento)")
```

```python
hotelsdfTesteo.drop("company_id", axis=1, inplace=True)
hotelsdfTesteo.reset_index(drop=True)
```

### Valores a convertir


Siempre posible, vamos a aplicar el mismo criterio que arriba

```python
valoresAConvertirTesteo = hotelsdfTesteo.dtypes[(hotelsdfTesteo.dtypes !='int64') & (hotelsdfTesteo.dtypes !='float64')].index
valoresAConvertirTesteo = valoresAConvertirTesteo.to_list()
valoresAConvertirTesteo
```

#### Booking ID

```python
hotelsdfTesteo.drop("booking_id", axis=1, inplace=True)
hotelsdfTesteo.reset_index(drop=True)
valoresAConvertirTesteo.remove('booking_id')
```

#### Agent ID


Tomamos el mismo criterio que el checkpoint 1. Transformamos a 0

```python
hotelsdfTesteo.loc[hotelsdfTesteo['agent_id'].isnull(), 'agent_id'] = 0
```

#### Reservation Status & Reservation status date



Dropeamos estas columnas debido a que no nos dan ninguna informacion adicional

```python
hotelsdfTesteo.drop("reservation_status_date", axis=1, inplace=True)
hotelsdfTesteo.reset_index(drop=True)
valoresAConvertirTesteo.remove('reservation_status_date')
```

#### Country y Continents


Para los valores faltantes de country tomamos el mismo criterio del checkpoint 1. Los convertimos en portugal

```python
hotelsdfTesteo.loc[hotelsdfTesteo['country'].isnull(), 'country'] = 'PRT'
```

```python
hotelsdfTesteo["continente"] = hotelsdfTesteo["country"].replace(COUNTRY_ALPHA3_TO_COUNTRY_ALPHA2)
hotelsdfTesteo["continente"] = hotelsdfTesteo["continente"].replace(COUNTRY_ALPHA2_TO_CONTINENT)
```

```python
country = hotelsdfTesteo['country'].unique().tolist()
valoresAConvertirTesteo.append("continente")
print(country) 
```

```python
continentes = hotelsdfTesteo['continente'].unique().tolist()
print(continentes) 
```

Tal como ocurrio con el dataset de Train, observamos que hay algunos continente (y por tanto sus paises y registros asociados) que parecen ser outliers.
Los estudiamos

```python
hotelsdfTesteo[ hotelsdfTesteo['continente'] =="ATA"]
```

Hay un registro correspondiente a "Antartida". como no podemos dropearlo, le ponemos de continente "north america".\
Le asignamos el valor de America del norte debido a que estados unidos es el pais con mas bases en la antartica

```python
hotelsdfTesteo.loc[hotelsdfTesteo['continente'] == "ATA", 'continente'] = "North America"
```

```python
hotelsdfTesteo[ hotelsdfTesteo['continente'] =="ATF"]
```

"ATF", que es la sigla de Fr. So. Ant. Tr (French southern and antartic lands).
Ponemos su contienente en Europa. 

```python
hotelsdfTesteo.loc[hotelsdfTesteo['continente'] == "ATF", 'continente'] = "Europe"
```

```python
hotelsdfTesteo[hotelsdfTesteo['continente'] =="ATF"]
```

#### Analisis de valores faltantes de continente

```python
hotelsdfTesteo[hotelsdfTesteo['continente'].isna()]
```

Vemos que hay una serie de registros que no tienen el dato del pais. Sin embargo, no son muchos. Debido a esto, vamos a asignarle estos registros el valor de aquel contiente que tenga la mayor cantidad de registros

```python
sns.countplot(data = hotelsdfTesteo, x = 'continente', palette= 'Set2')
plt.title('Cantidad de registros por continente')
plt.xlabel('Continente')
plt.ylabel('Cantidad de registros')
```

Vemos que el continente con mayor cantidad de registros es europa, asique lo asignamos a ese valor

```python
hotelsdfTesteo.loc[hotelsdfTesteo['continente'].isnull(), 'country'] = 'Europe'
```

Miro q se hayan cambiado bien todos los continentes y no haya valores raros

```python
continentes = hotelsdfTesteo['continente'].unique().tolist()
print(continentes)
```

Como hicimos con el dataset de train, y ya habiendo procesado la columna continente, dropeamos la columna country

```python
hotelsdfTesteo=hotelsdfTesteo.drop(['country'], axis='columns', inplace=False)
hotelsdfTesteo.reset_index(drop=True)
valoresAConvertirTesteo.remove('country')
```

#### previous bookings not cancelled


Al igual q en el train, dropeamos esta col

```python
hotelsdfTesteo=hotelsdfTesteo.drop(['previous_bookings_not_canceled_num'], axis='columns', inplace=False)
hotelsdfTesteo.reset_index(drop=True)
```

```python
hotelsdfTesteo.isnull().sum()
```

### One hot encoding del testeo

De la misma manera al dataset de pruebas aplicamos one hot encoding sobre las columnas de variables cualitativas

```python
hotelsdfTesteo = pd.get_dummies(hotelsdfTesteo, columns=valoresAConvertirTesteo, drop_first=True)
hotelsdfTesteo.head()
```

### Corroboracion de columnas


Despues de todas estas transformaciones vamos a corrobar que los dataframes tengan la misma cantidad de columnas.

```python
set_test = set(hotelsdfTesteo.columns)
set_arbol = set(hotelsdfArbol.columns)

missing = list(sorted(set_test - set_arbol))
added = list(sorted(set_arbol - set_test))

print('Faltan en arbol:', missing)
print('Sobran en arbol:', added)
```

Vemos que en el dataframe del arbol nos sobra la columna "is canceled", cosa que hace sentido ya que esa es la columna con la que vamos a entrenar al dataset. Sin embargo, vemos que tambien hay 3 columnas que faltan en el dataset de arbol. 

Vamos a reasignar los valores de las columnas de test para que coincidan.

El siguiente codigo nos calcula cuantas personas tiene cada tipo de cuarto

```python
cantDeCuartos = {}
cantidadDeCasosSumados = 0

cantDeCuartos["A"] = 0 #Arrancamos asignado 0 a los cuartos de A. Estos fueron removidos por el one hot. Lo vamos a calcular al final.
for letra in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']:
    tipoDeCuarto = 'reserved_room_type_' + letra
    cantidadDeCasosSumados += 1
    if tipoDeCuarto not in hotelsdfTesteo.columns:
        continue
    hotelsdfTesteo[tipoDeCuarto]
    resultado = hotelsdfTesteo[hotelsdfTesteo[tipoDeCuarto] == 1][tipoDeCuarto].sum()
    cantDeCuartos[letra] = resultado

cuartosA = len(hotelsdfTesteo) - cantidadDeCasosSumados
cantDeCuartos["A"] = cuartosA


cantDeCuartos
```

Vemos que L y P tienen una extremadamente pequena cantidad de apariciones. \
Lo vamos a anadir al roomtype A al ser el que tiene la mayor cantidad de apariciones.

Para anadirlos a la columna a, simplemente tenemos que eliminar las columnas L y P (ya que la columna A es la eliminada por el one hot)

```python
hotelsdfTesteo.drop("reserved_room_type_L", axis=1, inplace=True)
hotelsdfTesteo.drop("reserved_room_type_P", axis=1, inplace=True)
hotelsdfTesteo.reset_index(drop=True)
print()
```

Vamos a aplicar el mismo criterio a assigned room type

```python
cantDeCuartos = {}
cantidadDeCasosSumados = 0

cantDeCuartos["A"] = 0 #Arrancamos asignado 0 a los cuartos de A. Estos fueron removidos por el one hot. Lo vamos a calcular al final.
for letra in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']:
    tipoDeCuarto = 'assigned_room_type_' + letra
    cantidadDeCasosSumados += 1
    if tipoDeCuarto not in hotelsdfTesteo.columns:
        continue
    hotelsdfTesteo[tipoDeCuarto]
    resultado = hotelsdfTesteo[hotelsdfTesteo[tipoDeCuarto] == 1][tipoDeCuarto].sum()
    cantDeCuartos[letra] = resultado

cuartosA = len(hotelsdfTesteo) - cantidadDeCasosSumados
cantDeCuartos["A"] = cuartosA


cantDeCuartos
```

Aca tambien vemos que P tiene muy pocas aparciones. Asique aplicamos el mismo criterio de antes

```python
hotelsdfTesteo.drop("assigned_room_type_P", axis=1, inplace=True)
hotelsdfTesteo.reset_index(drop=True)
print()
```

Vemos ahora que nuestras columnas coinciden

```python
set_test = set(hotelsdfTesteo.columns)
set_arbol = set(hotelsdfArbol.columns)

missing = list(sorted(set_test - set_arbol))
added = list(sorted(set_arbol - set_test))

print('Faltan en arbol:', missing)
print('Sobran en arbol:', added)
```

## Entrenamiento del modelo

Se genera un dataset con los datos necesarios para predecir la cancelacion y creamos un dataset conteniendo el target, para luego, generar conjuntos de test y train

```python
hotelsdfArbol_x=hotelsdfArbol.drop(['is_canceled'], axis='columns', inplace=False)


hotelsdfArbol_y = hotelsdfArbol['is_canceled'].copy()

x_train, x_test, y_train, y_test = train_test_split(hotelsdfArbol_x,
                                                    hotelsdfArbol_y, 
                                                    test_size=0.2,  #proporcion 80/20
                                                    random_state=9) #Semilla 9, como el Equipo !!
```

Ahora ya tenemos generados nuestros conjuntos de train y test; y tenemos nuestro dataframe con los datos numericos, vamos a generar nuestro modelo

Iniciamos con una profundidad maxima arbitraria, en este caso 20 y creamos un arbol utilizando el criterio **Gini** 

Dicho modelo sera uno generado directamente tomando en cuenta todos los valores y sin generar ningun tipo de poda, para observar como se comporta un modelo sin tratar

```python
PROFUNDIDAD_MAX = 20

tree_model = tree.DecisionTreeClassifier(criterion="gini",
                                         max_depth = PROFUNDIDAD_MAX) 
model = tree_model.fit(X = x_train, y = y_train)
```

Una vez entrenado el modelo realizamos una predicción con el mismo

```python
y_pred = model.predict(x_test)
y_pred
```

```python
ds_resultados=pd.DataFrame(zip(y_test,y_pred),columns=['test','pred'])
ds_resultados
```

Estas columns representan 20% de nuestro dataframe que fue dedicado al testeo del modelo


Vamos a graficar la matriz de confusion para visualizar los resultados de nuesto modelo:

```python
tabla=confusion_matrix(y_test, y_pred)
sns.heatmap(tabla,cmap='GnBu',annot=True,fmt='g')
plt.xlabel('Predicted')
plt.ylabel('True')
```

Presentamos las reglas conseguidas en árbol no optizado:

```python
reglas = export_text(tree_model, feature_names=list(hotelsdfArbol_x.columns.tolist()))
print(reglas)
```

A continuacion vamos a graficar el arbol resultante: \
(**Advertencia**: Suele tardar unos minutos en terminar de renderizar la imagen)

```python
plt.figure(figsize=(100,100))

tree_plot_completo=tree.plot_tree(model,
                                 feature_names=hotelsdfArbol_x.columns.to_list(),
                                 filled=True,
                                 rounded=True,
                                 class_names=['Not Canceled','Is canceled']) #model.classes_
plt.show(tree_plot_completo)
```

Con la imagen se ve que el arbol resultante tiene unas dimensiones exageradas, vemos ademas que tiene una profundidad de 20 como especificamos

Vemos que en un árbol sin optimizar de profundidad 20 y sin configurar una mejora en los hiperparametros obtenemos las siguientes metricas:

```python
accuracy=accuracy_score(y_test,y_pred)
recall=recall_score(y_test,y_pred)
f1=f1_score(y_test,y_pred,)
precision=precision_score(y_test,y_pred)

print("Accuracy: "+str(accuracy))
print("Recall: "+str(recall))
print("Precision: "+str(precision))
print("f1 score: "+str(f1))
```

```python
#Realizamos una predicción sobre el set de test
y_pred = model.predict(hotelsdfTesteo)
#Valores Predichos
y_pred
```
Con este modelo, obtuvimos el siguiente resultado:

![PrimeraEntrega](informe/images/primeraPrediccion.jpg)


# Busqueda de hiperparametros, poda y validación cruzada

## Randomized Search Cross Validation

Mediante la tecnica de ramdomized search cross validations hacemos una busqueda de los mejores hiperparametros

Tomamos 15 combinaciones posibles entre los parametros existentes y buscamos la combinación que mejor optimiza la metrica F1. La decisión de mejorar la metrica F1 viene de equilibrar tanto presion y recall debido a que la naturaleza del problema no requiere la mejora de alguna en particular, lo que significa que clasifica correctamente la mayoria de los casos positivos y encuentra la maxima cantidad de ellos

Nos basamos en los siguientes parametros:

```python

combinaciones=15
limite_hojas_nodos = list(range(2, 50))
valor_poda = 0.0001 #0.0007
profundidad = list(range(0,40))
folds=10


params_grid = {'criterion':['gini','entropy'],
               'min_samples_leaf':limite_hojas_nodos,
               'min_samples_split': limite_hojas_nodos, 
               'ccp_alpha':np.linspace(0,valor_poda,combinaciones),
               'max_depth':profundidad}

kfoldcv = StratifiedKFold(n_splits=folds)

base_tree = DecisionTreeClassifier() 

scorer_fn = make_scorer(sk.metrics.f1_score)

randomcv = RandomizedSearchCV(estimator=base_tree,
                              param_distributions = params_grid,
                              scoring=scorer_fn,
                              cv=kfoldcv,
                              n_iter=combinaciones) 

randomcv.fit(x_train,y_train)
```

Mostramos los mejores hiperparametros devueltos por el arbol y el valor del f1_score

```python
print("Mostramos los mejores resultados: ")
print(randomcv.best_params_)
print()
print("Mostramos el mejor resultado obtenido de busqueda aleatoria: ")
print("f1_score = ",randomcv.best_score_)
```

Algunos valores obtenidos del algoritmo

```python
randomcv.cv_results_['mean_test_score']
```

## Predicción y Evaluación del Modelo con mejores hiperparámetros

Generamos el árbol con los hiperparametros que optimizan su eficiencia y a su vez presentamos el conjunto de valores con su peso relativo a la toma de la decisión 

```python
arbol_mejores_parametros=DecisionTreeClassifier().set_params(**randomcv.best_params_)
arbol_mejores_parametros.fit(x_train,y_train)
```

*Conjunto de reglas:*

```python
features_considerados = hotelsdfArbol_x.columns.to_list()
best_tree = randomcv.best_estimator_
feat_imps = best_tree.feature_importances_

for feat_imp,feat in sorted(zip(feat_imps,features_considerados)):
  if feat_imp>0:
    print('{}: {}'.format(feat,feat_imp))
```

Es importante destacar tres de las variables seleccionadas en la primera parte de nuestro analisis (Checkpoint 1):  lead_time, average_daily_rate y previous_cancelations_nums estan enmarcadas dentro de las diez caracteristicas que aportan màs información en la construcción del árbol

*Mostramos las reglas internas del árbol:*

```python
reglas = export_text(arbol_mejores_parametros, feature_names=list(features_considerados))
print(reglas)
```

Se puede observar una considerable simplificacion en la ramificacion de las reglas de este árbol comparado contra el primer árbol generado en el análisis 


### Grafica representativa del árbol optimizado

Mostramos los primeros cinco niveles del árbol optimazado y observamos una diferencia con el primer árbol generado en el analisis:

```python
dot_data = StringIO()
export_graphviz(arbol_mejores_parametros, out_file=dot_data,  
                 filled=True, rounded=True,
                 special_characters=True,
                 feature_names=features_considerados,
                 class_names=['no cancelo','cancelo'],
                 max_depth=5)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
```

Considerando lo antes mencionado podemos apreciar que:
1. El primer nodo particiona segun el tipo de deposito: sin rembolso, donde, la gente tiende a mantener la reserva y con rembolso donde se tiende a cancelar
2. El segundo nivel árbol toma en consideración el lead time y el numero de cambios en la reserva. Con un lead time menor a 11.5 tiene una menor cantidad de reservas canceladas, mientras, en el otro nodo clasifica cancelado si el numero de cambios en la reserva es menor que cero
3. En un tercer nivel observamos que las variables que más aportan informacion son: previous cancelation number, market segment type online TA, customer type trasient party y arrival month day 13

### Prediccion con split de train

Hacemos una primera evaluación del árbol haciendo uso de los datos de prueba y medimos su desempeño

```python
y_pred= arbol_mejores_parametros.predict(x_test)
print('F1-Score: {}'.format(f1_score(y_test, y_pred, average='binary')))
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm, cmap='Blues',annot=True,fmt='g')
plt.xlabel('Predecidos')
plt.ylabel('Verdaderos')
plt.title("Desempeño del modelo con datos de prueba")

```

*Un vistazo al primer conjunto de prediccione:*

```python
arbol_mejores_parametros.predict_proba(x_test)
```

## Entrenamiento Cross Validation

Procedemos a realizar entrenamiento del árbol mediante el metodo de la validación cruzada en 10 folds considerando que fue como se entreno previamente al árbol mas optimo. Esto buscando siempre mantener la metrica F1 en su valor más alto, como también comprobar que el árbol mantiene un desempeño esperado y detectar posibles casos de *Overfitting o Underfitting*

```python
kfoldcv =StratifiedKFold(n_splits=folds) 
scorer_fn = make_scorer(sk.metrics.f1_score)

resultados = cross_validate(arbol_mejores_parametros,x_train, y_train, cv=kfoldcv,scoring=scorer_fn,return_estimator=True)

metricsCV = resultados['test_score']

arbol_mejor_performance = resultados['estimator'][np.where(metricsCV==max(metricsCV))[0][0]]

```

```python
metric_labelsCV = ['F1 Score']*len(metricsCV) 
sns.set_context('talk')
sns.set_style("darkgrid")
plt.figure(figsize=(8,8))
sns.boxplot(metricsCV)
plt.title("Modelo entrenado con 10 folds")
```

```python
y_pred= arbol_mejor_performance.predict(x_test)
print(classification_report(y_test,y_pred))
print('F1-Score: {}'.format(f1_score(y_test, y_pred, average='binary'))) 
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm, cmap='Blues',annot=True,fmt='g')
plt.xlabel('Predicted')
plt.ylabel('True')
```

---

# Conclusión

1. Al comparar los dos modelos construidos, se observó una mejora considerable en el segundo modelo en factores como la performance y las métricas en comparación con el primero. Esto sugiere que se pudo optimizar significativamente el modelo mediante la incorporación de tecnicas como: ramdomized cross search
2. Las métricas del árbol se mejoraron mediante la optimización del F1 score, el cual fue seleccionado debido a la naturaleza del problema en el que no había un motivo particular para seleccionar una métrica específica (como recall o precision).
3. A pesar de la poda y la reducción de la libertad del árbol para extenderse, se logró una mejora significativa en su métrica F1. Esto demuestra que la poda y la limitación de la extensión del árbol no necesariamente afectan negativamente su rendimiento, sino que pueden mejorar la capacidad de generalización del modelo.
