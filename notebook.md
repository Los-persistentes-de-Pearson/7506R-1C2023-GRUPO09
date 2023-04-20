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

En este jupyter notebook vamos a explorar un conjunto de datos sobre reservas de hoteles y tratar de hallar un modelo que nos permita predecir si la reserva va a ser cancelada

# Preparación del ambiente de trabajo
Importamos todas las librerías que vamos a usar


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
from matplotlib import pyplot as plt
import scipy.stats as st
from calendar import month_name
#Si estamos  en colab tenemos que instalar la libreria "dtreeviz" aparte. 
if IN_COLAB == True:
    !pip install 'dtreeviz'
import dtreeviz.trees as dtreeviz
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score,f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
```

## Cargamos de datos a un dataframe

Se carga la información a un dataframe de pandas, se genera una copia y se trabaja sobre la misma

```python
hotelsDfOriginal = pd.read_csv("./hotels_train.csv")
hotelsdf = hotelsDfOriginal.copy()

print("El data frame esta compuesto por "f"{hotelsdf.shape[0]}"" filas y "f"{hotelsdf.shape[1]}"" columnas")
```

## Vistazo inicial

Un vistazo básico a la información contenida en el dataframe:

```python
pd.concat([hotelsdf.head(2), hotelsdf.sample(5), hotelsdf.tail(2)])
```


Renombramos las columnas del dataframe con nombres mas claros

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
    'is_canceled':'is_canceled',
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

hotelsdf.rename(columns = nuevas_columnas, inplace = True)
```

Por otro lado, podemos observar que tipo de dato almacena cada columna y cuales tienen datos faltantes

```python
hotelsdf.info()
```

# Analisis univariado
Vamos a dividir las variables en cuantitativas y cualitativas.

|     Nombre de la variable           |       Tipo      |      Descripcion         |
| ----------------------------------- | --------------- | ------------------------ |  
| average_daily_rate                  | Cuantitativa    | Promedio de la ganancia diaria, por habitación                              |
| adult_num                           | Cuantitativa    |           Numero de adultos en la reserva              |
| agent_id                            | Cualitativa     | ID de la agencia de viaje que hizo la reserva                         |
| arrival_month_day                   | Cuantitativa    | Día del mes que llego                         |
| arrival_month                       | Cualitativa     |  Mes de llegada                        |
| arrival_week_number                 | Cuantitativa    |   Numero de la semana de llegada                       |
| arrival_year                        | Cuantitativa    |    Año de llegada                      |
| assigned_room_type                  | Cualitativa     |     Tipo de cuarto                    |
| babies_num                          | Cuantitativa    |      Cantidad de bebes                     |
| booking_changes_num                 | Cuantitativa    |      Cantidad de cambios a la reserva hasta el día de llegada                    |
| booking_id                          | Cualitativa     |        ID de la reserva                  |
| children_num                        | Cuantitativa    |         Cantidad de niños                 |
| company_id                          | Cualitativa     |         ID  de la compañía que hizo la reserva                 |
| country                             | Cualitativa     |        País de origen                  |
| customer_type                       | Cualitativa     |         Tipos de reserva                 |
| days_in_waiting_list                | Cuantitativa    |         Cantidad de días en la lista de espera                 |
| deposit_type                        | Cualitativa     |          Tipo de deposito para la reserva                |
| distribution_channel                | Cualitativa     |  Medio por el cual se hizo la reserva                        |
| hotel_name                          | Cualitativa     |  Nombre del hotel                        |
| is_canceled                         | Cualitativa     |   Si la reserva fue cancelada o no                       |
| is_repeated_guest                   | Cualitativa     |    Si el invitado ya había ido al hotel                      |
| lead_time                           | Cuantitativa    |   Cantidad de días entre el día que se realizo la reserva y el día de llegada                       |
| market_segment_type                 | Cualitativa     |       Categoría de mercado                   |
| meal_type                           | Cualitativa     |       Tipo de comida pedida                   |
| previous_bookings_not_canceled_num  | Cuantitativa    |       Cantidad de reservas previas no canceladas                   |
| previous_cancellations_num          | Cuantitativa    |       Cantidad de reservas canceladas                   |
| required_car_parking_spaces_num     | Cuantitativa    |       Cantidad de lugares de estacionamiento pedido                  |
| reserved_room_type                  | Cualitativa     |       Tipo de cuarto reservado                   |
| weekend_nights_num                  | Cuantitativa    |        Cantidad de noches de fin de semana que estuvo                  |
| week_nights_num                     | Cuantitativa    |         Cantidad de noches de semana que estuvo                 |
| special_requests_num                | Cuantitativa    |         Cantidad de pedidos especiales hechos                 |



## Cuantitativas

Se trabaja inicialmente sobre las variables que han sido identificadas como cuantitativas, se grafican y se intenta realizar la identificación de outliers, por otro lado, aquellas que de un análisis exploratorio previo arrojaron la existencia de *nulls/nans* se realiza algún tipo de reemplazo por el valor más conveniente

Creamos una lista con todas las variables cuantitativas

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
"previous_bookings_not_canceled_num",
"previous_cancellations_num",
"required_car_parking_spaces_num",
"special_requests_num",
"weekend_nights_num",
"week_nights_num",
]
```
### Adult number 

Realizamos un análisis sobre la variable adult number

#### Valores estadísticos relevantes 

```python
hotelsdf.adult_num.describe()
```

Dentro de los parámetros estadísticos representativos observamos un mínimo de 0 adultos y un máximo de 55, ambos representando registros con valores anormales. 

#### Gráfica de distribución

Para mas información sobre la frecuencia de los valores se grafican las frecuencias

```python
sns.countplot(data = hotelsdf, x = 'adult_num', palette= 'Set2')
plt.title('Cantidad de adultos por reserva')
plt.xlabel('Numero de adultos')
plt.ylabel('Frecuencia')
```

#### Outliers

```python
a_eliminar_con_cero = hotelsdf[hotelsdf['adult_num'] == 0]
a_eliminar_con_cero 
```

```python
a_eliminar_mayores_3 = hotelsdf[hotelsdf['adult_num'] > 3]
a_eliminar_mayores_3
```

```python
print(f'Total de registros a eliminar: {len(a_eliminar_con_cero) + len(a_eliminar_mayores_3)}')
```

Existen 41 registros con valores superiores a 3, los cuales representan outliers. A su vez, se incluyen a estos registros aquellos valores identificados previamente 

#### Ajustes de valor

Eliminamos dichos valores que representan un porcentaje ínfimo y pueden llegar a desviar las futuras predicciones

```python
hotelsdf.drop(a_eliminar_con_cero.index, inplace = True)
hotelsdf.drop(a_eliminar_mayores_3.index, inplace = True)
hotelsdf.reset_index(drop=True)
hotelsdf[(hotelsdf["adult_num"] > 4) | hotelsdf['adult_num'] == 0]
```

Por otro lado realizamos de nuevo las gráficas de la distribución para verificar que no cambie significativamente

```python
sns.countplot(data = hotelsdf, x = 'adult_num', palette= 'Set2')
plt.title('Cantidad de adultos por reserva')
plt.xlabel('Numero de adultos')
plt.ylabel('Frecuencia')
```

### arrival month day

#### Valores estadisticos relevantes

```python
hotelsdf["arrival_month_day"].describe()
```

Los parámetros estadísticos relevantes no muestran por si solos valores irregulares en el análisis

#### Gráfica de distribución

Una gráfica puede llegar a esclarecer o identificar valores fuera de lo común dentro del dataframe

```python
plt.figure(figsize=(8,4))
sns.countplot(data = hotelsdf, x = 'arrival_month_day')
plt.title("Dia de llegada del mes")
plt.xlabel(xlabel = 'Dia de llegada')
plt.ylabel(ylabel = 'Frecuencia')
```

El análisis uní variado de arrival month day no arroja información relevante, pero por otro lado, muestra que la variable no presenta ningún valor inesperado y desmuestra que no hay un día de predilecto del mes

```python
plt.xlabel(xlabel = 'Dia de llegada')
sns.boxplot(data = hotelsdf['arrival_month_day'])
plt.title("Dia de llegada del mes")
plt.ylabel(ylabel = 'Frecuencia')
```
Por lado un boxplot afirma las concluciones derivadas del gráfico anterior 


### arrival week number 

#### Valores estadisticos relevantes

```python
hotelsdf.arrival_week_number.describe()
```
Un vistazo inicial a los parámetros estadisticos no muestra inconsistencias en los registros

#### Gráfica de distribución

```python
plt.figure(figsize=(15,5))
sns.countplot(data = hotelsdf, x = 'arrival_week_number', palette='Set2')
plt.title('Semanas del año')
plt.xlabel('Numero de la semana')
plt.ylabel('Frecuencia')
```
De la gráfica concluimos que no existen outliers entre los registros 

### arrival year 

#### Valores estadisticos relevantes

```python
hotelsdf.arrival_year.describe()
```

#### Gráfica de distribución

```python
sns.countplot(data = hotelsdf, x = 'arrival_year')
plt.xlabel('Años')
plt.ylabel('Frecuencia')
plt.title('Año de las reservas')
```
Todos los registros corresponden a los años: 2015, 2016 y 2017 siendo el año 2016 el mas frecuente entre los registros

### Average Daily Rate

Realizamos un analisis sobre la variable average daily rate

#### Valores estadisticos relevantes 

```python
hotelsdf.average_daily_rate.describe()
```

#### Grafica de distribución

```python
sns.kdeplot(data = hotelsdf.average_daily_rate)
plt.xlabel(xlabel = 'Montos')
plt.ylabel(ylabel = 'Frecuencia')
plt.title('Distribucion del Precio promedio de renta diaria')
```

#### Outliers

Del grafico anterior se observan registros de average daily rate los cuales tienen asignados 0, se debe estudiar a que se deben esos valores, asi como tambien tratar el valor negativo que aparece como mínimo, por otro lado, analizamos cuantos de los precios presentes en los registros presentan una desviacion considerable de los valores esperados

```python
sns.boxplot(data = hotelsdf['average_daily_rate'])
plt.title("Precio promedio de renta diaria")
plt.xlabel('Average daily rate')
plt.ylabel('Montos')
```

```python

valores_con_cero = len(hotelsdf[hotelsdf['average_daily_rate'] <= 0])
total_valores = len(hotelsdf.average_daily_rate)
porcentaje_con_cero = valores_con_cero/total_valores
print(f" Los de adrs que registran un valor de 0 representa un porcentaje de:{porcentaje_con_cero}' por lo tanto considerando que no son representativos, eliminamos dichos registros inconsistentes ")
```

#### Ajustes de valor

eliminar valores con 0

```python
a_eliminar_con_cero = hotelsdf[hotelsdf['average_daily_rate'] <= 0].index
hotelsdf.drop(a_eliminar_con_cero, inplace = True)
hotelsdf.reset_index(drop=True)
```

Utilizamos Z-score para clasificar las desviasiones presentes en los valores


```python

media_requisitos=np.mean(hotelsdf.average_daily_rate)

std_ard=np.std(hotelsdf.average_daily_rate)

hotelsdf['z_adr']=(hotelsdf.average_daily_rate - media_requisitos)/std_ard

hotelsdf['z_adr']=st.zscore(hotelsdf.average_daily_rate)
hotelsdf[(hotelsdf['z_adr'] > 3) | (hotelsdf['z_adr'] < -2)]
```

Graficamos el Z-score del adr


```python
plt.hist(hotelsdf.z_adr)
plt.title('Histograma Z-Score req')
plt.xlabel('Z-Score req')
plt.xticks([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
plt.show()
```

```python
desviacion_uno = hotelsdf[(hotelsdf['z_adr'] > 3)]
desviacion_dos = hotelsdf[(hotelsdf['z_adr'] < -2)]
hotelsdf.drop(desviacion_uno.index, inplace = True)
hotelsdf.drop(desviacion_dos.index, inplace = True)
hotelsdf.reset_index(drop=True)
```

```python
total_valores = len(hotelsdf.average_daily_rate)
cantidad_a_eliminar = desviacion_uno.average_daily_rate.count() + desviacion_dos.average_daily_rate.count()
print("Vamos a eliminar " + str(cantidad_a_eliminar)  + " valores ya son valores que tienen una desviacion estandar muy marcada con  respecto al resto de los valores. Ademas, estos valores representan un " +  str(cantidad_a_eliminar/total_valores) + " porcentaje del total")
```

Graficamos nuevamente con el proposito de verificar la nueva distribución adquirida luego de la modificacion 



```python
data = hotelsdf.average_daily_rate
sns.kdeplot(data = data)
plt.xlabel(xlabel = 'Average daily rate')
plt.ylabel(ylabel = 'Frecuencia')
plt.title('Distribucion del average daily rate')
```

```python
hotelsdf.drop(labels = 'z_adr', inplace = True, axis = 1)
hotelsdf.reset_index(drop=True)
```

### babies number 

#### Valores estadisticos relevantes


```python
hotelsdf.babies_num.describe()
```

#### Grafica de distribución

```python
eje_y = hotelsdf.babies_num.value_counts()
eje_x = eje_y.index.tolist()
sns.barplot(y = eje_y, x = eje_x, palette='Set2')
plt.xlabel('Cantidad de bebes')
plt.ylabel(ylabel='Frecuencia')
plt.title('Numero de bebes por reserva')
```

#### Outliers

```python
hotelsdf[(hotelsdf.babies_num >= 1) & (hotelsdf.adult_num < 1)]
```

#### Ajustes de valor

```python
hotelsdf.drop(hotelsdf[hotelsdf.babies_num == 9].index, inplace = True)
hotelsdf.reset_index(drop=True)
```

### booking changes number 

#### Valores estadisticos relevantes

```python
hotelsdf.booking_changes_num.describe()
```

#### Grafica de distribución

```python
eje_y = hotelsdf.booking_changes_num.value_counts()
eje_x = eje_y.index.tolist()
sns.barplot(y = eje_y, x = eje_x, palette='Set2')
plt.xlabel('Numero de cambios')
plt.ylabel(ylabel='Frecuencia')
plt.title('Cantidad de cambios por reserva')
```
### children number 
#### Grafica de distribución

```python
hotelsdf["children_num"].describe()
```
Children number representa la cantidad de niños que fueron registrados en la reserva.\
Esta variable es **discreta**, porque representa una cantidad discreta de niños.\
Sin embargo, esta almacenada como float64 porque tiene valores faltantes.


#### Valores nulos/faltantes

```python
hotelsdf.children_num.isna().sum()
```

Vemos que tenemos 4 valores faltantes.
Vamos a ver cuales son

```python
hotelsdf[hotelsdf["children_num"].isna() == True]
```

```python
cantidadFilas = len(hotelsdf.index)
cantidadDeChildrenNumVacios = hotelsdf.children_num.isna().sum()
print("Considerando que la cantidad de datos de children_num faltante es " + str((cantidadDeChildrenNumVacios * 100) / cantidadFilas) + "%, lo podemos borrar")
```

```python
# Borramos las columnas sin valores
hotelsdf.drop((hotelsdf[hotelsdf["children_num"].isna() == True].index.values),inplace=True)
hotelsdf.reset_index(drop=True)
```

```python
# Casteamos la columna de children number a int, ahora que ya no tiene los valores nana
hotelsdf = hotelsdf.astype({'children_num':'int'})
```

```python
# Corroboramos que el casteo funciono
print(hotelsdf["children_num"].dtypes)
```

#### Outliers

```python
eje_y = hotelsdf["children_num"].value_counts()
eje_x = eje_y.index.tolist()
sns.barplot(y = eje_y, x = eje_x, palette='Set2')
plt.xlabel('Cantidad de ninos')
plt.ylabel(ylabel='Frecuencia')
plt.title('Numero de ninos por reserva')

hotelsdf["children_num"].value_counts()
```

Vemos que la gran mayoria de las reservas fueron hechas con 0 niños.\
Unos menos con 1 y 2; e incluso menos con 3. \
Sin embargo, nos figura una fila que reservo con 10 niños. Dicha fila es la siguiente:

```python
hotelsdf[hotelsdf["children_num"] == 10]
```

#### Ajustes de valor




Considerando que es un valor tanto mas alto que el resto, que es un unico caso y considerando que fue hecha con **2 adultos** nada mas; podemos considerar que este outlier y que lo podemos remover. 

```python
hotelsdf.drop((hotelsdf[hotelsdf["children_num"] == 10].index.values),inplace=True)
hotelsdf.reset_index(drop=True)
```

Una vez ajustados los valores, nuestros valores toman la siguiente forma:

```python
eje_y = hotelsdf["children_num"].value_counts()
eje_x = eje_y.index.tolist()
sns.barplot(y = eje_y, x = eje_x, palette='Set2')
plt.xlabel('Cantidad de ninos')
plt.ylabel(ylabel='Frecuencia')
plt.title('Numero de ninos por reserva')

hotelsdf["children_num"].value_counts()
```

### days in the waiting list 


#### Valores estadisticos relevantes

```python
hotelsdf["days_in_waiting_list"].describe()
```

Days in waiting list representa la cantidad de dias que la reserva estuvo en la lista de espera antes de serconfirmada.
Esta variable es **discreta**, porque representa una cantidad discreta de dias.\
Esta esta alamacenada como int:

```python
print(hotelsdf["days_in_waiting_list"].dtype)
```

#### Valores nulos/faltantes

```python
hotelsdf.days_in_waiting_list.isna().sum()
```

No tiene valores vacios


#### Grafica de distribución

```python
print("Los valores que toma la variable son los siguientes:")
daysInWaitingListValores = (hotelsdf["days_in_waiting_list"].unique())
daysInWaitingListValores.sort()
print(daysInWaitingListValores)
print()
print("Y toma dichos valores con la siguiente frecuencia")
hotelsdf["days_in_waiting_list"].value_counts()
```

Observamos que la gran mayoria de la gente estuvo 0 dias en la lista de espera. 


Vamos a graficar los valores mayores a 0 para poder apreciar la distribución de los otros datos

```python
mayor0=hotelsdf[hotelsdf["days_in_waiting_list"] > 0]
mayor0.reset_index(drop=True)
plt.hist(mayor0.days_in_waiting_list)
plt.title('Histograma dias en la lista de espera')
plt.xlabel('Cantidad de dias')
plt.show()
```

Vamos a trazar un boxplot para tratar de identificar valores outliers

```python
sns.boxplot(data = hotelsdf['days_in_waiting_list'])
plt.title("Dias en la lista de espera")
plt.xlabel('Frecuencia')
plt.ylabel('Dias')
```

La forma de este grafico nos muestra que tenemos muchos casos de 1 sola ocurrencia para todos los valores que no son 0.
Sin embargo esos valores representan un:

```python
print(str((len(mayor0)*100)/len(hotelsdf)) + "%")
```

Vale casi un 4% del total. Consideramos un tanto elevado para eliminarlos 


### lead time 


#### Valores estadisticos relevantes

```python
hotelsdf["lead_time"].describe()
```

Lead time representa la cantidad de dias que hubo entre el dia que se realizo la reserva y el dia de llegada.\
Esta variable es **discreta**, porque representa una cantidad discreta de dias.\
Esta esta alamacenada como int:

```python
print(hotelsdf["lead_time"].dtype)
```

#### Valores nulos/faltantes

```python
hotelsdf.days_in_waiting_list.isna().sum()
```

No tiene valores faltantes


#### Grafica de distribución


Vamos a analizar la frecuencia de los distintos valores que lead time puede tomar

```python
print("Los valores que toma la variable son los siguientes:")
daysInWaitingListValores = (hotelsdf["lead_time"].unique())
daysInWaitingListValores.sort()
print(daysInWaitingListValores)
print()
print("Y toma dichos valores con la siguiente frecuencia")
hotelsdf["lead_time"].value_counts()
```

Vamos a graficarlos para ver su distribución

```python
plt.hist(hotelsdf.lead_time)
plt.title('Histograma dias de anticipacion de la reserva')
plt.xlabel('Cantidad de dias')
plt.ylabel('Frecuencia')
plt.show()
```

Vemos que la mayoria de los valores estan por debajo de 300

```python
sns.boxplot(data=hotelsdf.lead_time)
plt.xlabel("Cantidad de reservas")
plt.ylabel("Cantidad de dias de anticipacion")
plt.title("Boxplot dias de anticipacion de la reserva")
plt.show()
```

```python
hotelsdf[hotelsdf["lead_time"] >= 400]
```

#### Outliers


```python
porcentaje = str((len(hotelsdf[hotelsdf["lead_time"] >= 400]) * 100)/len(hotelsdf))
print("Los valores mas llamativos son aquellos por encima de 400. Dichos valores representan un: " + porcentaje + "%")
```

Es un porcentaje lo suficientemente bajo para poder borrarlos

```python
hotelsdf.drop(hotelsdf[hotelsdf["lead_time"] >= 400].index, inplace = True)
hotelsdf.reset_index(drop=True)
```

Vamos a observar como se ve nuestro grafico despues de sacar los outliers

```python
sns.boxplot(data=hotelsdf.lead_time)
plt.xlabel("Cantidad de reservas")
plt.ylabel("Cantidad de dias de anticipacion")
plt.title("Boxplot dias de anticipacion de la reserva")
plt.show()

plt.hist(hotelsdf.lead_time)
plt.title('Histograma dias de anticipacion de la reserva')
plt.xlabel('Cantidad de dias')
plt.ylabel('Frecuencia')
plt.show()
```

### previous booking not canceled number

#### Valores estadisticos relevantes

```python
hotelsdf["previous_bookings_not_canceled_num"].describe()
```

Esta variable representa la cantidad de reservas que no fueron canceladas por el usuario antes de la reserva actual


#### Valores nulos/faltantes

```python
hotelsdf.previous_bookings_not_canceled_num.isna().sum()
```

#### Grafica de distribución


```python
print("Los valores que toma la variable son los siguientes:")
daysInWaitingListValores = (hotelsdf["previous_bookings_not_canceled_num"].unique())
daysInWaitingListValores.sort()
print(daysInWaitingListValores)
print()
print("Y toma dichos valores con la siguiente frecuencia")
hotelsdf["previous_bookings_not_canceled_num"].value_counts()
```

Vamos a graficar los valores mayores a 0 para poder apreciar la distribución de los otros datos

```python
mayor0=hotelsdf[hotelsdf["previous_bookings_not_canceled_num"] > 0]
mayor0.reset_index(drop=True)
plt.hist(mayor0.days_in_waiting_list)
plt.title('Histograma dias en la lista de espera mayor a 0')
plt.xlabel('Cantidad de dias')
plt.show()
```

Del grafico se observa que la gran mayoria de la gente que no cancelo, no cancelaron entre 1 y 10 veces


#### Outliers

```python
sns.boxplot(data = hotelsdf['previous_bookings_not_canceled_num'])
plt.xlabel('Dias en la lista de espera')
plt.ylabel('Dias')
```

Debido a la gran cantidad de valores con 0, y a la poca cantidad de valores sin 0 todos los valores distintos a 0 figuran como outliers. \
Dichos valores representan:

```python
print(str((len(hotelsdf[hotelsdf["previous_bookings_not_canceled_num"] > 0])*100)/len(hotelsdf)) + "%")
```

Considerando el bajo volumen que representan, decidimos dropearlos

```python
hotelsdf.drop(hotelsdf[hotelsdf["previous_bookings_not_canceled_num"] > 0].index, inplace = True)
hotelsdf.reset_index(drop=True)
```

Sin embargo, al dropearlos, el resto de nuestros valores son 0. Esto quiere decir que todo el resto de las columnas presentan los mismos valores. \
Es por esto que decidimos eliminar la totalidad de la columna visto a que no nos aporta informacion.

```python
hotelsdf.drop("previous_bookings_not_canceled_num", axis=1, inplace=True)
cuantitativas.remove("previous_bookings_not_canceled_num")
hotelsdf.reset_index(drop=True)
```

### previous booking cancellation number


#### Valores estadisticos relevantes
```python
hotelsdf["previous_cancellations_num"].describe()
```

Esta variable representa la cantidad de reservas que si fueron canceladas por el usuario antes de la reserva actual


#### Valores nulos/faltantes
```python
hotelsdf.previous_cancellations_num.isna().sum()
```

#### Grafica de distribución
```python
print("Los valores que toma la variable son los siguientes:")
daysInWaitingListValores = (hotelsdf["previous_cancellations_num"].unique())
daysInWaitingListValores.sort()
print(daysInWaitingListValores)
print()
print("Y toma dichos valores con la siguiente frecuencia")
hotelsdf["previous_cancellations_num"].value_counts()
```
```python
sns.countplot(data = hotelsdf, x='previous_cancellations_num', palette='Set1')
plt.title('Countplot reservas previas no canceladas')
plt.xlabel('Cantidad de reservas')
plt.show()
```

Del grafico y la distribución previa se observa que la gran mayoria de la gente que cancelo, cancelo 1 vez.


#### Outliers

```python
sns.boxplot(data = hotelsdf['previous_cancellations_num'])
plt.xlabel('Dias en la lista de espera')
plt.ylabel('Dias')
```

Del grafico se ve que todos los valores por encima de 0 estan  por fuera de los cuantiles.\
Sin embargo, esos datos representan:

```python
print(str((len(hotelsdf[hotelsdf["previous_cancellations_num"] > 0])*100)/len(hotelsdf)) + "%")
```

Porcentaje que es demasiado elevado como para eliminar


Sin embargo, la mayoria de estos datos estan concetrados en los registros con 1 cancelacion. Si tomamos un umbral un poco mayor podemos descartar los valores atipicos. Por ejemplo, los registros con 2 cancelaciones o mas represenan un outlier

```python
print(str((len(hotelsdf[hotelsdf["previous_cancellations_num"] >= 2])*100)/len(hotelsdf)) + "%")
```

Al ser un porcentaje tan insignificante, decidimos eliminar esas

```python
hotelsdf.drop(hotelsdf[hotelsdf["previous_cancellations_num"] >= 2].index, inplace = True)
hotelsdf.reset_index(drop=True)
```

Observamos como nuestros valores cambiaron despues del ajuste

```python
sns.countplot(data = hotelsdf, x='previous_cancellations_num', palette='Set1')
plt.title('Countplot reservas previas no canceladas')
plt.xlabel('Cantidad de reservas')
plt.show()
```

#### Ajustes de valor

### required car space number 

#### Valores estadisticos relevantes

```python
hotelsdf.required_car_parking_spaces_num.describe()
```

#### Valores nulos/faltantes

```python
print("La cantidad de valores nulos/faltantes es", hotelsdf.required_car_parking_spaces_num.isna().sum())
```

#### Grafica de distribución

```python
sns.countplot(data = hotelsdf, x='required_car_parking_spaces_num')
plt.title("Cantidad de reservas por espacios de estacionamiento")
plt.ylabel("Frecuencia")
plt.xlabel("Espacios de autos requeridos")
```

#### Outliers


Viendo el grafico podemos identificar que el numero de espacios de estacionamiento mas comun es 0, seguido por 1. 
Además encontramos algunos pocos casos en los que se reservaron 2, 3 y 8 espacios.
Estos ultimos son posibles Outliers.

Mostramos dichos registros junto con la columnas de adult_num para analizarlos más en detalle y determinar si alguno de ellos puede ser Outlier y por que.
Nuestro criterio para determinar que un valor es adecuado para esta variable es que haya como mucho 1 espacio de estacionamiento por adulto en la reserva.

```python
registrosDosOMasEspacios = hotelsdf[hotelsdf["required_car_parking_spaces_num"]>=2]
display(registrosDosOMasEspacios[['adult_num', "required_car_parking_spaces_num"]].sort_values(
    by = "required_car_parking_spaces_num", 
    ascending = False
))
```

De la tabla anterior se pueden sacar las siguientes conclusiones:
- El registro con valor de 8 espacios de estacionamiento es claramente un Outlier ya que no es coherente que una habitacion para dos personas haya reservado esa cantidad de espacios de estacionamiento.
- El registro con el valor de 3 espacios de estacionamiento y 2 adultos tambien es un Outliers ya que tampoco es coherente que 2 personas hayan reservado 3 espacios de estacionamiento.
- Los registros restantes NO son Outliers ya que si bien contienen valores poco frecuentes, son coherentes con el criterio explicado en el parrafo de arriba.


#### Ajustes de valor


Con el analisis anteior, tomamos las siguientes decisiones:
- Para el registro con valor de 8 espacios de estacionamiento,lo eliminamos por tratarse de un Outlier muy grosero.
- En el registro registro con el valor de 3 espacios de estacionamiento y 2 adultos, cambiamos el valor de required_car_parking_spaces_num por el valor "2" suponiendo un error de tipeo.
- Se mantienen sin cambios el resto de los registros restantes listados arriba.

```python
hotelsdf.loc[ (hotelsdf.required_car_parking_spaces_num==3) & (hotelsdf.adult_num==2) , "required_car_parking_spaces_num"] = 2
```

```python
sns.countplot(data = hotelsdf, x='required_car_parking_spaces_num')
plt.title("Cantidad de reservas por espacios de estacionamiento")
plt.ylabel("Frecuencia")
plt.xlabel("Espacios de autos requeridos")
```

### special requests number 


#### Valores estadisticos relevantes

```python
hotelsdf.special_requests_num.describe() 
```

#### Valores nulos/faltanteS

```python
print("La cantidad de valores nulos/faltantes es", hotelsdf.special_requests_num.isna().sum())
```

#### Grafica de distribución

```python
sns.countplot(data = hotelsdf, x='special_requests_num', palette='Set1')
plt.title("Reservas por cantidad de requisitos especiales")
plt.xlabel("Cantidad requerimiento especiales")
plt.ylabel("Frecuencia")
```

```python
sns.boxplot(data=hotelsdf.special_requests_num)
plt.xlabel("Cantidad de reservas")
plt.ylabel("Canidad de requisitos especiales")
plt.title("Canidad de requisitos especiales por reserva")
plt.show()
```

#### Outliers


Viendo los graficos vemos que los valores mas frecuentes de requisitos especiales son 0 (ninguno), 1 y 2 y algunos menos con 3. Ademas hay muy pocos con 4 y 5. 
Los valores que podrian levantar sospecha son 4 y 5.
Miramos la cantidad de registros de cada uno de ellos para ver que no sean casos puntuales.

```python
print("hay", hotelsdf[hotelsdf.special_requests_num==4].shape[0] ,"reservas con 4 requisitos especiales")
print("hay", hotelsdf[hotelsdf.special_requests_num==5].shape[0] ,"reservas con 5 requisitos especiales")
```

#### Ajustes de valor


Debido a la la cantidad de reservas para estos casos y que el rango de valores es relativamente acotado, no parecen ser casos puntuales. 
Procedemos a cambiar la cantidad de requisitos especiales de dichos registros el valor mas frecuente

```python
media_special_requests = round(hotelsdf.special_requests_num.mean())
hotelsdf.loc[hotelsdf['special_requests_num'] >= 4, 'special_requests_num'] = media_special_requests
```

Graficamos nuevamente la distribución de la variable para validar los cambios realizados 

```python
sns.countplot(data = hotelsdf, x='special_requests_num', palette='Set1')
plt.title("Reservas por cantidad de requisitos especiales")
plt.xlabel("Cantidad requerimiento especiales")
plt.ylabel("Frecuencia")
```

### weekend nights number


#### Valores estadisticos relevantes

```python
hotelsdf.weekend_nights_num.describe() 
```

#### Valores nulos/faltantes

```python
print("La cantidad de valores nulos/faltantes es", hotelsdf.weekend_nights_num.isna().sum())
```

#### Grafica de distribución

```python
sns.countplot(data = hotelsdf, x='weekend_nights_num', palette='Set1')
plt.title("Reservas por cantidad de noches de fin de semana")
plt.xlabel("Numero de noches de fin de semana")
plt.ylabel("Frecuencia")
```

```python
sns.boxplot(data=hotelsdf.weekend_nights_num)
plt.xlabel("Cantidad de reservas")
plt.ylabel("Cantidad de noches de fin de semana")
plt.title("Cantdad de noches de fin de semana por reserva")
plt.show()
```

#### Outliers


Podriamos suponer como posibles outliers, reservas con muchos dias de estadia. A simple vista se puede ver que hay pocas reservas con 5 o mas noches de fin de semana de estadia. Comenzamos estudiando los valores de 9 o mas dias de fin de semana ya que equivaldrian a un minimo de 4 semanas de estadia.

```python
mayores_a_nueve = hotelsdf[hotelsdf["weekend_nights_num"]>=9]
mayores_a_nueve.shape[0]
```

```python
sns.countplot(data = mayores_a_nueve, x='weekend_nights_num', palette='Set1')
plt.title("Reservas por cantidad de noches de fin de semana")
plt.xlabel("Numero de noches de fin de semana")
plt.ylabel("Frecuencia")
```

#### Ajustes de valor


Son solo 13 registros, es decir, representan muy poca cantidad del total. Tomamos la decision de eliminarlos para evitar que generen ruido al momento de generar el modelo.

```python
mas_de_nueve_noches_finde = hotelsdf[hotelsdf["weekend_nights_num"]>=9]
hotelsdf.drop(mas_de_nueve_noches_finde.index, inplace = True)
hotelsdf.reset_index(drop=True)
```

Hasta ahora analizamos las estadias con mas de 9 noches de fin de semana (al menos un mes de estadia)
Sin embargo nos resta estudiar, los casos de 5, 6, 7 y 8 dias de fin de semana.
Vemos cuantos registros son

```python
mayores_a_5_menores_a_nueve_finde = hotelsdf[hotelsdf["weekend_nights_num"]>=5]
mayores_a_5_menores_a_nueve_finde.shape[0]
```

```python
mayores_a_5_menores_a_nueve_finde = hotelsdf[hotelsdf["weekend_nights_num"]>=5]
sns.countplot(data = mayores_a_5_menores_a_nueve_finde, x='weekend_nights_num', palette='Set1')
```

Como son muchos mas registros posponemos su analisis para estudiarlos en un analisis multivariado despues de terminar de estudiar todas las variables cuantitativas pero los dejamos marcados como posibles registros a modificar.

```python
sns.countplot(data = hotelsdf, x='weekend_nights_num', palette='Set1')
plt.title("Reservas por cantidad de noches de fin de semana")
plt.xlabel("Numero de noches de fin de semana")
plt.ylabel("Frecuencia")
```

### week nights number 


#### Valores estadisticos relevantes

```python
hotelsdf.week_nights_num.describe() 
```

#### Valores nulos/faltantes

```python
print("La cantidad de valores nulos/faltantes es", hotelsdf.week_nights_num.isna().sum())
```

#### Grafica de distribución

```python
sns.countplot(data = hotelsdf, x='week_nights_num', palette='Set1')
plt.title("Reservas por cantidad de noches de de semana")
plt.xlabel("Noches de semana")
plt.ylabel("Frecuencia")
```

Se puede ver que la gran mayoria de las reservas son estadias de entre ninguna (0) y 5 noches de semana y en menor medida estadias de entre 6 y 10 noches de semana. 
Como en el grafico no se ven puntualmente los registros con estadias de 11 o mas noches de semana, los graficamos de nuevo para ver como se distribuyen y estudiarlos mejor

```python
mayores_a_11_noches_semana = hotelsdf[hotelsdf["week_nights_num"]>=11]
mayores_a_11_noches_semana.shape[0]
```

```python
sns.countplot(data = mayores_a_11_noches_semana, x='week_nights_num', palette='Set1')
plt.title("Estancias de mas de once dias")
plt.xlabel("Noches de semana")
plt.ylabel("Frecuencia")
```

Como son muchos registros y no contienen valores incoherentes a primera vista posponemos su tratamiento para estudiarlos con un analisis multivariado comparandolo con weekend_nights_number en dicha seccion.

Ocurre que ademas los registros que este caso representarian una desviacion muy grande fueron eliminados al momento de eliminar aquellos outliers de noches de fin de semana


## Cualitativas

Variables cualitativas

En un principio establecemos una lista que contenga todas las variables cualitativas

```python
cualitativas = [
"agent_id",
"arrival_month",
"assigned_room_type",
"booking_id",
"company_id",
"country",
"customer_type",
"deposit_type",
"distribution_channel",
"hotel_name",
'is_canceled',
"is_repeated_guest",
"market_segment_type",
"meal_type",
"reserved_room_type",
]
```

### Valores nulos

```python
cualitativas_nulas = hotelsdf[cualitativas].isnull().sum()
cualitativas_nulas = cualitativas_nulas[cualitativas_nulas > 0]

cuantitativas_nulas_per = pd.Series()

for indice in cualitativas_nulas.index:
    cuantitativas_nulas_per[indice] = cualitativas_nulas[indice]/len(hotelsdf[indice])*100

sns.barplot(x = cuantitativas_nulas_per.index, y = cuantitativas_nulas_per)
plt.ylabel(ylabel= 'Porcentaje')
plt.xlabel(xlabel= 'Nombre columna')
plt.title(label = 'Porcentaje de valores nulos')
plt.ylim(0, 100)
plt.yticks([0,10,20,30,40,50,60,70,80,90,100])
plt.show()
```

Viendo que la columna company_id tiene un 92% de valores faltantes es conveniente para el analisis eliminar la columna 

```python
hotelsdf.drop("company_id", axis=1, inplace=True)
hotelsdf.reset_index(drop=True)
cualitativas.remove("company_id")
```

Por otro lado la variable booking ID no sera graficada puesto que corresponde a una cadena que representa un codigo unico para cada reserva 

### Agent ID

#### Valores que toma

```python
agent_id_valores = (hotelsdf["agent_id"].unique())
agent_id_valores.sort()
print(agent_id_valores)
```

#### Ajuste de valores faltantes

Reemplazamos valores faltantes por 0 ya que no existe previamente y servira para regular los tipos de datos de la columna

```python
hotelsdf.loc[hotelsdf['agent_id'].isnull(), 'agent_id'] = 0
hotelsdf[hotelsdf.agent_id.isnull()]
hotelsdf['agent_id'] = hotelsdf['agent_id'].astype(int)
```

#### Grafica de distribución

```python
cantidad = len(hotelsdf['agent_id'].value_counts().index.tolist())
print(f"La cantidad de identificaciones de empresa es: {cantidad}")
```

Debido a que existen 295 id de empresas, graficamos un muestreo de los 10 ids mas frecuentes en el dataframe

```python
data = hotelsdf.agent_id.value_counts().sort_values(ascending=False).head(10)
sns.barplot(y = data, x = data.index.tolist())
plt.title('Top 10 ID de agencia mas frecuentes')
plt.xlabel('ID de agencia')
plt.ylabel('Frecuencia')
```

El resto de valores tienen representaciones de ids validas pero aparecen de manera menos frecuente, al ser tantos, mostramos los mas frecuentes para darnos una idea. En este caso el id mas frecuente es el 9 con aproximadamente unos 16 mil registros 

### arrival month

#### Valores que toma

```python
arrival_month_valores = (hotelsdf["arrival_month"].unique())
month_lookup = list(month_name)
months = arrival_month_valores
sorted(months, key=month_lookup.index)
```

#### Grafica de la distribución

```python
plt.title('Meses de llegada')
sns.countplot(data = hotelsdf, x = 'arrival_month', palette='Set2')
plt.xticks(rotation=45)
plt.xlabel('Meses')
plt.ylabel('Frecuencia')
```

Del siguiente grafico observamos que el mes de agosto es el mes con mas reservas hechas, por otro lado enero es el mes con menos reservas 

### Assigned Room type

### Valores que toma

```python
assigned_room_type = hotelsdf['assigned_room_type'].unique().tolist()
ordenado = sorted(assigned_room_type)
print(ordenado) 
```

Realizamos un grafico de la frecuencia de los tipos de habitaciones asignadas 

#### Grafica de distribución
```python
sns.countplot(data = hotelsdf, x='assigned_room_type', palette='Set2')
plt.title('Tipos de habitacion asignada')
plt.xlabel('Tipo de habitacion')
plt.ylabel('Frecuencia')
```

Del cual concluimos que las habitaciones de tipo: H, I y K son las menos frecuentas y la habitacion tipo A se lleva la mayoria de las apariciones en los registros 

### Country

#### Valores que toma

```python
country = hotelsdf['country'].unique().tolist()
print(country) 
```

#### Grafica de distribución

```python
data = hotelsdf.country.value_counts().sort_values(ascending=False).head(20)
plt.xticks(rotation=45)
sns.barplot(y = data, x = data.index.tolist(), palette='Set2')
plt.title('Paises por reserva')
plt.xlabel('Pais')
plt.ylabel('Frecuencia')
```

Del grafico concluimos que Portugal es el pais del cual hay mayor numero de reservas, seguido de: Alemania, Fracia, España. Por otro lado aun hay registros con valores faltantes 

#### Valores faltantes

Para evitar la eliminacion de los registros y debido a la muy marcada tendencia de las reservas a venir de Portugal asignamos a los valores faltantes dicho pais, puesto que representas aproximadamente un 0.2% de los datos 

```python
hotelsdf.loc[hotelsdf['country'].isnull(), 'country'] = 'PRT'
```
Dicha asignacion no genera una desviacion, la prueba de eso en el siguiente grafico 

```python
data = hotelsdf.country.value_counts().sort_values(ascending=False).head(20)
plt.xticks(rotation=45)
sns.barplot(y = data, x = data.index.tolist(), palette='Set2')
plt.title('Paises por reserva')
plt.xlabel('Pais')
plt.ylabel('Frecuencia')
```


### Customer type

#### Valores que toma

```python
customer_typeValores = (hotelsdf["customer_type"].unique())
customer_typeValores.sort()
print(customer_typeValores)
```

#### Grafica de distribución

```python
sns.countplot(data = hotelsdf, x = 'customer_type', palette='Set2')
plt.title("Tipos de clientes")
plt.ylabel("Frecuencia")
plt.xlabel("Tipo de cliente")
```


### Deposit type

#### Valores que toma

```python
deposit_typeValores = (hotelsdf["deposit_type"].unique())
deposit_typeValores.sort()
print(deposit_typeValores)
```

#### Grafica de distribución
```python
sns.countplot(data = hotelsdf, x = 'deposit_type', palette='Set2')
plt.title("Tipo de deposito en las reservas")
plt.ylabel("Frecuencia")
plt.xlabel("Tipo de deposito")
```
Del grafico apreciamos las frecuencias y los tipos de depositos disponibles en el dataframe, siendo No Deposit el mas frecuente y por el contrario Refundable se queda con la menor frecuencia


### Distribution channel

#### Valores que toma

```python
distribution_channelValores = (hotelsdf["distribution_channel"].unique())
distribution_channelValores.sort()
print(distribution_channelValores)
```

#### Grafica de distribución
```python
sns.countplot(data = hotelsdf, x = 'hotel_name', palette='Set2')
plt.title("Nombre de los hoteles")
plt.xlabel("Nombre del hotel")
plt.ylabel("Frecuencia")
```

Estudiamos la variable nombre de hotel, encontrando dos hoteles en el dataframe 

### Hotel Name

#### Valores que toma

```python
hotel_nameValores = (hotelsdf["hotel_name"].unique())
hotel_nameValores.sort()
print(hotel_nameValores)
```

#### Grafica de distribución
```python
sns.countplot(data = hotelsdf, x = 'hotel_name', palette='Set2')
plt.title("Nombre de los hoteles")
plt.xlabel("Nombre del hotel")
plt.ylabel("Frecuencia")
```

Estudiamos la variable nombre de hotel, encontrando dos hoteles en el dataframe 

### Is canceled (Target)

#### Valores que toma

```python
is_canceledValores = (hotelsdf["is_canceled"].unique())
is_canceledValores.sort()
print(is_canceledValores)
```

#### Grafica de distribución
La variable a predecir, dicha variable tiene valores 0 y 1, siendo 0 no cancelado y 1 las reservas canceladas

#### Grafica de distribución
```python
para_ver = pd.DataFrame()
para_ver['is_canceled'] = hotelsdf['is_canceled'].map({1: 'Cancelado', 0: 'No cancelado'})
sns.countplot(data = para_ver, x = 'is_canceled', palette='Set2')
plt.title('Estado final de la reserva, variable target')
plt.ylabel('Frecuencia')
plt.xlabel("Estado")
```

### Is repeated guest

#### Valores que toma


```python
is_repeated_guestValores = (hotelsdf["is_repeated_guest"].unique())
is_repeated_guestValores.sort()
print(is_repeated_guestValores)
```

#### Grafica de distribución

```python
para_ver['is_repeated_guest'] = hotelsdf['is_repeated_guest'].map({1: 'Si', 0: 'No'})
sns.countplot(data = para_ver, x = 'is_repeated_guest', palette='Set2')
plt.title('Huespedes con visitas previas al hotel en la reserva')
plt.ylabel('Frecuencia')
plt.xlabel('Huesped repetido')
```

Del grafico es facil concluir que la mayoria de las reservas fueron realizados por usuarios que visitan por primera vez el hotel escogido



### Market segment

#### Valores que toma


```python
market_segment_typeValores = (hotelsdf["market_segment_type"].unique())
market_segment_typeValores.sort()
print(market_segment_typeValores)
```

#### Grafica de distribución
#### Grafica de distribución

```python
plt.xticks(rotation=30)
sns.countplot(data=hotelsdf, x = 'market_segment_type', palette='Set2')
plt.title("Tipo de segmento de mercado")
plt.ylabel("Frecuencia")
plt.xlabel("Segmento del mercado")
```

### meal type

#### Valores que toma

```python
meal_typeValores = (hotelsdf["meal_type"].unique())
meal_typeValores.sort()
print(meal_typeValores)
```

#### Grafica de distribución

```python
sns.countplot(data=hotelsdf, x = 'meal_type', palette='Set2')
plt.title("Tipo de comida por reserva")
plt.ylabel("Frecuencia")
plt.xlabel("Tipo de comida")
```

### Reserved room type

#### Valores que toma

```python
reserved_room_typeValores = (hotelsdf["reserved_room_type"].unique())
reserved_room_typeValores.sort()
print(reserved_room_typeValores)
```

#### Grafica de distribución

```python
sns.countplot(data=hotelsdf, x = 'reserved_room_type')
plt.title("Tipo de habitacion reservada")
plt.ylabel("Frecuencia")
plt.xlabel("Tipo de habitacion")
```

Como ya habiamos observado en la cantidad de dias de fin de semana, la mayor cantidad de gente se queda 


# Estado del data frame post analisis univariado


Vamos a observar el estado de nuestro dataframe actualmente para observar que efecto tuvo nuestro analisis en el volumen de los datos

```python
pd.concat([hotelsdf.head(2), hotelsdf.sample(5), hotelsdf.tail(2)])
```

```python
hotelsdf.info()
```

```python
porcentaje = str(100 - len(hotelsdf) * 100 / len(hotelsDfOriginal))[:5]
print("Vemos que despues del proceso de Ingenieria de caracteristicas, la cantidad de datos se redujo en un " + porcentaje + "%") 
```

Ademas observamos que no tenemos mas datos faltantes, visto en como los unicos valores del tipo float64 es average_daily_rate, el cual es un valor de punto flotante.


# Analisis multivariado
## Medicion de la correlacion entre las variables cuantitativas

Una vez hecho el tratado sobre outliers y datos faltantes medimos la correlacion entre las variables cuantitativas encontradas en el dataframe

```python
# Este if es se usa para evitar problemas de versiones de pandas entre la version local y la presente en Google Collab
if (pd.__version__) == "1.5.2":
    correlaciones = hotelsdf[cuantitativas].corr(numeric_only=True)

else:
    correlaciones = hotelsdf[cuantitativas].corr()

    
sns.set(style = 'darkgrid')
plt.figure( figsize = (12, 9))
sns.heatmap(data = correlaciones,annot = True, vmin = -1, vmax =1, fmt='.2f')
sns.color_palette("mako", as_cmap=True)
plt.show()
```

### Week nights number vs Weekend nights number

Como dijimos previamente, analizar por separado las noches de dia de semana y las noches de dias de fin de semana no basta parta estudiar dichas variables. El primer problema que podria surgir es que la cantidad de noches de semana y de fin de semana no guarden una relacion coherente. Al graficarlos debería ocurrir lo siguiente:
- Cuando el numero de noches n de fin de semana es impar, las pendientes de las rectas tienen una variacion de +/- 5 noches de semana
- Cuando el numero de noches n de fin de semana es par, las pendientes de las rectas tienen una variacion de +- 10 noches de semana

```python
sns.scatterplot(x=hotelsdf.weekend_nights_num,y=hotelsdf.week_nights_num)
plt.title('Dispersograma noches finde vs noches de semana')
plt.show()
```

Al observar el grafico observamos que todos los puntos se encuentran en los rangos explicados anteriormente.


Sin embargo, ocurre que si bien puede resultar util tener datos sobre las noches de semana y las de fin de semana, un dato que nos podria resultar aun mas util es la cantidad de noches totales de estadia.
Agregamos una columna con dicho dato

```python
hotelsdf["dias_totales"] = hotelsdf["week_nights_num"] + hotelsdf["weekend_nights_num"]
```

Puesto que ahora tenemos una nueva variable, realizamos un breve analisis univariado sobre la misma para determinar si existen Outliers no detectados en las columnas de week y weekend nights number.

```python
plt.figure(figsize=(15,5))
sns.countplot(data = hotelsdf, x = 'dias_totales', palette= 'Set2')
plt.title('Cantidad de reservas por dias de estadia')
plt.xlabel('Dias de estadia')
plt.ylabel('Frecuencia')
```

La mayoria de las reservas son de estadias de entre 1 y 7 dias de estadia. En menor medida se observan reservas para estadias entre 8 y 14 dias y por ultimos unas pocas entre 15 y 30 dias. Realizamos un boxplot para darnos una idea de que numero utilizar como corte para determinar outliers.

```python
plt.xlabel(xlabel = 'Dia estadia')
sns.boxplot(data = hotelsdf['dias_totales'])
plt.title("reservas por dias de estadia")
plt.ylabel(ylabel = 'Frecuencia')
```

Segun el grafico se alejarian de la media todos los valores de 8 o mas dias de estadia. Vemos cuantos registros son y que porcentaje representan del total

```python
reservas_mas_de_ocho_dias = hotelsdf[(hotelsdf.dias_totales>=8)].shape[0]
print("hay",reservas_mas_de_ocho_dias,"que representan un porcentaje del total de", reservas_mas_de_ocho_dias*100/hotelsdf.shape[0],"%")
```

Puesto que este valor es muy elevado, apelamos al sentido comun. Reservas de hasta 14 dias de estadia son muy posibles, por lo cual estudiamos las de mas 15 o mas dias.

```python
quince_o_mas_dias = hotelsdf[hotelsdf["dias_totales"]>=15]
#plt.figure(figsize=(15,5))
sns.countplot(data = quince_o_mas_dias, x = 'dias_totales', palette= 'Set2', hue = "is_canceled")
plt.title('Cantidad de dias de estadia')
plt.xlabel('Dias de estadia')
plt.ylabel('Frecuencia')
```

Primero vemos cuantos registros son en total

```python
print("hay",quince_o_mas_dias.shape[0],"que se quedan 15 o mas dias y representan un porcentaje del total de", (quince_o_mas_dias.shape[0])*100/hotelsdf.shape[0],"%")
```

Luego vemos cuantos de esos cancelan

```python
cancelaron_y_quince_o_mas_dias = hotelsdf[ (hotelsdf.dias_totales>=15) & (hotelsdf.is_canceled == 1) ].shape[0]

print("hay",cancelaron_y_quince_o_mas_dias,"que cancelaron y se quedaron mas de 15 o mas dias.Osea un", cancelaron_y_quince_o_mas_dias*100/quince_o_mas_dias.shape[0],"% de los que se que se quedan mas de 15 dias cancelan")
```

Vemos que el porcentaje de reservas de mas de 15 dias que cancelan es muy alto. Sin embargo, la cantidad de registros con los que ocurre esto son muy pocos. Dejarlos, podria generar ruido al momento de realizar la prediccion. Nos podria llevar, erroneamente a pensar que alguien que se quedo muchos dias cancelaria cuando esto no necesariamente es asi. En el problema q estamos resolviendo, es prefereible no detectar a alguien que cancela, que suponer que alguien cancelaria y que luego no lo haga ya que en terminos de presupuestos, disponibilidad o cualquiera sea el uso que se le de a esta prediccion, no estar preparardo para una reserva perjudicaria mucho mas que estarlo "por las dudas".
Procedemos a eliminarlos

```python
a_eliminar_con_quince_o_mas_dias = hotelsdf[hotelsdf['dias_totales'] >= 15]
hotelsdf.drop(a_eliminar_con_quince_o_mas_dias.index, inplace = True)
hotelsdf.reset_index(drop=True)
```

```python
sns.countplot(data = hotelsdf, x='meal_type', hue='is_canceled')
plt.title("Tipo de comida en la reserva por cancelacion")
plt.xlabel("Tipo de comida")
plt.ylabel("Frecuencia")
```

Una vez analizada la variable contra el target podemos decir que la misma no proporciona suficiente información para el análisis y por lo tanto descartamos la misma para descongestionarla cantidad de variables a usar en el análisis futuro

#### Dias_totales vs Lead_time

```python
sns.scatterplot(x=hotelsdf.dias_totales,y=hotelsdf.lead_time)
plt.title('Dispersograma dias totales lead time')
plt.show()
```

Los días totales y la cantidad de tiempo previo desde la reserva hasta la fecha de llegada se distribuyen de manera homogénea. No se identifican outliers

#### ADR y Tipo de cliente

```python
boxplot = hotelsdf.boxplot(column='average_daily_rate', by='customer_type')
plt.title('Precio diario promedio por tipo de cliente')
plt.suptitle("")
plt.xlabel("Tipo de cliente")
plt.ylabel("Precio diario promedio")
```

Se puede observar en las gráficas que hay valores que se escapan de lo esperado cuando se hace la medición en relación al tipo de cliente, al contabilizar estas observaciones concluimos que son pocas y por lo tanto, eliminamos dichos registros que representan una desviación. También hay que considerar que la media de todos los registros es de aproximadamente 100 (en la unidad correspondiente) y por lo tanto la desviación estándar es muy chica mostrando algunos valores como outliers a pesar de haber pasado por un tratamiento previo 

```python
#obtenemos los indices de los outliers
indices_outliers = hotelsdf[(hotelsdf['customer_type'] == 'Group') & (hotelsdf['average_daily_rate'] > 200)].index
hotelsdf.drop(indices_outliers, inplace = True)
indices_outliers2 = hotelsdf[(hotelsdf['customer_type'] == 'Contract') & (hotelsdf['average_daily_rate'] > 200)].index
hotelsdf.drop(indices_outliers2, inplace = True)
hotelsdf.reset_index(drop=True)
```

Graficamos nuevamente para verificar que dicho tratamiento no generara una desviación considerable en el análisis

 ```python 
boxplot = hotelsdf.boxplot(column='average_daily_rate', by='customer_type')
plt.title('Precio diario promedio por tipo de cliente')
plt.suptitle("")
plt.xlabel("Tipo de cliente")
plt.ylabel("Precio diario promedio")
```

### ADR y Tipo de habitacion

```python 
boxplot = hotelsdf.boxplot(column='average_daily_rate', by='assigned_room_type')
plt.title('Precio diario promedio por tipo de habitacion')
plt.suptitle("")
plt.xlabel("Tipo de cliente")
plt.ylabel("Precio diario promedio")
```

Del gráfico anterior es claro que aparecen outliers en el precio promedio diario de habitación cuando este es agrupado por tipo de habitación. Se identifican los conjuntos de datos que deben ser eliminados o tratados. 

```python
indices_tipo_k = hotelsdf[(hotelsdf['assigned_room_type'] == 'K') & (hotelsdf['average_daily_rate'] > 160)].index
indices_tipo_i = hotelsdf[(hotelsdf['assigned_room_type'] == 'I') & (hotelsdf['average_daily_rate'] > 210)].index
indices_tipo_b = hotelsdf[(hotelsdf['assigned_room_type'] == 'B') & (hotelsdf['average_daily_rate'] < 30)].index
indices_tipo_b2 = hotelsdf[(hotelsdf['assigned_room_type'] == 'B') & (hotelsdf['average_daily_rate'] > 210)].index
print(f"El total de los datos a eliminar es {len(indices_tipo_k ) + len(indices_tipo_b) + len(indices_tipo_b2) + len(indices_tipo_i)}")
```
```python
hotelsdf.drop(indices_tipo_k, inplace=True)
hotelsdf.drop(indices_tipo_i, inplace=True)
hotelsdf.drop(indices_tipo_b, inplace=True)
hotelsdf.drop(indices_tipo_b2, inplace=True)
hotelsdf.reset_index(drop=True)
```

Mostramos nuevamente la distribución de las variables alteradas

```python
boxplot = hotelsdf.boxplot(column='average_daily_rate', by='assigned_room_type')
plt.title('Precio diario promedio por tipo de habitacion')
plt.suptitle("")
plt.xlabel("Tipo de cliente")
plt.ylabel("Precio diario promedio")
```

### Adult number, children number y babies number

Realizamos un grafico con la intencion de detectar outliers 

```python
#Visualizacion 3D

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(projection='3d')
x1 = hotelsdf.adult_num
y1 = hotelsdf.children_num
z1 = hotelsdf.babies_num
ax.scatter(x1,y1,z1, label = 'No cancelados')
ax.set_xlabel('Adultos')
ax.set_ylabel('Niños')
ax.set_zlabel('Bebes')
ax.elev = 5  
ax.azim = -75
plt.title('Comparacion adultos, niños y bebes')
```

A partir del grafico anterior no se puede hacer una observacion relevante en la deteccion de outliers

## Relación contra el target: is_canceled


Vamos a graficar algunas variables haciendo foco en si cancelaron o no. Estas fueron elegidas en base a nuestros analisis multi y univariados y segun el significado que tienen estas variables en el contexto del problema. Optamos por: "lead_time", "average_daily_rate", "previous_cancellations_num", "dias_totales" y "reserved_room_type"


### Lead time

```python
sns.kdeplot(data= hotelsdf, x = "lead_time", hue= "is_canceled")
plt.title("Densidad de registros de lead_time s haciendo foco en is_canceled")
plt.xlabel("Lead time")
plt.ylabel("Densidad")
```

Estas graficas podrian sugerir que reservas realizadas con mayor anticipacion tendrian mas probabilidad de ser canceladas.


### previous_cancellations_num

```python
sns.countplot(data= hotelsdf, x="previous_cancellations_num",  hue= "is_canceled")
plt.title("Cantidad de reservas respecto a la cantidad de cancelaciones previas")
plt.xlabel("Cancelaciones previas")
plt.ylabel("Cantidad de reservas")
```

Sin embargo si hacemos zoom en las reservas con una cancelacion previa...

```python
hotelsdf_con_una_cancelaciones = hotelsdf[ hotelsdf["previous_cancellations_num"] ==1]
sns.countplot(data= hotelsdf_con_una_cancelaciones, x="previous_cancellations_num",  hue= "is_canceled")
plt.title("Cantidad de reservas para una cancelacion previa")
plt.xlabel("Cancelaciones previas")
plt.ylabel("Cantidad de reservas")
```

```python
reservas_con_una_cancelacion = hotelsdf_con_una_cancelaciones.shape[0]
total_reservas_cancelaciones_prev = hotelsdf["previous_cancellations_num"].shape[0]
print("las reservas con 1 cancelacion previa represntan un ", reservas_con_una_cancelacion*100/total_reservas_cancelaciones_prev,"%" )
```

Observando la grafica vemos que existe un salto muy importante en la cantidad de reservas canceladas cuando la cantidad de reservas canceladas anteriormente es 1. Si bien esta variable parece tener una influencia muy importante sobre el valor ocurre con un numero muy pequeño de registros (un 6%)


### Average_daily_rate

```python
sns.kdeplot(data= hotelsdf, x = "average_daily_rate", hue= "is_canceled")
plt.title("Densidad de registros del precio promedio diario por hab haciendo foco en la cancelacion")
plt.xlabel("Precio promedio diario por hab")
plt.ylabel("Densidad de registros")
```

Se puede ver que las garficas de ADR haciendo foco en is_canceled se comportan de manera similiar para todos los valores. No podemos estabecer que exista una influencia directa de esta variable sobre el target.


### Dias Totales

```python
sns.countplot(data = hotelsdf, x = 'dias_totales', palette= 'Set2', hue = "is_canceled")
plt.title('Cantidad de dias de estadia')
plt.xlabel('Dias de estadia')
plt.ylabel('Frecuencia')
plt.title("Cantidad de reservas por dias de estadia")
```

Podemos ver que no existe una relacion directa entre la cantidad de dias de estadia y si la reserva esta cancelada o no.


### reserved_room_type

```python
sns.countplot(data=hotelsdf, x = 'reserved_room_type',  hue= "is_canceled")
plt.title('Cantidad de reservas segun el tipo de cuarto haciendo foco en la cancelacion')
plt.xlabel('Tipo de habitacion')
plt.ylabel("Cantidad de reservas por dias de estadia")
plt.show()
```

Podemos ver que no existe una relacion directa entre el tipo de habitacion elegido y si la reserva esta cancelada o no.


## Conclusion


Como conclusión de esta primera etapa podríamos decir que la única variable que parece tener cierta influencia sobre el target es "lead_time".
Para el resto de las variables, no podemos afirmar que existe una correlación directa entre ellas y el target. Esto se puede observar en sus graficas de distribución en las cuales la cantidad de reservas canceladas es practicamente igual a las no canceladas para casi la totalidad del rango.

