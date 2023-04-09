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

# Exploracion Inicial e ingenieria de caracteristicas
Importamos todas las librerias que vamos a usar


```python
import pandas as pd 
import numpy as np
import sklearn as sk
import seaborn as sns
from matplotlib import pyplot as plt
import scipy.stats as st
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
    'reservation_status':'reservation_status',
    'reservation_status_date':'reservation_status_date',
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
### Evaluar la remocion de la variable id
De este vistazo inicial, se observa que la columna **id** no parece tener un patron distingible.
Analizamos si hay algun valor de ID repetido, para tratar de reconocer un patron

```python
ides = hotelsdf["booking_id"].value_counts()
ides[ides > 1]
```

Como todos los ID's son unicos y no hay ningun ID vacio; consideramos que es un ID sin ningun analitico.

```python
# Codigo para borrar la columna de ID
hotelsdf.drop("booking_id", axis=1, inplace=True)
```

# Analisis de variables
Vamos a dividir las variables en cuantitativas y cualitativas.

|     Nombre de la variable           |       Tipo      |      Descripcion         |
| ----------------------------------- | --------------- | ------------------------ |  
| average_daily_rate                  | Cuantitativa    | Promedio de la ganancia diaria, por habitacion                              |
| adult_num                           | Cuantitativa    |           numero de adultos en la reserva              |
| agent_id                            | Cualitativa     |                          |
| arrival_month_day                   | Cuantitativa    |                          |
| arrival_month                       | Cualitativa     |                          |
| arrival_week_number                 | Cuantitativa    |                          |
| arrival_year                        | Cuantitativa    |                          |
| assigned_room_type                  | Cualitativa     |                          |
| babies_num                          | Cuantitativa    |                          |
| booking_changes_num                 | Cuantitativa    |                          |
| children_num                        | Cuantitativa    |                          |
| company_id                          | Cualitativa     |                          |
| country                             | Cualitativa     |                          |
| customer_type                       | Cualitativa     |                          |
| days_in_waiting_list                | Cuantitativa    |                          |
| deposit_type                        | Cualitativa     |                          |
| distribution_channel                | Cualitativa     |                          |
| hotel_name                          | Cualitativa     |                          |
| is_repeated_guest                   | Cualitativa     |                          |
| lead_time                           | Cuantitativa    |                          |
| market_segment_type                 | Cualitativa     |                          |
| meal_type                           | Cualitativa     |                          |
| previous_bookings_not_canceled_num  | Cuantitativa    |                          |
| previous_cancellations_num          | Cuantitativa    |                          |
| required_car_parking_spaces_num     | Cuantitativa    |                          |
| reservation_status                  | Cualitativa     |                          |
| reservation_status_date             | Cuantitativa    |                          |
| reserved_room_type                  | Cualitativa     |                          |
| weekend_nights_num                  | Cuantitativa    |                          |
| week_nights_num                     | Cuantitativa    |                          |
| special_requests_num                | Cuantitativa    |                          |


## Cuantitativas

Se trabaja inicialmente sobre las variables que han sido identificadas como cuantitativas, se grafican y se intenta realizar la identificación de outliers, por otro lado, aquellas que de un analisis exploratorio previo arrojaron la existencia de *nulls/nans* se realiza algún tipo de reemplazo por el valor más conveniente

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
"reservation_status_date",
"special_requests_num",
"weekend_nights_num",
"week_nights_num",
]
```
## Valores null/na

Observamos cuales de las variables cuantitativas poseen valores nulos/faltantes en sus registros

```python
nulos_cuantitativos = hotelsdf[cuantitativas].isnull().sum()
nulos_cuantitativos = nulos_cuantitativos[nulos_cuantitativos > 0]
nulos_cuantitativos
```
En un principio solo la variable children_num posee valores nulos, en su propia seccion seran tratados dichas observaciones

### Adult number 

Realizamos un analisis sobre la variable adult number

##### Valores estadisticos relevantes 

```python
hotelsdf.adult_num.describe()
```

Dentro de los parametros estadisticos representativos observamos un minimo de 0 adultos y un maximo de 55, ambos representando registros con valores anormales. 

##### Grafica de distribucion

Para mas informacion sobre la frecuencia de los valores se grafican las frecuencias

```python
sns.countplot(data = hotelsdf, x = 'adult_num', palette= 'Set2')
plt.title('Cantidad de adultos por reserva')
plt.xlabel('Numero de adultos')
plt.ylabel('Frecuencia')
```

##### Outliers

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

##### Ajustes de valor

Eliminamos dichos valores que representan un porcentaje infimo y pueden llegar a desviar las futuras predicciones

```python
hotelsdf.drop(a_eliminar_con_cero.index, inplace = True)
hotelsdf.drop(a_eliminar_mayores_3.index, inplace = True)
hotelsdf.reset_index()
hotelsdf[(hotelsdf["adult_num"] > 4) | hotelsdf['adult_num'] == 0]
```

Por otro lado realizamos de nuevo las graficas de la distribucion para verificar que no cambie significativamente

```python
sns.countplot(data = hotelsdf, x = 'adult_num', palette= 'Set2')
plt.title('Cantidad de adultos por reserva')
plt.xlabel('Numero de adultos')
plt.ylabel('Frecuencia')
```

### arrival month day

##### Valores estadisticos relevantes

```python
hotelsdf["arrival_month_day"].describe()
```

Los parametros estadisticos relevantes no muestran por si solos valores irregulares en el analisis

##### Grafica de distribucion

Una grafica puede llegar a esclarecer o identificar valores fuera de lo comun dentro del dataframe

```python
plt.figure(figsize=(8,4))
sns.countplot(data = hotelsdf, x = 'arrival_month_day')
plt.title("Dia de llegada del mes")
plt.xlabel(xlabel = 'Dia de llegada')
plt.ylabel(ylabel = 'Frecuencia')
```

El analisis univariado de arrival month day no arroja informacion relevante, pero por otro lado, muestra que la variable no presenta ningun valor inesperado y desmuestra que no hay un dia de predilecto del mes

```python
plt.xlabel(xlabel = 'Dia de llegada')
sns.boxplot(data = hotelsdf['arrival_month_day'])
plt.title("Dia de llegada del mes")
plt.ylabel(ylabel = 'Frecuencia')
```
Por lado un boxplot afirma las concluciones derivadas del grafico anterior 


### arrival week number 

##### Valores estadisticos relevantes

```python
hotelsdf.arrival_week_number.describe()
```
Un vistazo inicial a los parametros estadisticos no muestra inconsistencias en los registros

##### Grafica de distribucion

```python
plt.figure(figsize=(15,5))
sns.countplot(data = hotelsdf, x = 'arrival_week_number', palette='Set2')
plt.title('Semanas del año')
plt.xlabel('Numero de la semana')
plt.ylabel('Frecuencia')
```
De la grafica concluimos que no existen outliers entre los registros 

### arrival year 

##### Valores estadisticos relevantes

```python
hotelsdf.arrival_year.describe()
```

##### Grafica de distribucion

```python
sns.countplot(data = hotelsdf, x = 'arrival_year')
plt.xlabel('Años')
plt.ylabel('Frecuencia')
plt.title('Año de las reservas')
```

Todos los registros corresponden a los años: 2015, 2016 y 2017 siendo el año 2016 el mas frecuente entre los registros

### Average Daily Rate

Realizamos un analisis sobre la variable average daily rate

##### Valores estadisticos relevantes 

```python
hotelsdf.average_daily_rate.describe()
```

##### Grafica de distribucion

```python
sns.kdeplot(data = hotelsdf.average_daily_rate)
plt.xlabel(xlabel = 'Montos')
plt.ylabel(ylabel = 'Frecuencia')
plt.title('Distribucion del Precio promedio de renta diaria')
```

##### Outliers

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

eliminar valores con 0

```python
a_eliminar_con_cero = hotelsdf[hotelsdf['average_daily_rate'] <= 0].index
hotelsdf.drop(a_eliminar_con_cero, inplace = True)
```

##### Ajustes de valor


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
hotelsdf.reset_index()
```

```python
total_valores = len(hotelsdf.average_daily_rate)
cantidad_a_eliminar = desviacion_uno.average_daily_rate.count() + desviacion_dos.average_daily_rate.count()
print("Vamos a eliminar " + str(cantidad_a_eliminar)  + " valores ya son valores que tienen una desviacion estandar muy marcada con  respecto al resto de los valores. Ademas, estos valores representan un " +  str(cantidad_a_eliminar/total_valores) + " porcentaje del total")
```

Graficamos nuevamente con el proposito de verificar la nueva distribucion adquirida luego de la modificacion 



```python
data = hotelsdf.average_daily_rate
sns.kdeplot(data = data)
plt.xlabel(xlabel = 'Average daily rate')
plt.ylabel(ylabel = 'Frecuencia')
plt.title('Distribucion del average daily rate')
```

```python
hotelsdf.drop( columns = 'z_adr', inplace = True)
```

### babies number 

##### Valores estadisticos relevantes


```python
hotelsdf.babies_num.describe()
```

##### Grafica de distribucion

```python
eje_y = hotelsdf.babies_num.value_counts()
eje_x = eje_y.index.tolist()
sns.barplot(y = eje_y, x = eje_x, palette='Set2')
plt.xlabel('Cantidad de bebes')
plt.ylabel(ylabel='Frecuencia')
plt.title('Numero de bebes por reserva')
```

##### Outliers

```python
hotelsdf[(hotelsdf.babies_num >= 1) & (hotelsdf.adult_num < 1)]
```

##### Ajustes de valor

```python
hotelsdf.drop(hotelsdf[hotelsdf.babies_num == 9].index, inplace = True)
hotelsdf.reset_index()
```

### booking changes number 

##### Valores estadisticos relevantes

```python
hotelsdf.booking_changes_num.describe()
```

##### Grafica de distribucion

```python
eje_y = hotelsdf.booking_changes_num.value_counts()
eje_x = eje_y.index.tolist()
sns.barplot(y = eje_y, x = eje_x, palette='Set2')
plt.xlabel('Numero de cambios')
plt.ylabel(ylabel='Frecuencia')
plt.title('Cantidad de cambios por reserva')
```
##### Outliers

```python

```

##### Ajustes de valor

```python

```

### children number 

##### Valores estadisticos relevantes
##### Grafica de distribucion
##### Outliers
##### Ajustes de valor

### days in the waiting list 

##### Valores estadisticos relevantes
##### Grafica de distribucion
##### Outliers
##### Ajustes de valor

### lead time 

##### Valores estadisticos relevantes
##### Grafica de distribucion
##### Outliers
##### Ajustes de valor

### previous booking not cancelled number

##### Valores estadisticos relevantes
##### Grafica de distribucion
##### Outliers
##### Ajustes de valor

### previous booking cancellation number

##### Valores estadisticos relevantes
##### Grafica de distribucion
##### Outliers
##### Ajustes de valor

### required car space number 

##### Valores estadisticos relevantes
##### Grafica de distribucion
##### Outliers
##### Ajustes de valor

### reservation status date 

##### Valores estadisticos relevantes
##### Grafica de distribucion
##### Outliers
##### Ajustes de valor

### special request number 

##### Valores estadisticos relevantes
##### Grafica de distribucion
##### Outliers
##### Ajustes de valor

### weekend nights number

##### Valores estadisticos relevantes
##### Grafica de distribucion
##### Outliers
##### Ajustes de valor

### week nights number 

##### Valores estadisticos relevantes
##### Grafica de distribucion
##### Outliers
##### Ajustes de valor

## Medicion de la correlacion entre las variables cuantitativas

Una vez hecho el tratado sobre outliers y datos faltantes se mide la correlacion entre las variables cuantitativas encontradas en el dataframe 

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

## Cualitativas

Variables cualitativas

En un principio establecemos una lista que contenga todas las variables cualitativas

```python
cualitativas = [
"agent_id",
"arrival_month",
"assigned_room_type",
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
"reservation_status",
"reserved_room_type",
]
```

Observamos de manerea rapida los posibles valores que pueden tomar dichas variables


```python 
for variable in cualitativas:
  print("Variable: " + variable)
  print(hotelsdf[variable].value_counts().index.tolist())
  print()
```

## Valores Nulos Faltante

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

De la observación anterior se concluye que la variable company id, no proporciona información suficiente y al tener mas del 90% de sus valores nulos conviene descartarla

```python 
hotelsdf.drop("company_id", axis=1, inplace=True)
```

### Agent ID

##### Ajuste de valores faltantes

Reemplazamos valores faltantes por 0 ya que no existe previamente y servira para regular los tipos de atos de la columna

```python
hotelsdf.loc[hotelsdf['agent_id'].isnull(), 'agent_id'] = 0
hotelsdf[hotelsdf.agent_id.isnull()]
hotelsdf['agent_id'] = hotelsdf['agent_id'].astype(int)
```

##### Grafica de distribucion

```python
cantidad = len(hotelsdf['agent_id'].value_counts().index.tolist())
print(f"La cantidad de identificaciones de empresa es: {cantidad}")
```

Debido a que existen 295 id de empresas, graficamos un muestreo de los 10 ids mas frecuentes en el dataframe

```python
data = hotelsdf.agent_id.value_counts().sort_values(ascending=False).head(10)
sns.barplot(y = data, x = data.index.tolist())
#detallar 
```

##### Outliers
##### Ajustes de valor

### arrival month
##### Grafica de distribucion

```python
eje_y = hotelsdf.arrival_month.value_counts()
eje_x = eje_y.index.tolist()
plt.figure(figsize=(8,5))
plt.xticks(rotation=45)
sns.barplot(x = eje_x, y = eje_y)
```

##### Outliers
##### Ajustes de valor

### Assigned Room type

```python
eje_y = hotelsdf.assigned_room_type.value_counts()
eje_x = eje_y.index.tolist()
plt.figure(figsize=(8,5))
sns.barplot(x = eje_x, y = eje_y)
```
##### Grafica de distribucion
##### Outliers
##### Ajustes de valor

### Country
##### Grafica de distribucion
```python
data = hotelsdf.country.value_counts().sort_values(ascending=False).head(20)
plt.xticks(rotation=45)
sns.barplot(y = data, x = data.index.tolist())
```
##### Ajuste de valores faltantes
##### Outliers
##### Ajustes de valor

### Custemer type
##### Grafica de distribucion

```python
eje_y = hotelsdf.customer_type.value_counts()
eje_x = eje_y.index.tolist()
sns.barplot(x = eje_x, y = eje_y)
```

##### Outliers
##### Ajustes de valor

### Deposit type
##### Grafica de distribucion
```python
eje_y = hotelsdf.deposit_type.value_counts()
eje_x = eje_y.index.tolist()
sns.barplot(x = eje_x, y = eje_y)
```
##### Outliers
##### Ajustes de valor

### Distribution channel
##### Grafica de distribucion
```python
eje_y = hotelsdf.distribution_channel.value_counts()
eje_x = eje_y.index.tolist()
sns.barplot(x = eje_x, y = eje_y)
```
##### Outliers
##### Ajustes de valor

### Hotel Name
##### Grafica de distribucion
```python
eje_y = hotelsdf.hotel_name.value_counts()
eje_x = eje_y.index.tolist()
sns.barplot(x = eje_x, y = eje_y)
```
##### Outliers
##### Ajustes de valor

### Is canceled (Target)
##### Grafica de distribucion
```python
eje_y = hotelsdf.is_canceled.value_counts()
eje_x = eje_y.index.tolist()
sns.barplot(x = eje_x, y = eje_y)
```
##### Outliers
##### Ajustes de valor

### Is repeated guest
##### Grafica de distribucion
```python
eje_y = hotelsdf.is_repeated_guest.value_counts()
eje_x = eje_y.index.tolist()
sns.barplot(x = eje_x, y = eje_y)
```
##### Outliers
##### Ajustes de valor


### Market segment
##### Grafica de distribucion
```python
eje_y = hotelsdf.market_segment_type.value_counts()
eje_x = eje_y.index.tolist()
plt.xticks(rotation=45)
sns.barplot(x = eje_x, y = eje_y)
```
##### Outliers
##### Ajustes de valor

### meal type
##### Grafica de distribucion
```python
eje_y = hotelsdf.meal_type.value_counts()
eje_x = eje_y.index.tolist()
plt.xticks(rotation=45)
sns.barplot(x = eje_x, y = eje_y)
```
##### Outliers
##### Ajustes de valor

### Reservation Status
##### Grafica de distribucion
```python
eje_y = hotelsdf.reservation_status.value_counts()
eje_x = eje_y.index.tolist()
plt.xticks(rotation=45)
sns.barplot(x = eje_x, y = eje_y)
```
##### Outliers
##### Ajustes de valor

### Reserved room type
##### Grafica de distribucion
```python
eje_y = hotelsdf.reserved_room_type.value_counts()
eje_x = eje_y.index.tolist()
plt.xticks(rotation=45)
sns.barplot(x = eje_x, y = eje_y)
```
##### Outliers
##### Ajustes de valor
