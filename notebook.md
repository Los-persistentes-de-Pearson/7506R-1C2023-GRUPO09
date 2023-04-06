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

### Adult number 

Realizamos un analisis sobre la variable adult number

##### Valores estadisticos relevantes 

```python
hotelsdf.adult_num.describe()
```

##### Valores nulos/faltantes

```python
hotelsdf.adult_num.isna().sum()
```

##### Grafica de distribucion

```python
eje_x = hotelsdf.adult_num.value_counts().index.tolist()
eje_y = hotelsdf.adult_num.value_counts()
sns.barplot(x = eje_x, y = eje_y, palette = 'Set2')
plt.xlabel(xlabel = 'Cantidad de adultos')
plt.ylabel(ylabel = 'Cantidad de adulto')
plt.title('Distribucion de la variable')
```

##### Outliers

En el grafico anterior se listan todos las cantidades de adultos de los registro.
Se puede ver que exiten reservas con 0 adultos, lo cual no tiene mucho sentido.
Mostramos algunos registros para darnos una idea de cuantos son y ver si podemos obtener informacion adicional

```python
hotelsdf[hotelsdf["adult_num"]==0]
```

##### Ajustes de valor

### arrival month day

##### Valores estadisticos relevantes

```python
hotelsdf["arrival_month_day"].describe()
```

##### Valores nulos/faltantes

```python
hotelsdf.arrival_month_day.isna().sum()
```

##### Grafica de distribucion

```python
eje_x = hotelsdf.arrival_month_day.value_counts().index.tolist()
eje_y = hotelsdf.arrival_month_day.value_counts()
plt.figure(figsize = (9, 5))
plt.xlabel(xlabel = 'Dia de llegada')
sns.barplot(x = eje_x, y = eje_y, palette= 'Set2')
plt.title("Dia de llegada del mes")
plt.ylabel(ylabel = 'Frecuencia')
#Revisar si este grafico es relevante
#Aca no hay outliers
#ajustar eje x se ve apiñado
```

##### Outliers

```python

```

##### Ajustes de valor

```python

```

### arrival week number 

##### Valores estadisticos relevantes

```python
hotelsdf.arrival_week_number.describe()
```

##### Valores nulos/faltantes

```python
hotelsdf.arrival_week_number.isnull().sum()
```

##### Grafica de distribucion

```python
eje_y = hotelsdf.arrival_week_number.value_counts()
eje_x = eje_y.index.tolist()
plt.figure(figsize=(15, 5))
plt.xlabel(xlabel='Numero de la semana del año')
plt.title(label = 'Llegadas por semana del año')
sns.barplot(x = eje_x, y = eje_y, palette =  'Set2')
plt.ylabel(ylabel='Frecuencias')
```

##### Outliers
##### Ajustes de valor

### arrival year 

##### Valores estadisticos relevantes

```python
hotelsdf.arrival_year.describe()
```

##### Valores nulos/faltantes

```python
hotelsdf.arrival_year.isnull().sum()
```
##### Grafica de distribucion

```python
eje_y = hotelsdf.arrival_year.value_counts()
eje_x = eje_y.index.tolist()
sns.barplot(y = eje_y, x= eje_x, palette= 'Set2')
plt.title('Años de las reservas')
plt.ylabel(ylabel='Frecuencia')
plt.xlabel(xlabel='Años')
```

##### Outliers

```python

```

##### Ajustes de valor


```python

```

### Average Daily Rate

Realizamos un analisis sobre la variable average daily rate

##### Valores estadisticos relevantes 

```python
hotelsdf.average_daily_rate.describe()
```

##### Valores nulos/faltantes

```python
hotelsdf.average_daily_rate.isna().sum()
```

##### Grafica de distribucion

```python
data = hotelsdf.average_daily_rate
sns.kdeplot(data = data)
plt.xlabel(xlabel = 'Average daily rate')
plt.ylabel(ylabel = 'Frecuencia')
plt.title('Distribucion del average daily rate')
```

##### Outliers

Del grafico anterior se observan registros de adr los cuales tienen asignados 0, se debe estudiar a que se deben esos valores, asi como tambien tratar el valor negativo que aparece como mínimo

##### Ajustes de valor

### babies number 

##### Valores estadisticos relevantes


```python
hotelsdf.babies_num.describe()
```

##### Valores nulos/faltantes

```python
eje_y = hotelsdf.babies_num.value_counts()
eje_x = eje_y.index.tolist()
sns.barplot(y = eje_y, x = eje_x, palette='Set2')
plt.xlabel('Cantidad de bebes')
plt.ylabel(ylabel='Frecuencia')
plt.title('Numero de bebes por reserva')
```

##### Grafica de distribucion

```python

```

##### Outliers

```python
hotelsdf[hotelsdf.babies_num == 9]
```

##### Ajustes de valor

```python

```

### booking changes number 

##### Valores estadisticos relevantes

```python
hotelsdf.booking_changes_num.describe()
```

##### Valores nulos/faltantes

```python
hotelsdf.booking_changes_num.isna().sum()
```

##### Grafica de distribucion

```python

```
##### Outliers

```python
eje_y = hotelsdf.booking_changes_num.value_counts()
eje_x = eje_y.index.tolist()
sns.barplot(y = eje_y, x = eje_x, palette='Set2')
plt.xlabel('Numero de cambios')
plt.ylabel(ylabel='Frecuencia')
plt.title('Cantidad de cambios por reserva')
```

##### Ajustes de valor

```python

```

### children number 

##### Valores estadisticos relevantes

```python
hotelsdf["children_num"].describe()
```
Children number representa la cantidad de niños que fueron registrados en la reserva.\
Esta variable es **discreta**, porque representa una cantidad discreta de niños.\
Sin embargo, esta almacenada como float64 porque tiene valores faltantes.


##### Valores nulos/faltantes

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
```

```python
# Casteamos la columna de children number a int, ahora que ya no tiene los valores nana
hotelsdf = hotelsdf.astype({'children_num':'int'})
```

```python
# Corroboramos que el casteo funciono
print(hotelsdf["children_num"].dtypes)
```

##### Outliers

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

##### Ajustes de valor




Considerando que es un valor tanto mas alto que el resto, que es un unico caso y considerando que fue hecha con **2 adultos** nada mas; podemos considerar que este outlier y que lo podemos remover. 

```python
hotelsdf.drop((hotelsdf[hotelsdf["children_num"] == 10].index.values),inplace=True)
```

### days in the waiting list 

##### Valores estadisticos relevantes
##### Valores nulos/faltantes
##### Grafica de distribucion
##### Outliers
##### Ajustes de valor

### lead time 

##### Valores estadisticos relevantes
##### Valores nulos/faltantes
##### Grafica de distribucion
##### Outliers
##### Ajustes de valor

### previous booking not cancelled number

##### Valores estadisticos relevantes
##### Valores nulos/faltantes
##### Grafica de distribucion
##### Outliers
##### Ajustes de valor

### previous booking cancellation number

##### Valores estadisticos relevantes
##### Valores nulos/faltantes
##### Grafica de distribucion
##### Outliers
##### Ajustes de valor

### required car space number 

##### Valores estadisticos relevantes
##### Valores nulos/faltantes
##### Grafica de distribucion
##### Outliers
##### Ajustes de valor

### reservation status date 

##### Valores estadisticos relevantes
##### Valores nulos/faltantes
##### Grafica de distribucion
##### Outliers
##### Ajustes de valor

### special request number 

##### Valores estadisticos relevantes
##### Valores nulos/faltantes
##### Grafica de distribucion
##### Outliers
##### Ajustes de valor

### weekend nights number

##### Valores estadisticos relevantes
##### Valores nulos/faltantes
##### Grafica de distribucion
##### Outliers
##### Ajustes de valor

### week nights number 

##### Valores estadisticos relevantes
##### Valores nulos/faltantes
##### Grafica de distribucion
##### Outliers
##### Ajustes de valor

## Cualitativas


Vemos que columnas tienen variables faltantes

```python
serie_de_nans = hotelsdf.isna().sum()
serie_de_nans
```
Vemos entonces que las columnas con variables faltantes son:

```python
serie_de_nans[serie_de_nans > 0]
```

Resolvemos los variables faltantes una columna a la vez


## Observamos variables faltantes de children_num
Aca vemos que la variable children_num esta almacenada como un float. 
Vamos a ver si hay algun valor de childer con valor decimal distinto a 0.

```python
hotelsdf["children_num"].value_counts()
```
Vemos que no hay ningun valor con decimal distinto a 0, podemos castear la columna a int.


Segun la clasificacion de las variables podemos establecer lo siguiente sobre los datos extraidos del dataframe

Variables cuantitativas, entre las cuales podemos encontrar:

- lead_time "time beetwen reservation and arrival"
- arrival_date_year
- arrival_date_week_number  
- arrival_date_day_of_month
- stays_in_weekend_nights
- days_in_waiting_list
- stays_in_week_nights
- adult
- children
- babies
- previous_cancellations
- previous_bookings_not_canceled
- booking_changes
- days_in_waiting_list
- adr "average day rate"
- required_car_parking_spaces
- total_of_special_requests
- reservation_status_date

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

```python
hotelsdf[cuantitativas].describe()
```


Variables cualitativas

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
"is_repeated_guest",
"market_segment_type",
"meal_type",
"reservation_status",
"reserved_room_type",
]

#no tiene sentido imprimir cosas como id, company, deposite_type

for variable in cualitativas:
  print("Variable: " + variable)
  print(hotelsdf[variable].value_counts().index.tolist())
  print()
```
