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

# Pruebas

En este jupyter notebook vamos a explorar un conjunto de datos sobre reservas de hoteles y tratar de hallar un modelo que nos permita predecir si la reserva va a ser cancelada


## Exploracion Inicial


### Imports

Importamos todas las librerias que vamos a usar

```python
import pandas as pd 
import numpy as np
import sklearn as sk
import seaborn as sns
from matplotlib import pyplot as plt
```

### Cargamos la base de datos como un dataframe

Cargamos los datos en un dataframe de pandas. Creamos una copia del dataframe original y trabajamos encima de la copia

```python
hotelsDfOriginal = pd.read_csv("./hotels_train.csv")
hotelsdf = hotelsDfOriginal.copy()

print("El data frame esta compuesto por "f"{hotelsdf.shape[0]}"" filas y "f"{hotelsdf.shape[1]}"" columnas")
```

### Vistazo inicial

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
    'id':'booking_id', #chekear con el profesor
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

## Observamos variables faltantes

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


### Observamos variables faltantes de children_num
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


Creamos una lista con todas las variables cuantitativas

```python
cuantitativas = ["lead_time"
,"arrival_year"
,"arrival_week_number"
,"arrival_month_day"
,"weekend_nights_num"
,"days_in_waiting_list"
,"week_nights_num"
,"adult_num"
,"children_num"
,"babies_num"
,"previous_cancellations_num"
,"previous_bookings_not_canceled_num"
,"booking_changes_num"
,"average_daily_rate"
,"required_car_parking_spaces_num"
,"special_requests_num"
,"reservation_status_date"]

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
cualitativas = ['hotel_name',"arrival_month", 'meal_type', 'country', 'market_segment_type', 'distribution_channel', 'is_repeated_guest', 'reserved_room_type', 'assigned_room_type', 'deposit_type', 'customer_type', 'reservation_status']

#no tiene sentido imprimir cosas como id, company, deposite_type

for variable in cualitativas:
  print("Variable: " + variable)
  print(hotelsdf[variable].value_counts().index.tolist())
  print()
```



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

```python
hotelsdf["company_id"].value_counts()
```
