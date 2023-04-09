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
##### Valores nulos/faltantes
##### Grafica de distribucion
##### Outliers
##### Ajustes de valor

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

## Cualitativas


### required car space number 


##### Valores estadisticos relevantes

```python
hotelsdf.required_car_parking_spaces_num.describe()
```

##### Valores nulos/faltantes

```python
print("La cantidad de valores nulos/faltantes es", hotelsdf.required_car_parking_spaces_num.isna().sum())
```

##### Grafica de distribucion

```python
sns.countplot(data = hotelsdf, x='required_car_parking_spaces_num')
plt.title("Cantidad de reservas por espacios de estacionamiento")
```

##### Outliers


Viendo el grafico podemos identificar que el numero de espacios de estacionamiento mas comun es 0, seguido por 1. 
Además encontramos algunos pocos casos en los que se reservaron 2, 3 y 8 espacios.
Estos ultimos son posibles Outliers (candidatearr????)
Sin embargo, esperamos a terminar de hacer todos los analisis univariados y luego al hacer los multivariados, compararemos esta variable contra la variable adult_num para observar si existe alguna incoherencia con la cantidad de adultos alojados en dicha reserva.

#TODO
MOVER LO DE ABAJO A ANALISIS MULTIVARIADO
Mostramos dichos registros junto con las columnas de hotel_name y adult_num para analizarlos más en detalle y determinar si alguno de ellos puede ser Oulier y por que.
Nuestro criterio para determinar que un valor es adecuado para esta variable es que haya como mucho 1 espacio de estacionamiento por adulto en la reserva.

```python
registrosDosOMasEspacios = hotelsdf[hotelsdf["required_car_parking_spaces_num"]>=2]
#PREG deberia hacer un .copy x las dudas?
display(registrosDosOMasEspacios[['hotel_name', 'adult_num', "required_car_parking_spaces_num"]].sort_values(
    by = "required_car_parking_spaces_num", 
    ascending = False
))
```

OJO, VER TEMA DE NUMERO DE REGISTROOOOO

De la tabla anterior se pueden sacar las siguientes conclusiones:
- En el resgistro n° 8269, el valor de 8 espacios de estacionamiento es claramente un Outlier ya que no es coherente que una habitacion para dos personas haya reservado esa cantidad de espacios de estacionamiento.
- EL resgistros n° 13713 con el valor de 3 espacios de estacionamiento es tambien un Outliers ya que tampoco es coherente que 2 personas hayan reservado 3 espacios de estacionamiento.
- Los registros restantes NO son Outliers ya que si contienen valores poco freciuentes, son coherentes con el criterio explicado en el parrafo de arriba.


##### Ajustes de valor


Con el analisis anteior, tomamos las siguiuentes decisiones:
- En el registro n° 8269, cambiamos el valor de required_car_parking_spaces_num por el valor mas frecuente (1) para no eliminar el registro por este simple detalle.
- En el registro n° 13713, cambiamos el valor de required_car_parking_spaces_num por el valor "2" suponiendo un error de tipeo.
- Se mantienen sin cambios el resto de los registros restantes listados arriba.

```python
#codigo para ajustar valores.
```

### special requests number 


##### Valores estadisticos relevantes

```python
hotelsdf.special_requests_num.describe() 
```

##### Valores nulos/faltanteS

```python
print("La cantidad de valores nulos/faltantes es", hotelsdf.special_requests_num.isna().sum())
```

##### Grafica de distribucion

```python
sns.countplot(data = hotelsdf, x='special_requests_num', palette='Set1')
plt.title("Reservas por cantidad de requisitos espediales")
```

```python
sns.boxplot(data=hotelsdf.special_requests_num)
plt.xlabel("Cantidad de reservas")
plt.ylabel("Canidad de requisitos especiales")
plt.title("Canidad de requisitos especiales por reserva")
plt.show()
```

##### Outliers


Viendo los graficos vemos que los valores mas frecuentes de requisitos especiales son 0 (ninguno), 1 y 2 y algunos menos con 3. Ademas hay muy pocos con 4 y 5. 
Los valores que podrian levantar sosppecha son 4 y 5.
Miramos la cantidad de registros de cada uno de ellos para ver que no sean casos puntuales.

```python
print("hay", hotelsdf[hotelsdf.special_requests_num==4].shape[0] ,"reservas con 4 requisitos especiales")
print("hay", hotelsdf[hotelsdf.special_requests_num==5].shape[0] ,"reservas con 5 requisitos especiales")
```

##### Ajustes de valor


Debido a la la cantidad de reservas para estos casos, no parcen ser casos puntuales. 
Procedemos a cmabiar la cantidad de requisitos especiales de dichos registros el valor mas frecuente

```python
media_special_requests = round(hotelsdf.special_requests_num.mean())
hotelsdf.loc[hotelsdf['special_requests_num'] >= 4, 'special_requests_num'] = media_special_requests
```

### weekend nights number


##### Valores estadisticos relevantes

```python
hotelsdf.weekend_nights_num.describe() 
```

##### Valores nulos/faltantes

```python
print("La cantidad de valores nulos/faltantes es", hotelsdf.weekend_nights_num.isna().sum())
```

##### Grafica de distribucion

```python
sns.countplot(data = hotelsdf, x='weekend_nights_num', palette='Set1')
plt.title("Reservas por cantidad de noches de fin de semana")
```

```python
sns.boxplot(data=hotelsdf.weekend_nights_num)
plt.xlabel("Cantidad de reservas")
plt.ylabel("Canidad de noches de fin de semana")
plt.title("Canidad de noches de fin de semana por reserva")
plt.show()
```

##### Outliers


Podriamos suponer como posibles outliers, reservas con muchos dias de estadia. A siple vista se puede ver que hay pocas reservas con 5 o mas noches de fin de semana de estadia. Comenzamos estudiando los valores de 9 o mas dias de fin de semana ya que equivaldrian a un minimo de 4 semanas de estadia.

```python
mayores_a_nueve = hotelsdf[hotelsdf["weekend_nights_num"]>=9]
mayores_a_nueve.shape[0]
```

```python
sns.countplot(data = mayores_a_nueve, x='weekend_nights_num', palette='Set1')
```

##### Ajustes de valor


Son solo 13 registros, es decir, representan muy poca cantidad del total. Tomamos la decision de eliminarlos para evitar que generen ruido al momento de generar el modelo.

```python
mas_de_nueve_noches_finde = hotelsdf[hotelsdf["weekend_nights_num"]>=9]
hotelsdf.drop(mas_de_nueve_noches_finde.index, inplace = True)
hotelsdf.reset_index()
```

Hasta ahora analizamos las estadias con mas de 9 noches de fin de semana (al menos un mes de esatdia)
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

Como son muchos mas registros posponemos su analisis para despues de terminar de estudiar todas las variables cuantitativas pero los dejamos marcados como posibles registros a modificar.


### week nights number 


##### Valores estadisticos relevantes

```python
hotelsdf.week_nights_num.describe() 
```

##### Valores nulos/faltantes

```python
print("La cantidad de valores nulos/faltantes es", hotelsdf.week_nights_num.isna().sum())
```

##### Grafica de distribucion

```python
sns.countplot(data = hotelsdf, x='week_nights_num', palette='Set1')
plt.title("Reservas por cantidad de noches de de semana")
```

Se puede ver que la gran mayoria de las reservas son estadias de entre ningna (0) y 5 noches de semana y en menor medida estadias de entre 6 y 10 noches de semana. 
Como en el grafico no se ven puntualmente los registros con estadias de 11 o mas noches de semana, los graficamos de nuevo para ver como se distribuyen y estudiarlos mejor

```python
mayores_a_11_noches_semana = hotelsdf[hotelsdf["week_nights_num"]>=11]
mayores_a_11_noches_semana.shape[0]
```

```python
sns.countplot(data = mayores_a_11_noches_semana, x='week_nights_num', palette='Set1')
```

Como son muchos registros y no contienen valores incoherentes posponemos su tratamiento para estudiarlos con un analisis multivariado en la siguiente seccion.


### (lo de abasjo va a multivariadoooo)

<!-- #region -->


 Comoanalizar solo las noches de estadia no es un buen indicador de la cantidad de dias totales ya que, por ejemplo, 2 dias de fin de semana pueden ser 2 dias de estadia total (solo el fin de semana) o 7 dias de estadia total (Domingo a Sabado de la siguiente semana). Por ello, esperamos a graficar dias de semana y a generar una columna con dias de estadia para analizar mejor ambas variables y recien ahi determinar si existen ouliers.
<!-- #endregion -->

Ya que consideramos que la cantidad de dias puede influir, observamos que no haya una inconsistencia en la carga de datos con relacion a la cantidad de dias de semana. Para ello, comparamos la cantidad de noches de fin de semana con las noches de semana que se quedo. Dberiamos obtener varias rectas con las siguientes condiciones:
- por cada 1 noche de fin de semana puede haber entre 0 y 5 dias de semana.
- por cada 2 noches de fin de semana puede haber entre 0 y 10 dias de semana.
Osea si n es el numero de noches de fin de semana con n par, como minimo (n/2)*5 -5  numero de dias de semana y como maximo hay (n/2)*5
Osea si n es el numero de noches de fin de semana con n impar queda determinado en (n-1/2)*5


```python
hotelsdf[hotelsdf["weekend_nights_num"]==8]
```

```python
sns.scatterplot(x=hotelsdf.weekend_nights_num,y=hotelsdf.week_nights_num)
plt.title('Dispersograma noches finde vs noches de semana')
plt.show()
```

Nos dio lo esperado. No hay datos incosistentes en cuanto a su comparacion con el numero de noches de semana.


##### Ajustes de valor


Como ya habiamos observado en la cantidad de dias de fin de semana, la mayor cantidad de gente se queda 


##### Outliers


##### Ajustes de valor


Vemos que columnas tienen variables faltantes
