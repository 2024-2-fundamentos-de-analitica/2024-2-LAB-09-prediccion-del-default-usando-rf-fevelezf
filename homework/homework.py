# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score
import gzip
import pickle
import json
import os

# Cargar los datos
train_df = pd.read_csv('files/input/train_data.csv/train_default_of_credit_card_clients.csv')
test_df = pd.read_csv('files/input/test_data.csv/test_default_of_credit_card_clients.csv')

# Renombrar la columna "default payment next month" a "default"
train_df.rename(columns={'default payment next month': 'default'}, inplace=True)
test_df.rename(columns={'default payment next month': 'default'}, inplace=True)

# Remover la columna "ID"
train_df.drop(columns=['ID'], inplace=True)
test_df.drop(columns=['ID'], inplace=True)

# Eliminar registros con información no disponible
train_df.dropna(inplace=True)
test_df.dropna(inplace=True)

# Agrupar valores de EDUCATION > 4 en la categoría "others"
train_df['EDUCATION'] = train_df['EDUCATION'].apply(lambda x: 4 if x > 4 else x)
test_df['EDUCATION'] = test_df['EDUCATION'].apply(lambda x: 4 if x > 4 else x)


# Dividir los datos en características (X) y etiquetas (y)
x_train = train_df.drop(columns=['default'])
y_train = train_df['default']
x_test = test_df.drop(columns=['default'])
y_test = test_df['default']

# Crear un pipeline para el modelo de clasificación
# Definir las columnas categóricas
categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE']

# Crear un transformador para las variables categóricas
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features)
    ], remainder='passthrough')

# Crear el pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])


# Definir los hiperparámetros a optimizar
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2]
}

# Realizar la búsqueda de hiperparámetros
grid_search = GridSearchCV(pipeline, param_grid, cv=10, scoring='balanced_accuracy')
grid_search.fit(x_train, y_train)

# Mejor modelo encontrado
best_model = grid_search.best_estimator_
print(best_model)


# Guardar el modelo comprimido
with gzip.open('files/models/model.pkl.gz', 'wb') as f:
    pickle.dump(best_model, f)


# Predecir en los conjuntos de entrenamiento y prueba
y_train_pred = best_model.predict(x_train)
y_test_pred = best_model.predict(x_test)

# Calcular las métricas
train_metrics = {
    'dataset': 'train',
    'precision': precision_score(y_train, y_train_pred),
    'balanced_accuracy': balanced_accuracy_score(y_train, y_train_pred),
    'recall': recall_score(y_train, y_train_pred),
    'f1_score': f1_score(y_train, y_train_pred)
}

test_metrics = {
    'dataset': 'test',
    'precision': precision_score(y_test, y_test_pred),
    'balanced_accuracy': balanced_accuracy_score(y_test, y_test_pred),
    'recall': recall_score(y_test, y_test_pred),
    'f1_score': f1_score(y_test, y_test_pred)
}

# Definir la ruta del archivo
output_dir = 'files/output'
output_file = os.path.join(output_dir, 'metrics.json')

# Crear la carpeta si no existe
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Guardar las métricas en el archivo JSON
with open(output_file, 'w') as f:
    json.dump([train_metrics, test_metrics], f, indent=4)


# Calcular las matrices de confusión
train_cm = confusion_matrix(y_train, y_train_pred)
test_cm = confusion_matrix(y_test, y_test_pred)

# Convertir las matrices de confusión a diccionarios
train_cm_dict = {
    'type': 'cm_matrix',
    'dataset': 'train',
    'true_0': {'predicted_0': train_cm[0, 0], 'predicted_1': train_cm[0, 1]},
    'true_1': {'predicted_0': train_cm[1, 0], 'predicted_1': train_cm[1, 1]}
}

test_cm_dict = {
    'type': 'cm_matrix',
    'dataset': 'test',
    'true_0': {'predicted_0': test_cm[0, 0], 'predicted_1': test_cm[0, 1]},
    'true_1': {'predicted_0': test_cm[1, 0], 'predicted_1': test_cm[1, 1]}
}

# Guardar las matrices de confusión en el archivo JSON
with open('files/output/metrics.json', 'a') as f:
    json.dump([train_cm_dict, test_cm_dict], f, indent=4)