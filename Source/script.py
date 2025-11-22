#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Programa para predicción de enfermedad cardiovascular con regresión logística

Created on Fri Nov 21 07:45:11 2025

@author: mgel_crz
"""

# Importar librería para trabajar con data frames
import pandas as pd

# Importar numpy para trabajar con arreglos
import numpy as np

# Importar módulo para normalizar el data frame
from sklearn.preprocessing import MinMaxScaler

# Importar módulo para el modelo de regresión logística
from sklearn import linear_model

# Importar módulos para la división de los datos para entrenamiento y prueba
from sklearn.model_selection import train_test_split

# Importar módulos para validaciones del modelo
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Importar módulos para métricas de evaluación
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

# Leer data set como data frame
data = pd.read_csv('Datasets/heart.csv')

# Eliminar posibles datos nulos del data frame
data = data.dropna()

# Mostrar tipo de datos de las columnas del data frame
# print(data.info())

# Aplicar One-Hot Encoding a columnas de tipo objeto del dat frame
dummies = pd.get_dummies(data['Sex']).astype(int)
dummies2 = pd.get_dummies(data['ChestPainType']).astype(int)
dummies3 = pd.get_dummies(data['RestingECG']).astype(int)
dummies4 = pd.get_dummies(data['ExerciseAngina']).astype(int)
dummies5 = pd.get_dummies(data['ST_Slope']).astype(int)

# Eliminar columnas de tipo objeto del data frame
data = data.drop(['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope'], axis = 1)

# Normalizar columnas de entrada del data frame
scaler = MinMaxScaler()
output_column = data['HeartDisease']
columns = data.columns.drop(['HeartDisease'])
data = pd.DataFrame(scaler.fit_transform(data[columns]), columns = columns)

# Agregar columnas con One-Hot Encoding y de salida al data frame normalizado
data = data.join(dummies)
data = data.join(dummies2)
data = data.join(dummies3)
data = data.join(dummies4)
data = data.join(dummies5)
data = data.join(output_column)

# Guardar data frame normalizado como archivo CSV
data = pd.DataFrame(data, columns = data.columns)
data.to_csv('heart_normalized.csv', index = False)

# Obtener los conjuntos de datos de entrada y salida
Y = np.array(data['HeartDisease'])
X = np.array(data.drop(['HeartDisease'], axis = 1))

# Dividir los conjuntos de datos de entrada y salida normalizados en 70% para entrenamiento y 30% para prueba
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .3, random_state = 42)

# Crear y entrenar el modelo de regresión logística con el conjunto de datos de entranamiento
model = linear_model.LogisticRegression(penalty = 'l2', C = 1.0, solver = 'lbfgs', max_iter = 100, fit_intercept = True)
model.fit(X_train, Y_train)

# Validación del modelo
cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
score = cross_val_score(model, X, Y, cv = cv)
print(f"Puntajes individuales: {score}")
print(f"Promedio de exactitud: {score.mean():.2f}")
print(f"Desviación estándar: {score.std():.2f}")

# ---------------------------------------------------------
# Evaluación del modelo en el conjunto de prueba con métricas adicionales y tabla de resultados
# ---------------------------------------------------------
print("\n--- Evaluación del Modelo en el Conjunto de Prueba ---")

# Realizar predicciones en el conjunto de prueba
Y_pred = model.predict(X_test)
# Obtener probabilidades para ROC-AUC
Y_prob = model.predict_proba(X_test)[:, 1]

# Calcular métricas adecuadas para regresión logística
accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred)
recall = recall_score(Y_test, Y_pred)
f1 = f1_score(Y_test, Y_pred)
roc_auc = roc_auc_score(Y_test, Y_prob)

# Mostrar los resultados en forma de tabla
results_df = pd.DataFrame({
    'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'],
    'Valor': [accuracy, precision, recall, f1, roc_auc]
})
print(results_df.to_string(index=False, formatters={'Valor': '{:.4f}'.format}))

# Mostrar reporte de clasificación completo
print("\n--- Reporte de Clasificación ---")
print(classification_report(Y_test, Y_pred))

# ---------------------------------------------------------
# Predicción con lote de nuevos datos 
# ---------------------------------------------------------
print("\n--- Predicción para 5 Pacientes de Prueba ---")

# 1. Definir columnas esperadas por el modelo
feature_cols = data.columns.drop('HeartDisease')

# 2. Datos de los 5 pacientes (Simulación de base de datos clínica)
# Paciente 1: Riesgo bajo/medio.
# Paciente 2: Riesgo ALTO (Dolor Asintomático, ST plano, Angina inducida).
# Paciente 3: Riesgo BAJO (Mujer, dolor atípico, ST ascendente).
# Paciente 4: Riesgo ALTO (Edad avanzada, colesterol alto, presión alta).
# Paciente 5: Caso Borde (Joven pero con colesterol muy alto).

# Datos numéricos
raw_numericals_data = {
    'Age':         [45,  55,  40,  70,  35],
    'RestingBP':   [120, 140, 110, 160, 130],
    'Cholesterol': [220, 280, 180, 300, 400],
    'FastingBS':   [0,   1,   0,   1,   0],
    'MaxHR':       [140, 120, 160, 105, 150],
    'Oldpeak':     [0.0, 2.5, 0.0, 3.0, 0.5]
}

# Categorías correspondientes a cada paciente (Orden: P1, P2, P3, P4, P5)
# Usamos los códigos originales: Sexo, ChestPain, RestingECG, ExerciseAngina, ST_Slope
patients_cats = [
    ['M', 'ATA', 'Normal', 'N', 'Up'],   # Paciente 1
    ['M', 'ASY', 'ST',     'Y', 'Flat'], # Paciente 2
    ['F', 'ATA', 'Normal', 'N', 'Up'],   # Paciente 3
    ['M', 'ASY', 'LVH',    'Y', 'Down'], # Paciente 4
    ['M', 'NAP', 'Normal', 'N', 'Flat']  # Paciente 5
]

# 3. Crear DataFrame vacío para el lote de pacientes
# Inicializamos en ceros para facilitar el One-Hot Encoding
new_patients = pd.DataFrame(np.zeros((5, len(feature_cols))), columns=feature_cols)

# 4. Escalar datos numéricos en bloque
# Creamos un DF temporal solo con numericos para pasarlo al scaler
raw_numericals_df = pd.DataFrame(raw_numericals_data)
scaled_vals = scaler.transform(raw_numericals_df)

# Asignar valores escalados a las columnas correspondientes en el DF final
cols_num = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
for i, col in enumerate(cols_num):
    new_patients[col] = scaled_vals[:, i]

# 5. Asignar variables categóricas (One-Hot)
# Recorremos cada paciente y activamos (ponemos en 1) sus columnas correspondientes
for i, cats in enumerate(patients_cats):
    for cat in cats:
        if cat in new_patients.columns:
            new_patients.loc[i, cat] = 1

# 6. Realizar predicciones para todo el lote
predictions = model.predict(new_patients.values)
probabilities = model.predict_proba(new_patients.values)[:, 1]

# 7. Mostrar resultados en formato tabla
print(f"{'ID':<3} | {'Edad':<4} | {'Presión':<7} | {'Diagnóstico del Modelo':<25} | {'Probabilidad':<12}")
print("-" * 65)

for i in range(5):
    diag = "PRESENCIA (Enfermo)" if predictions[i] == 1 else "AUSENCIA (Sano)"
    prob = probabilities[i]
    # Datos visuales para referencia
    edad = raw_numericals_data['Age'][i]
    presion = raw_numericals_data['RestingBP'][i]
    
    print(f"P{i+1:<2} | {edad:<4} | {presion:<7} | {diag:<25} | {prob:.2%}")
