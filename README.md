
# Predicción de enfermedad cardíaca con regresión logística

Este proyecto implementa un modelo de aprendizaje supervisado en Python para predecir la presencia de enfermedades cardíacas basándose en atributos clínicos de pacientes. Utiliza un modelo de **Regresión Logística** optimizado y evalúa su rendimiento mediante validación cruzada y métricas detalladas.

## 📋 Descripción del Proyecto

El script procesa un conjunto de datos clínicos, realiza ingeniería de características (normalización y codificación), entrena un modelo predictivo y simula diagnósticos para nuevos pacientes. El objetivo es clasificar si un paciente tiene riesgo de enfermedad cardíaca (1) o no (0).

### Características Principales:
* **Preprocesamiento de Datos:**
    * Limpieza de valores nulos.
    * **One-Hot Encoding** para variables categóricas (Sexo, Tipo de Dolor, ECG, etc.).
    * **MinMax Scaling** para normalizar variables numéricas entre 0 y 1.
* **Entrenamiento:** Modelo de Regresión Logística (`liblinear`/`lbfgs` con regularización L2).
* **Validación:** `StratifiedKFold` para asegurar robustez en los resultados.
* **Evaluación:** Reporte completo con *Accuracy, Precision, Recall, F1-Score* y *ROC-AUC*.
* **Simulación:** Módulo final que predice el riesgo para un lote de 5 pacientes nuevos con datos sintéticos.

## 📂 Estructura del Dataset

El modelo espera un archivo `heart.csv` en la carpeta `Datasets/` con las siguientes columnas:

| Columna | Descripción | Tipo |
| :--- | :--- | :--- |
| `Age` | Edad del paciente | Numérico |
| `Sex` | Sexo (M/F) | Categórico |
| `ChestPainType` | Tipo de dolor de pecho (ATA, NAP, ASY, TA) | Categórico |
| `RestingBP` | Presión arterial en reposo | Numérico |
| `Cholesterol` | Colesterol sérico | Numérico |
| `FastingBS` | Azúcar en sangre en ayunas (1: > 120 mg/dl, 0: en caso contrario) | Binario |
| `RestingECG` | Resultados de electrocardiograma en reposo | Categórico |
| `MaxHR` | Frecuencia cardíaca máxima alcanzada | Numérico |
| `ExerciseAngina` | Angina inducida por ejercicio (Y/N) | Binario |
| `Oldpeak` | Depresión del ST inducida por el ejercicio | Numérico |
| `ST_Slope` | Pendiente del segmento ST (Up, Flat, Down) | Categórico |
| `HeartDisease` | **Variable Objetivo** (1: Enfermedad, 0: Normal) | Binario |

## 🛠️ Requisitos de Instalación

Para ejecutar este proyecto, necesitas Python y las siguientes librerías. Puedes instalarlas vía pip:

```
pip install pandas numpy scikit-learn
```
