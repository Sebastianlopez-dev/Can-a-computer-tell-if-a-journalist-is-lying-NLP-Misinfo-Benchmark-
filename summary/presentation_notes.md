# 🎙️ Guion de Presentación: Fake News Classifier (10 min)

Este documento contiene la estructura estratégica para la presentación final del proyecto de NLP. El objetivo es demostrar rigurosidad técnica y una evolución clara desde modelos básicos hasta semánticos.

---

## 🏗️ Slide 1: El Desafío y la Estrategia (1 min)
**Título de Diapositiva:** Noticias Falsas: Del Conteo a la Semántica.

### Puntos Clave:
*   **Problema:** Detectar desinformación en un dataset de ~40k registros.
*   **Estrategia:** Comparar un enfoque clásico (TF-IDF) contra uno semántico (Word2Vec).
*   **Mensaje:** Establecer un "suelo" de rendimiento (Baseline) para medir el éxito real.

> **Speech:** "Hoy presento mi solución para el NLP Challenge. Mi enfoque no fue solo crear un clasificador de alta precisión, sino entender la evolución de las técnicas de representación de datos. Mi estrategia se divide en dos mundos: el conteo de frecuencias y el entendimiento semántico, usando un Baseline robusto para medir el éxito real de los modelos avanzados."

---

## 🍱 Slide 2: Fase 1 - La Fundación Técnica (2 min)
**Título de Diapositiva:** Pre-procesamiento y el Benchmark de Naive Bayes.

### Puntos Clave:
*   **Data Fusion:** `title` + `text` para maximizar el contexto semántico.
*   **Pipeline:** Regex, Normalización, Lematización y Stopwords.
*   **Representación:** TF-IDF con Bigramas (5,000 features).
*   **Modelo:** Naive Bayes (Baseline).

> **Speech:** "Para empezar, realicé una Document-level Segmentation fusionando titulares y cuerpo de noticia. Las noticias falsas suelen tener patrones predictivos en sus titulares (clickbait) que se correlacionan con el cuerpo. Establecí un Baseline con Naive Bayes: es el 'piso' de rendimiento. Si un modelo complejo no supera a Naive Bayes, el coste computacional no está justificado."

---

## 🧠 Slide 3: Fase 2 - La Evolución Semántica (2 min)
**Título de Diapositiva:** Word2Vec y la búsqueda del "Entendimiento".

### Puntos Clave:
*   **Concepto:** Transición de palabras como índices a palabras como conceptos.
*   **Representación:** Vectores densos (100 dimensiones).
*   **El "Juez Justo":** Logistic Regression (LR) como evaluador imparcial.

> **Speech:** "En la Fase 2, dimos el salto de las palabras como simples unidades de conteo a unidades de significado. Usé Word2Vec para capturar el contexto. Aquí tomé una decisión crítica: usé Regresión Logística como mi 'Fair Judge', ya que es capaz de manejar tanto matrices dispersas como vectores densos, permitiendo una comparación científica imparcial entre ambos enfoques."

---

## 📊 Slide 4: Visualizando lo Invisible (2 min)
**Título de Diapositiva:** Análisis de Clusters y Dimensionalidad (PCA/t-SNE).

### Puntos Clave:
*   **Análisis:** Gráficas de clusters en el notebook 2.2.
*   **Objetivo:** Ver qué es lo que la computadora "ve" realmente.

> **Speech:** "Utilicé PCA para reducir la dimensionalidad y entender cómo la computadora organiza las noticias. En las visualizaciones, observamos cómo el modelo semántico logra separar las noticias por contexto profundo, mientras que el modelo tradicional de frecuencias a veces falla al identificar sutilezas donde se repiten palabras clave pero el significado varía."

---

## 🏆 Slide 5: El Duelo de Resultados (1.5 min)
**Título de Diapositiva:** Comparativa de Métricas y Matriz de Confusión.

### Puntos Clave:
*   **Métricas:** F1-Score, Curvas ROC y AUC.
*   **Ganador:** El modelo semántico demostró mayor estabilidad y precisión.

> **Speech:** "Al comparar los modelos, observamos que el modelo semántico no solo mejoró el F1-Score, sino que mostró mayor estabilidad en el AUC de la curva ROC. Esto confirma nuestra hipótesis: en el lenguaje humano, la semántica vale más que la simple repetición de términos aislados."

---

## 🚀 Slide 6: Cierre y Escalabilidad (1.5 min)
**Título de Diapositiva:** Validación Final y Siguientes Pasos (Phase 3).

### Puntos Clave:
*   **Entregable:** Predicciones de 5 columnas para `validation_data.csv`.
*   **Futuro:** Transición hacia Transformers (BERT/NLP de última generación).

> **Speech:** "He completado las predicciones finales respetando el formato estricto de 5 columnas. Este proyecto es la base perfecta para escalar hacia Phase 3: implementar Transformers como BERT para alcanzar el estado del arte en clasificación de noticias. Muchas gracias, ¿tienen alguna pregunta técnica?"
