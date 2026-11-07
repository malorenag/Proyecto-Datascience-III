# Proyecto-Datascience-III
Este primer checkpoint es la base sobre la cual construiremos clasificadores de texto, sistemas de traducción y modelos generativos en los próximos módulos.
Proyecto-Datascience-III

Este primer checkpoint es la base sobre la cual construiremos clasificadores de texto, sistemas de traducción y modelos generativos en los próximos módulos.

CLASIFICACION DE NOTICIAS AG NEWS CON PYTORCH

Objetivo

Construir un pipeline reproducible de entrenamiento y validación para poder clasificar las noticias en inglés en cuatro categorías utilizando una red neuronal PyTorch.

Dataset

Se utiliza AG News, un dataset de noticias en inglés con 4 categorías: Internacional, Deportes, Negocios y Ciencia/Tecnología.

Datasets de la cátedra — NLP & Deep Learning

Dos datasets de clasificación de texto supervisada, listos para usar en toda la cadena de pre-entregas (Módulos 2 → 3 → 4 → Proyecto Final). Ambos vienen limpios, balanceados o estratificados, con train/test ya separados y sin valores nulos.

Regla de oro: elegí un dataset al inicio del Módulo 2 y usá el mismo hasta el Proyecto Final. No cambies de corpus a mitad del camino.
Los dos datasets

AG News	BBC News
Idioma	Inglés	Inglés
Tarea	Clasificar noticias por tema	Clasificar noticias por sección
Clases	4 (World, Sports, Business, Sci_Tech)	5 (business, entertainment, politics, sport, tech)
Train / Test	8.000 / 2.000	1.700 / 425
Tipo de texto	Documentos cortos (mediana ≈ 38 palabras)	Artículos largos (mediana ≈ 332 palabras)
Balance	Perfectamente balanceado	Estratificado (leve desbalanceo natural)
Ideal para enseñar	max_len corto, entrenamiento rápido con LoRA	percentil-95 de max_len, truncation, desbalanceo
¿Cuál recomendar? Por defecto, AG News: es el más liviano y entrena rápido en la GPU gratuita de Colab (clave para el Módulo 4). BBC News es la mejor opción si querés que el ejercicio del percentil-95 de longitud y el análisis de desbalanceo (F1 macro vs. weighted) sean más ricos, a costa de artículos más largos.

Formato

Cada archivo es un CSV con dos columnas:

text: el texto de la noticia (ya normalizado en espacios/saltos de línea).
label: la categoría (string).
text,label
"Red Hat Appoints New CFO. September 2, 2004 ...",Business
Cómo usarlos en cada pre-entrega

Módulo 2 (EDA): cargá el train.csv, pasá preprocess_text sobre text, y hacé el EDA (histograma de longitud, N-gramas y distribución de label).
Módulo 3 (TF-IDF): TfidfVectorizer sobre text, clasificador clásico sobre label. Baseline de referencia ya verificado: F1-macro ≈ 0.90 (AG News) y ≈ 0.98 (BBC) con LogisticRegression.
Módulo 4 (LoRA): convertí label a enteros y pasá a datasets de Hugging Face (ver cargar_dataset.py). Modelos base sugeridos: distilbert-base-uncased o bert-base-uncased (ambos en inglés, acordes al idioma del corpus).
Proyecto Final: compará el baseline TF-IDF contra el Transformer + LoRA sobre el mismo test.csv.
Ver cargar_dataset.py para el snippet de carga (pandas y Hugging Face).

Requisito de entorno (Módulo 4)

El fine-tuning necesita GPU. Usá Google Colab o Kaggle (GPU gratuita). Verificá con:

import torch; print(torch.cuda.is_available())
Fuentes y licencia

AG News: corpus de AG (Antonio Gulli), de uso académico estándar (Zhang et al., 2015). Muestra balanceada y recortada por la cátedra.
Se distribuye aquí solo con fines educativos.