# 🍦 Pricing ML — Optimización de Precios por Zona

Proyecto de ML aplicado a una heladería multi-sucursal.  
**Objetivo:** encontrar el precio que maximiza el profit (o revenue) por producto × zona.

---

## Estructura del proyecto

```
pricing/
├── data/                     ← Coloca aquí tus archivos sales_*.xlsx + products.xlsx
├── src/
│   ├── etl.py                ← Carga y normalización de datos
│   ├── features.py           ← Feature engineering (calendario MX, temperatura, lags)
│   ├── models.py             ← LinearRegression vs LightGBM con TimeSeriesSplit CV
│   ├── optimizer.py          ← Curvas de demanda, elasticidad, precio óptimo
│   └── viz.py                ← Gráficas reutilizables
├── artifacts/                ← Modelos y resultados (se generan al correr el pipeline)
├── app.py                    ← Dashboard Streamlit
├── run_pipeline.py           ← Script principal
└── requirements.txt
```

---

## Setup rápido

```bash
# 1. Crea entorno virtual
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Instala dependencias
pip install -r requirements.txt

# 3. Coloca tus archivos en data/
#    Requeridos:  sales_ESTADO-MUN-NUM.xlsx  (uno por tienda)
#    Opcionales:  products.xlsx, stores.xlsx

# 4. Ejecuta el pipeline
python run_pipeline.py

# 5. Abre la app
streamlit run app.py
```

---

## Formato de archivos

### sales_*.xlsx (requerido)
| date       | store_code  | product_code | units | unit_price | promo | zone  |
|------------|-------------|--------------|-------|------------|-------|-------|
| 2024-01-01 | NLE-ALL-01  | hel_choc     | 238   | 26.53      | 1     | Norte |

**Columnas opcionales** (si no las incluyes, se calculan automáticamente):

| Columna         | Descripción                                |
|-----------------|--------------------------------------------|
| `temperatura`   | Temperatura del día (°C)                   |
| `es_vacaciones` | 1 si es periodo de vacaciones escolares    |
| `es_semana_santa` | 1 si cae en Semana Santa                |
| `es_quincena`   | 1 si está cerca del 15 o fin de mes       |

### products.xlsx (opcional pero recomendado)
| product_code | unit_cost | name              |
|--------------|-----------|-------------------|
| hel_choc     | 12.50     | Helado Chocolate  |

Sin `unit_cost` el modelo maximiza **revenue**. Con él, maximiza **profit**.

### stores.xlsx (opcional)
| store_code | zone     | city       |
|------------|----------|------------|
| NLE-ALL-01 | Norte    | Monterrey  |

---

## Features del modelo

| Feature            | Descripción                                   |
|--------------------|-----------------------------------------------|
| `unit_price_mean`  | Variable de decisión (precio)                 |
| `units_lag1/2/4`   | Demanda histórica (1, 2, 4 semanas atrás)     |
| `units_roll4_mean` | Media móvil 4 semanas                         |
| `price_pct_change` | Cambio porcentual en precio                   |
| `promo_days`       | Días con promoción en la semana               |
| `temperatura`      | Temperatura promedio semanal                  |
| `es_quincena`      | Indicador quincena (demanda alta)             |
| `es_vacaciones`    | Indicador vacaciones escolares MX             |
| `es_semana_santa`  | Indicador Semana Santa                        |
| `is_weekend_days`  | Días de fin de semana en la semana            |
| `week_of_year`     | Semana del año (estacionalidad)               |
| `month`            | Mes (estacionalidad)                          |
| `zone`             | Zona geográfica (OHE)                         |
| `store_code`       | Código de tienda (OHE)                        |

---

## Modelos

| Modelo              | Rol                                          |
|---------------------|----------------------------------------------|
| `LinearRegression`  | **Baseline obligatorio** — si LightGBM no le gana, el modelo complejo no sirve |
| `LightGBM`          | Modelo principal — captura interacciones y no-linealidades |

Validación: **TimeSeriesSplit** (no hay data leakage del futuro al pasado).  
Split: 80% entrenamiento / 20% test (últimas semanas).

---

## Elasticidad precio

La elasticidad se calcula numéricamente a partir de la curva de demanda estimada:

```
E(P) = (dQ/dP) × (P/Q)
```

- **E < -1**: demanda elástica — clientes sensibles al precio.
- **-1 < E < 0**: demanda inelástica — puedes subir precio con poco impacto en volumen.
- El precio óptimo que maximiza revenue siempre está donde **E = -1** (condición de Lerner).

---

## Integración con SharePoint (datos reales)

Para que las vendedoras actualicen datos desde Excel en SharePoint/OneDrive:

```python
# Instalar: pip install Office365-REST-Python-Client
from office365.sharepoint.client_context import ClientContext

ctx = ClientContext(site_url).with_credentials(...)
file = ctx.web.get_file_by_server_relative_url("/sites/heladeria/ventas/sales_NLE-ALL-01.xlsx")
with open("data/sales_NLE-ALL-01.xlsx", "wb") as f:
    file.download(f).execute_query()
```

Agrega este bloque al inicio de `run_pipeline.py` para sincronizar antes de entrenar.
