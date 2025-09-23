📊 Indicador de Correlaciones FX

Este servicio calcula correlaciones y sincronía entre los principales pares de divisas (66 en total).
Sirve como filtro de confirmación para operaciones intradía o macro, ayudando a validar si la dirección de un par está respaldada por otros pares correlacionados.

⚙️ Cómo funciona

Descarga precios históricos de Yahoo Finance para los 66 pares.

Calcula retornos logarítmicos y mide:

Correlación (ρ) → fuerza estadística de la relación en una ventana larga (N velas).

Sincronía (%) → qué tanto esa correlación se cumple en el corto plazo (K velas).

Clasifica la relación:

strong → |ρ| ≥ rho_thr y Sync ≥ sync_thr.

medium → relación media (ej. ρ≥0.8).

weak → baja o inconsistente.

Asigna una señal global al par:

✅ strong → tiene al menos 1 correlación positiva y 1 negativa fuertes.

⚠️ medium → solo un lado fuerte.

❌ weak/na → sin confirmación clara.

🕒 Modos de análisis

Intraday (M30, ~2 días)
Útil para operaciones de pocas horas.

Macro H4 (~3 meses)
Contexto semanal y mensual.

Macro D1 (~1 año)
Tendencia de fondo y correlaciones estables.

🔧 Configuración (/config)

Puedes ajustar parámetros para intraday, macro.H4, y **macro.D1`.

Cada bloque tiene:

period: cuánto histórico descargar de Yahoo ("3d", "90d", "1y").

interval: timeframe de las velas ("30m", "4h", "1d").

N: ventana para correlación ρ → mide la fuerza de relación a lo largo de este histórico.

K: ventana para sincronía → mide si la correlación se mantiene en las últimas K velas.

rho_thr: umbral mínimo de correlación fuerte (ej. 0.8 = 80%).

sync_thr: umbral mínimo de sincronía fuerte (ej. 80 = 80%).

winsorize: si true, recorta outliers para evitar distorsiones.

🚀 Flujo de uso

(Opcional) Ajusta config → POST /config.

Inicia cálculo → POST /start.

Revisa progreso → GET /status.

Descarga resultados → GET /results.

📈 Interpretación rápida

strong (verde) = confirmación confiable.

medium (amarillo) = válido pero menos robusto.

weak/na (rojo/gris) = no usar como confirmación.

Mira tanto correlaciones positivas (se mueven juntos) como negativas (se mueven opuestos).

Para operar un par, ideal que intraday sea strong y macro no contradiga.