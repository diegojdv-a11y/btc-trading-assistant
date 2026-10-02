"""
BTC/USD Trading Assistant
--------------------------
App de Streamlit que:
1. Trae datos públicos de Binance (velas, order book) — sin API key.
2. Calcula indicadores técnicos localmente (RSI, WaveTrend/Cipher B, soportes/resistencias).
3. Pide al usuario capturas manuales (liquidation map + Open Interest de Coinglass, Bookmap)
   y fundamentales.
4. Envía todo el contexto a Claude (Anthropic API) con reglas de trading fijas.
5. Muestra la recomendación de trade (o la ausencia de una entrada clara).

Nota: el Open Interest vía API de Binance Futures (fapi.binance.com) está bloqueado de forma
permanente (HTTP 451) desde hosting en la nube (GCP/AWS), incluido Streamlit Community Cloud.
Por eso el OI se lee visualmente desde la captura de Coinglass en vez de traerse por API.
"""

import base64
import json
import os

import numpy as np
import pandas as pd
import requests
import streamlit as st
import anthropic

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

SYMBOL_SPOT = "BTCUSDT"
TIMEFRAMES = ["5m", "15m", "1h", "4h"]
BINANCE_SPOT = "https://data-api.binance.vision"  # mirror publico de solo datos, sin el geo-bloqueo de api.binance.com
CLAUDE_MODEL = "claude-sonnet-5"  # cámbialo a "claude-haiku-4-5-20251001" si quieres bajar el costo aún más

st.set_page_config(page_title="BTC/USD Trading Assistant", layout="wide")

# ---------------------------------------------------------------------------
# DATOS DE MERCADO (Binance - público, sin API key)
# ---------------------------------------------------------------------------


@st.cache_data(ttl=60)
def get_klines(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    url = f"{BINANCE_SPOT}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(
        data,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ],
    )
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    return df


@st.cache_data(ttl=30)
def get_orderbook(symbol: str, limit: int = 50) -> dict:
    url = f"{BINANCE_SPOT}/api/v3/depth"
    params = {"symbol": symbol, "limit": limit}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# INDICADORES
# ---------------------------------------------------------------------------


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def wavetrend(df: pd.DataFrame, channel_len: int = 9, avg_len: int = 12, ma_len: int = 3):
    """Reimplementación del oscilador WaveTrend (base del Cipher B / VuManChu)."""
    ap = (df["high"] + df["low"] + df["close"]) / 3
    esa = ap.ewm(span=channel_len, adjust=False).mean()
    d = (ap - esa).abs().ewm(span=channel_len, adjust=False).mean()
    ci = (ap - esa) / (0.015 * d.replace(0, np.nan))
    wt1 = ci.ewm(span=avg_len, adjust=False).mean()
    wt2 = wt1.rolling(ma_len).mean()
    return wt1, wt2


def find_swing_points(series: pd.Series, window: int = 5):
    """Devuelve índices de máximos y mínimos locales (swing highs/lows)."""
    highs, lows = [], []
    vals = series.values
    for i in range(window, len(vals) - window):
        seg = vals[i - window: i + window + 1]
        if vals[i] == seg.max():
            highs.append(i)
        if vals[i] == seg.min():
            lows.append(i)
    return highs, lows


def detect_divergence(df: pd.DataFrame, wt1: pd.Series, window: int = 5, lookback: int = 40):
    """
    Divergencia regular simple:
    - Bajista: precio hace un máximo más alto, WaveTrend hace un máximo más bajo.
    - Alcista: precio hace un mínimo más bajo, WaveTrend hace un mínimo más alto.
    Compara los dos últimos swings dentro de la ventana 'lookback'.
    """
    recent = df.tail(lookback).reset_index(drop=True)
    wt_recent = wt1.tail(lookback).reset_index(drop=True)

    price_highs, price_lows = find_swing_points(recent["close"], window)
    wt_highs, wt_lows = find_swing_points(wt_recent.bfill(), window)

    result = {"bullish_divergence": False, "bearish_divergence": False, "detail": ""}

    if len(price_highs) >= 2 and len(wt_highs) >= 2:
        p1, p2 = price_highs[-2], price_highs[-1]
        if recent["close"][p2] > recent["close"][p1] and wt_recent[p2] < wt_recent[p1]:
            result["bearish_divergence"] = True
            result["detail"] += "Precio hizo máximo más alto pero WaveTrend hizo máximo más bajo. "

    if len(price_lows) >= 2 and len(wt_lows) >= 2:
        p1, p2 = price_lows[-2], price_lows[-1]
        if recent["close"][p2] < recent["close"][p1] and wt_recent[p2] > wt_recent[p1]:
            result["bullish_divergence"] = True
            result["detail"] += "Precio hizo mínimo más bajo pero WaveTrend hizo mínimo más alto. "

    return result


def find_support_resistance(df: pd.DataFrame, window: int = 5, num_levels: int = 4, tolerance_pct: float = 0.3):
    """Encuentra zonas de soporte/resistencia agrupando swing highs/lows cercanos."""
    highs_idx, lows_idx = find_swing_points(df["high"], window)
    lows_idx2, _ = find_swing_points(df["low"], window)

    levels = list(df["high"].iloc[highs_idx]) + list(df["low"].iloc[lows_idx2])
    levels = sorted(levels)

    clusters = []
    for lvl in levels:
        placed = False
        for c in clusters:
            if abs(lvl - c["price"]) / c["price"] * 100 < tolerance_pct:
                c["touches"] += 1
                c["price"] = (c["price"] * (c["touches"] - 1) + lvl) / c["touches"]
                placed = True
                break
        if not placed:
            clusters.append({"price": lvl, "touches": 1})

    clusters.sort(key=lambda c: c["touches"], reverse=True)
    return [round(c["price"], 1) for c in clusters[:num_levels]]


# ---------------------------------------------------------------------------
# CONSTRUCCIÓN DEL CONTEXTO PARA CLAUDE
# ---------------------------------------------------------------------------


def build_market_context() -> dict:
    context = {"timeframes": {}, "data_warnings": []}

    for tf in TIMEFRAMES:
        df = get_klines(SYMBOL_SPOT, tf, limit=200)
        wt1, wt2 = wavetrend(df)
        df_rsi = rsi(df["close"])
        sr_levels = find_support_resistance(df)
        divergence = detect_divergence(df, wt1)

        context["timeframes"][tf] = {
            "last_close": round(df["close"].iloc[-1], 1),
            "rsi": round(df_rsi.iloc[-1], 1) if not pd.isna(df_rsi.iloc[-1]) else None,
            "wavetrend_wt1": round(wt1.iloc[-1], 2) if not pd.isna(wt1.iloc[-1]) else None,
            "wavetrend_wt2": round(wt2.iloc[-1], 2) if not pd.isna(wt2.iloc[-1]) else None,
            "support_resistance_levels": sr_levels,
            "cipher_b_divergence": divergence,
            "last_5_candles": df[["open_time", "open", "high", "low", "close", "volume"]]
            .tail(5)
            .assign(open_time=lambda d: d["open_time"].astype(str))
            .to_dict(orient="records"),
        }

    try:
        ob = get_orderbook(SYMBOL_SPOT, limit=50)
        bids = sorted(ob["bids"], key=lambda x: float(x[1]), reverse=True)[:5]
        asks = sorted(ob["asks"], key=lambda x: float(x[1]), reverse=True)[:5]
        context["order_book"] = {
            "top_bid_walls": [{"price": float(p), "qty": float(q)} for p, q in bids],
            "top_ask_walls": [{"price": float(p), "qty": float(q)} for p, q in asks],
            "best_bid": float(ob["bids"][0][0]),
            "best_ask": float(ob["asks"][0][0]),
        }
    except Exception as e:
        context["order_book"] = None
        context["data_warnings"].append(f"Order book no disponible ({e}).")

    return context


SYSTEM_PROMPT = """Eres un analista técnico experto en trading de BTC/USD (futuros/spot, crypto).

Recibirás un JSON con: datos técnicos multi-timeframe (precio, RSI, WaveTrend/Cipher B, soportes/
resistencias, últimas velas), order book, y notas manuales del usuario sobre liquidation map
(Coinglass), Bookmap y variables fundamentales. El liquidation map y/o Bookmap pueden llegar como
capturas de pantalla adjuntas (imágenes) en vez de texto, o además del texto — analiza esas
imágenes visualmente como parte de tu evaluación cuando estén presentes.

La captura del liquidation map de Coinglass puede incluir también un panel de "Open Interest/
Market Cap" (histograma de barras, normalmente en la parte inferior del gráfico, timeframe 1H).
Si está presente, léelo como una señal cualitativa de tendencia (subiendo, bajando, plana, o
divergiendo del precio) — NO intentes inferir un porcentaje exacto de variación a partir de la
imagen, ya que no es un dato preciso como el resto de las variables numéricas.

Nota: "order_book" puede venir como null si esa fuente no estuvo disponible al momento de la
consulta (revisa "data_warnings" para ver por qué). En ese caso, basa tu análisis en las variables
restantes disponibles, y si la ausencia de esa variable te impide alcanzar una confianza razonable,
refléjalo con una confianza más baja o con "trade_disponible": false.

REGLAS QUE DEBES SEGUIR ESTRICTAMENTE:

1. Identifica primero cuál timeframe impulsa la tesis del trade ("timeframe_setup": "5m", "15m",
   "1h" o "4h") — el que aporta la señal principal (ej: la divergencia, el soporte/resistencia
   relevante). Este campo es obligatorio siempre que trade_disponible sea true.

2. El Stop Loss debe ubicarse más allá de un nivel estructural real (swing high/low o zona de
   soporte/resistencia) de ESE MISMO timeframe (timeframe_setup) — nunca lo calcules usando la
   volatilidad o el ruido de un timeframe menor al que impulsa la tesis. Si el setup es de 4h, el
   SL respeta estructura de 4h, aunque se vea "ancho" comparado con el ruido de 5m.

3. Decide el tipo de orden ("tipo_orden"):
   - "MARKET": el precio actual ya está en una zona de entrada válida para el timeframe_setup.
   - "LIMIT": la dirección es clara pero el precio necesita retroceder a un nivel mejor (ej: pullback
     a un soporte) antes de tener una entrada válida. "entrada" es el precio gatillo, no el actual.
   - "STOP": la dirección es clara pero se necesita confirmación de ruptura antes de entrar (ej:
     quiebre de una resistencia). "entrada" es el precio gatillo, no el actual.
   Si usas LIMIT o STOP, llena "invalidacion_orden": la condición bajo la cual esa orden pendiente
   deja de tener sentido y debe cancelarse (ej: "cancelar si el precio rompe 78,200 antes de activarse
   la entrada", o "cancelar si no se activa en las próximas 6-8 horas").

4. El RR mínimo permitido es 2:1, EXCEPTO si tu confianza es 8 o más, en cuyo caso el RR mínimo baja
   a 1:1. Esto aplica igual para MARKET, LIMIT y STOP, calculado sobre el precio de entrada/gatillo.

5. NO estás obligado a dar un trade. Si el mercado no ofrece una entrada clara (ni siquiera vía
   LIMIT/STOP) con un SL/TP/RR que tenga sentido, responde con "trade_disponible": false.

6. Cuando "trade_disponible" es false, SIEMPRE llena "proxima_revision" con una condición concreta
   de cuándo o bajo qué evento volver a analizar (ej: "esperar cierre de vela 4h", "revisar tras la
   apertura de Nueva York", o si de verdad no hay nada que esperar hoy, "sin trades hoy: rango sin
   definición macro"). Nunca la dejes vacía ni genérica tipo "revisar más tarde".

7. Explica el razonamiento: qué variables pesaron más en la decisión y por qué.

8. Llena SOLO UNO de los dos bloques de texto, nunca ambos:
   - Si trade_disponible es true: llena "razonamiento" (máximo 60 palabras). Deja "razon_no_trade"
     y "proxima_revision" en null.
   - Si trade_disponible es false: llena "razon_no_trade" y "proxima_revision" (cada uno conciso,
     máximo 40 palabras). Deja "razonamiento", "timeframe_setup", "tipo_orden", "entrada",
     "invalidacion_orden", "stop_loss", "take_profit", "ratio_rr" y "confianza" en null.

Responde ÚNICAMENTE con un JSON válido (sin texto adicional antes o después), con esta forma exacta:

{
  "trade_disponible": true/false,
  "timeframe_setup": "5m" | "15m" | "1h" | "4h" | null,
  "tipo_orden": "MARKET" | "LIMIT" | "STOP" | null,
  "direccion": "LONG" | "SHORT" | null,
  "entrada": number | null,
  "invalidacion_orden": "string, solo si tipo_orden es LIMIT o STOP" | null,
  "stop_loss": number | null,
  "take_profit": number | null,
  "ratio_rr": number | null,
  "confianza": number | null,
  "razonamiento": "string, maximo 60 palabras, solo si trade_disponible es true" | null,
  "razon_no_trade": "string, solo si trade_disponible es false" | null,
  "proxima_revision": "string, solo si trade_disponible es false" | null
}
"""


def encode_uploaded_image(uploaded_file) -> dict | None:
    """Convierte un archivo subido en Streamlit a un bloque base64 listo para la API."""
    if uploaded_file is None:
        return None
    data = uploaded_file.getvalue()
    return {
        "media_type": uploaded_file.type or "image/png",
        "data": base64.b64encode(data).decode("utf-8"),
    }


def build_message_content(context: dict, liq_image: dict | None, bookmap_image: dict | None) -> list:
    content = [{"type": "text", "text": json.dumps(context, indent=2, ensure_ascii=False)}]

    if liq_image is not None:
        content.append({"type": "text", "text": "Captura adjunta: liquidation map de Coinglass."})
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": liq_image["media_type"], "data": liq_image["data"]},
        })

    if bookmap_image is not None:
        content.append({"type": "text", "text": "Captura adjunta: Bookmap."})
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": bookmap_image["media_type"], "data": bookmap_image["data"]},
        })

    return content


def call_claude(
    context: dict,
    api_key: str,
    liq_image: dict | None = None,
    bookmap_image: dict | None = None,
) -> dict:
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=8192,
        thinking={"type": "adaptive"},  # el modelo decide cuanto pensar segun la complejidad real del caso
        output_config={"effort": "high"},  # nivel de esfuerzo alto, sin limitar la profundidad de analisis
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_message_content(context, liq_image, bookmap_image)}],
    )
    raw_text = None
    for block in message.content:
        if getattr(block, "type", None) == "text":
            raw_text = block.text
            break
    if raw_text is None:
        raise ValueError("La respuesta de Claude no incluyo ningun bloque de texto.")
    raw_text = raw_text.strip()
    # Por si el modelo agrega ```json ... ``` a pesar de la instrucción
    raw_text = raw_text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw_text, strict=False)
    except json.JSONDecodeError as e:
        raise ValueError(f"No se pudo interpretar la respuesta del modelo como JSON: {e}\n\nRespuesta cruda:\n{raw_text}")


# ---------------------------------------------------------------------------
# UI - STREAMLIT
# ---------------------------------------------------------------------------

st.title("📈 BTC/USD Trading Assistant")
st.caption("Análisis técnico + order book + liquidation map + Open Interest (Coinglass) + Bookmap + fundamentales → Claude")

with st.sidebar:
    st.header("Configuración")
    api_key_input = st.text_input(
        "Anthropic API Key",
        type="password",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        help="Se guarda solo en esta sesión. En producción, configúrala como 'Secret' en Streamlit Cloud.",
    )
    st.markdown("---")
    st.markdown("**Timeframes analizados:** 5m, 15m, 1h, 4h")
    st.markdown("**Fuente de datos de mercado:** Binance (pública, gratis)")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Datos automáticos (Binance)")
    st.info("Se traen automáticamente al hacer clic en 'Analizar' más abajo. (Velas + order book. Open Interest se lee desde la captura de Coinglass →)")

with col2:
    st.subheader("2. Datos manuales")

    liq_map_file = st.file_uploader(
        "Captura de Coinglass (liquidation map + Open Interest)",
        type=["png", "jpg", "jpeg"],
        key="liq_map_file",
        help="Incluye el panel de Open Interest/Market Cap si lo tienes activado (recomendado: timeframe 1H, zoom a los últimos 2-3 días). Arrastra el archivo, o haz clic aquí y pega con Ctrl+V.",
    )
    if liq_map_file is not None:
        st.image(liq_map_file, caption="Liquidation map / Open Interest", use_container_width=True)
    liq_map_notes = st.text_area(
        "Notas adicionales sobre el liquidation map / OI (opcional)",
        placeholder="Ej: esto es de las últimas 4 horas",
        key="liq_map_notes",
    )

    bookmap_file = st.file_uploader(
        "Captura de Bookmap",
        type=["png", "jpg", "jpeg"],
        key="bookmap_file",
        help="Arrastra el archivo, o haz clic aquí y pega con Ctrl+V.",
    )
    if bookmap_file is not None:
        st.image(bookmap_file, caption="Bookmap", use_container_width=True)
    bookmap_notes = st.text_area(
        "Notas adicionales sobre Bookmap (opcional)",
        placeholder="Ej: pared de venta fuerte visible en el lado derecho",
        key="bookmap_notes",
    )

    fundamentals = st.text_area(
        "¿Algo fundamental relevante hoy?",
        placeholder="Ej: FOMC en 2 días, sin eventos macro hoy",
    )

st.markdown("---")

if st.button("🔍 Analizar y generar entrada", type="primary", use_container_width=True):
    if not api_key_input:
        st.error("Falta tu Anthropic API Key (ingrésala en la barra lateral).")
    else:
        liq_image = encode_uploaded_image(liq_map_file)
        bookmap_image = encode_uploaded_image(bookmap_file)

        with st.spinner("Trayendo datos de mercado y calculando indicadores..."):
            context = build_market_context()
            context["notas_manuales"] = {
                "liquidation_map": liq_map_notes or ("Ver captura adjunta" if liq_image else "No proporcionado"),
                "bookmap": bookmap_notes or ("Ver captura adjunta" if bookmap_image else "No proporcionado"),
                "fundamentales": fundamentals or "No proporcionado",
            }

        for warning in context.get("data_warnings", []):
            st.warning(warning)

        with st.spinner("Consultando a Claude..."):
            try:
                result = call_claude(context, api_key_input, liq_image, bookmap_image)
            except Exception as e:
                st.error(f"Error al llamar a la API de Claude: {e}")
                st.stop()

        st.markdown("## Resultado")

        if result.get("trade_disponible"):
            direccion = result["direccion"]
            tipo_orden = result.get("tipo_orden", "MARKET")
            color = "🟢" if direccion == "LONG" else "🔴"

            orden_label = {
                "MARKET": "MERCADO (entrar ahora)",
                "LIMIT": "LIMIT (pendiente, retroceso)",
                "STOP": "STOP (pendiente, ruptura)",
            }.get(tipo_orden, tipo_orden)
            st.markdown(f"**Tipo de orden:** {orden_label}  |  **Timeframe del setup:** {result.get('timeframe_setup', '—')}")

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Dirección", f"{color} {direccion}")
            c2.metric("Entrada" if tipo_orden == "MARKET" else "Precio gatillo", result["entrada"])
            c3.metric("Stop Loss", result["stop_loss"])
            c4.metric("Take Profit", result["take_profit"])
            c5.metric("RR / Confianza", f"{result['ratio_rr']}:1 — {result['confianza']}/10")

            if tipo_orden in ("LIMIT", "STOP") and result.get("invalidacion_orden"):
                st.info(f"**Invalidación de la orden pendiente:** {result['invalidacion_orden']}")

            st.markdown("### Razonamiento")
            st.write(result["razonamiento"])
        else:
            st.warning("No hay un setup de trade claro en este momento.")
            st.write(result.get("razon_no_trade", ""))
            if result.get("proxima_revision"):
                st.markdown(f"**Próxima revisión:** {result['proxima_revision']}")

        with st.expander("Ver datos crudos enviados al modelo (debug)"):
            st.json(context)
        with st.expander("Ver respuesta cruda del modelo (debug)"):
            st.json(result)
