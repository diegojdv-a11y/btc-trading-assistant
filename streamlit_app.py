"""
BTC/USD Trading Assistant
--------------------------
App de Streamlit que:
1. Trae datos públicos de Binance (velas, order book, open interest) — sin API key.
2. Calcula indicadores técnicos localmente (RSI, WaveTrend/Cipher B, soportes/resistencias).
3. Pide al usuario datos manuales (liquidation map de Coinglass, Bookmap, fundamentales).
4. Envía todo el contexto a Claude (Anthropic API) con reglas de trading fijas.
5. Muestra la recomendación de trade (o la ausencia de una entrada clara).
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
SYMBOL_FUTURES = "BTCUSDT"
TIMEFRAMES = ["5m", "15m", "1h", "4h"]
BINANCE_SPOT = "https://data-api.binance.vision"  # mirror publico de solo datos, sin el geo-bloqueo de api.binance.com
BINANCE_FUTURES = "https://fapi.binance.com"  # sin mirror alternativo conocido; puede seguir bloqueado desde la nube
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


@st.cache_data(ttl=30)
def get_open_interest(symbol: str) -> dict:
    url = f"{BINANCE_FUTURES}/fapi/v1/openInterest"
    r = requests.get(url, params={"symbol": symbol}, timeout=10)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=60)
def get_oi_history(symbol: str, period: str = "1h", limit: int = 30) -> list:
    url = f"{BINANCE_FUTURES}/futures/data/openInterestHist"
    params = {"symbol": symbol, "period": period, "limit": limit}
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

    try:
        oi_now = get_open_interest(SYMBOL_FUTURES)
        oi_hist = get_oi_history(SYMBOL_FUTURES, period="1h", limit=24)
        oi_change_pct = None
        if len(oi_hist) >= 2:
            first = float(oi_hist[0]["sumOpenInterest"])
            last = float(oi_hist[-1]["sumOpenInterest"])
            oi_change_pct = round((last - first) / first * 100, 2) if first else None

        context["open_interest"] = {
            "current": float(oi_now["openInterest"]),
            "change_24h_pct": oi_change_pct,
        }
    except Exception as e:
        context["open_interest"] = None
        context["data_warnings"].append(
            f"Open Interest no disponible ({e}). Probable bloqueo geografico de Binance Futures "
            "desde el servidor de hosting; analiza con las variables restantes."
        )

    return context


SYSTEM_PROMPT = """Eres un analista técnico experto en trading de BTC/USD (futuros/spot, crypto).

Recibirás un JSON con: datos técnicos multi-timeframe (precio, RSI, WaveTrend/Cipher B, soportes/
resistencias, últimas velas), order book, open interest, y notas manuales del usuario sobre
liquidation map (Coinglass), Bookmap y variables fundamentales. El liquidation map y/o Bookmap
pueden llegar como capturas de pantalla adjuntas (imágenes) en vez de texto, o además del texto —
analiza esas imágenes visualmente como parte de tu evaluación cuando estén presentes.

Nota: "order_book" y "open_interest" pueden venir como null si esa fuente no estuvo disponible al
momento de la consulta (revisa "data_warnings" para ver cuál). En ese caso, basa tu análisis en las
variables restantes disponibles, y si la ausencia de esa variable te impide alcanzar una confianza
razonable, refléjalo con una confianza más baja o con "trade_disponible": false.

REGLAS QUE DEBES SEGUIR ESTRICTAMENTE:

1. Si decides dar un trade, debe incluir: dirección (LONG/SHORT), precio de entrada, Stop Loss,
   Take Profit, Ratio Riesgo/Beneficio (RR) y una nota de confianza de 1 a 10.
2. El RR mínimo permitido es 2:1, EXCEPTO si tu nota de confianza es 8 o más, en cuyo caso el RR
   mínimo permitido baja a 1:1.
3. NO estás obligado a dar un trade. Si el mercado no ofrece una entrada clara con un SL/TP/RR que
   tenga sentido, responde indicando que NO hay setup válido en este momento y explica por qué
   (ej: rango sin definición, señales contradictorias entre timeframes, liquidez insuficiente, etc).
4. Explica el razonamiento: qué variables pesaron más en la decisión y por qué.
5. Nunca fuerces una operación solo por dar una respuesta.
6. Llena SOLO UNO de los dos campos de texto, nunca ambos:
   - Si trade_disponible es true: llena "razonamiento" (máximo 60 palabras) y deja "razon_no_trade" en null.
   - Si trade_disponible es false: llena "razon_no_trade" (máximo 60 palabras) y deja "razonamiento" en null.

Responde ÚNICAMENTE con un JSON válido (sin texto adicional antes o después), con esta forma exacta:

{
  "trade_disponible": true/false,
  "direccion": "LONG" | "SHORT" | null,
  "entrada": number | null,
  "stop_loss": number | null,
  "take_profit": number | null,
  "ratio_rr": number | null,
  "confianza": number | null,
  "razonamiento": "string explicando el por qué, citando las variables clave",
  "razon_no_trade": "string, solo si trade_disponible es false"
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
        max_tokens=4096,
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
st.caption("Análisis técnico + order book + open interest + liquidation map + Bookmap + fundamentales → Claude")

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
    st.info("Se traen automáticamente al hacer clic en 'Analizar' más abajo.")

with col2:
    st.subheader("2. Datos manuales")

    liq_map_file = st.file_uploader(
        "Captura del liquidation map de Coinglass",
        type=["png", "jpg", "jpeg"],
        key="liq_map_file",
        help="Arrastra el archivo, o haz clic aquí y pega con Ctrl+V.",
    )
    if liq_map_file is not None:
        st.image(liq_map_file, caption="Liquidation map", use_container_width=True)
    liq_map_notes = st.text_area(
        "Notas adicionales sobre el liquidation map (opcional)",
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
            color = "🟢" if direccion == "LONG" else "🔴"
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Dirección", f"{color} {direccion}")
            c2.metric("Entrada", result["entrada"])
            c3.metric("Stop Loss", result["stop_loss"])
            c4.metric("Take Profit", result["take_profit"])
            c5.metric("RR / Confianza", f"{result['ratio_rr']}:1 — {result['confianza']}/10")
            st.markdown("### Razonamiento")
            st.write(result["razonamiento"])
        else:
            st.warning("No hay un setup de trade claro en este momento.")
            st.write(result.get("razon_no_trade", result.get("razonamiento", "")))

        with st.expander("Ver datos crudos enviados al modelo (debug)"):
            st.json(context)
        with st.expander("Ver respuesta cruda del modelo (debug)"):
            st.json(result)
