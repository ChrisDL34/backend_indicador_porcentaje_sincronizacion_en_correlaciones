# app.py
import threading
from typing import Dict, List, Optional, Tuple
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import numpy as np
import yfinance as yf
import pandas as pd
import logging, warnings

# ============ Silencio de yfinance / logs ruidosos ============
logging.getLogger("yfinance").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="yfinance")

app = FastAPI(title="FX Correlations (batch + inverse + USD triangulation)", version="1.3")

# ======================= Universo fijo: 66 pares =======================
PAIRS_66 = [
    "EURUSD","USDJPY","GBPUSD","USDCHF","USDSEK","USDCZK","USDHUF","USDPLN","USDDKK","USDHKD","USDSGD",
    "EURJPY","EURGBP","EURCHF","EURSEK","EURCZK","EURHUF","EURPLN","EURDKK","EURHKD","EURSGD",
    "GBPJPY","CHFJPY","SEKJPY","CZKJPY","HUFJPY","PLNJPY","DKKJPY","HKDJPY","SGDJPY",
    "GBPCHF","GBPSEK","GBPCZK","GBPHUF","GBPPLN","GBPDKK","GBPHKD","GBPSGD",
    "CHFSEK","CHFCZK","CHFHUF","CHFPLN","CHFDKK","CHFHKD","CHFSGD",
    "SEKCZK","SEKHUF","SEKPLN","SEKDKK","SEKHKD","SEKSGD",
    "CZKHUF","CZKPLN","CZKDKK","CZKHKD","CZKSGD",
    "HUFPLN","HUFDKK","HUFHKD","HUFSGD",
    "PLNDKK","PLNHKD","PLNSGD",
    "DKKHKD","DKKSGD",
    "HKDSGD",
]
PAIRS = PAIRS_66

def to_yahoo(pair: str) -> str:
    return f"{pair}=X"

def split_pair(p: str) -> Tuple[str, str]:
    return p[:3], p[3:]

# ========================== Pos/Neg map (idéntico) =========================
def build_pos_neg_map(pairs: List[str]):
    """Mapa de correlaciones solo para el universo dado (pares disponibles)."""
    info = {}
    bases = {p: split_pair(p)[0] for p in pairs}
    quotes = {p: split_pair(p)[1] for p in pairs}
    for p in pairs:
        b, q = bases[p], quotes[p]
        pos, neg = [], []
        for o in pairs:
            if o == p: 
                continue
            bo, qo = bases[o], quotes[o]
            # Positiva: comparten base o comparten cotizada
            if b == bo or q == qo:
                pos.append(o)
            # Negativa: base de uno = cotizada del otro (o viceversa)
            if b == qo or q == bo:
                neg.append(o)
        info[p] = {"positives": pos, "negatives": neg}
    return info

# ====================== Correlación + sincronía (idéntico) =================
def classify_strength(rho: float, sync_frac: float, thr_rho=0.80, thr_sync=80.0):
    """
    Clasifica fuerza:
      - strong: |ρ| ≥ thr_rho y Sync% ≥ thr_sync
      - medium: |ρ| ≥ 0.60 o Sync% ≥ 70%
      - weak: resto
    *thr_rho y thr_sync están **validados** para no bajar de 80% en la config.
    """
    arho, asyncp = abs(rho), sync_frac * 100.0
    if arho >= thr_rho and asyncp >= thr_sync:
        return "strong"
    if arho >= 0.60 or asyncp >= 70.0:
        return "medium"
    return "weak"

def correlation_and_sync(a: np.ndarray, b: np.ndarray, K: int, wins: bool=True):
    """Retorna (rho, sync_frac, (dir_a, dir_b)) o (None, None, None) si no alcanza."""
    ra = np.diff(np.log(a))
    rb = np.diff(np.log(b))
    n = min(len(ra), len(rb))
    if n < 10:
        return None, None, None
    ra, rb = ra[-n:], rb[-n:]

    if wins:
        def wz(x):
            lo, hi = np.percentile(x, 1), np.percentile(x, 99)
            return np.clip(x, lo, hi)
        ra, rb = wz(ra), wz(rb)

    rho = float(np.corrcoef(ra, rb)[0, 1])

    # Sync%: si rho>=0, miramos movimientos con mismo signo; si rho<0, signos opuestos
    sa, sb = np.sign(ra), np.sign(rb)
    hits = (sa == sb).sum() if rho >= 0 else (sa == -sb).sum()
    sync = hits / n  # 0..1

    # Direcciones recientes (sobre K)
    k = min(K, n)
    sra = ra[-k:].sum()
    srb = rb[-k:].sum()
    dir_a = "up" if sra > 0 else ("down" if sra < 0 else "flat")
    dir_b = "up" if srb > 0 else ("down" if srb < 0 else "flat")

    return rho, sync, (dir_a, dir_b)

# =========================== Config por defecto =========================
class BlockConfig(BaseModel):
    period: str
    interval: str
    N: int = Field(..., description="Barras para correlación (ρ)")
    K: int = Field(..., description="Barras para sincronía reciente")
    rho_thr: float = 0.80
    sync_thr: float = 80.0
    winsorize: bool = True

    @validator("rho_thr")
    def _rho_min_80(cls, v):
        if v < 0.80 or v > 1.0:
            raise ValueError("rho_thr debe estar en [0.80, 1.0]")
        return v

    @validator("sync_thr")
    def _sync_min_80(cls, v):
        if v < 80.0 or v > 100.0:
            raise ValueError("sync_thr debe estar en [80, 100]")
        return v

    @validator("N", "K")
    def _nk_positive(cls, v):
        if v <= 0: raise ValueError("N y K deben ser > 0")
        return v

class MacroConfig(BaseModel):
    H4: BlockConfig
    D1: BlockConfig

class FullConfig(BaseModel):
    intraday: BlockConfig
    macro: MacroConfig

# Recomendados: Intradía M30 ~2 días; Macro H4 ~3 meses; D1 ~1 año
CONFIG = FullConfig(
    intraday=BlockConfig(period="3d",  interval="30m", N=96,  K=16, rho_thr=0.80, sync_thr=80.0, winsorize=True),
    macro=MacroConfig(
        H4=BlockConfig(period="90d", interval="4h",  N=300, K=24, rho_thr=0.80, sync_thr=80.0, winsorize=True),
        D1=BlockConfig(period="1y",  interval="1d",  N=120, K=20, rho_thr=0.80, sync_thr=80.0, winsorize=True),
    )
)

# ========================= Estado del "job" único =======================
JOB = {
    "status": "idle",                         # idle | running | done | error
    "progress": {"intraday": 0.0, "macro": 0.0},
    "error": None,
    "results": None                           # dict con intraday y macro
}

# ============================== Endpoints ===============================
@app.get("/health")
def health():
    return {"ok": True, "status": JOB["status"]}

@app.get("/config", response_model=FullConfig)
def get_config():
    return CONFIG

@app.post("/config", response_model=FullConfig)
def set_config(new_cfg: FullConfig):
    global CONFIG
    if JOB["status"] == "running":
        raise HTTPException(409, "No puedes modificar la config mientras está corriendo.")
    CONFIG = new_cfg
    return CONFIG

@app.post("/start")
def start():
    if JOB["status"] == "running":
        return {"status": "running", "msg": "ya hay un cálculo en curso"}
    JOB["status"] = "running"
    JOB["progress"] = {"intraday": 0.0, "macro": 0.0}
    JOB["error"] = None
    JOB["results"] = None
    threading.Thread(target=_run_all, daemon=True).start()
    return {"status": "running"}

@app.get("/status")
def status():
    return {
        "status": JOB["status"],
        "progress": JOB["progress"],
        "error": JOB["error"]
    }

@app.get("/results")
def results():
    if JOB["status"] != "done":
        raise HTTPException(409, "Aún no hay resultados listos (status != done)")
    return JOB["results"]

# ============================== Batch Downloader ==============================
# Almacenamos las series Close en caché por (period, interval) -> {symbol: np.ndarray}
_BATCH_CACHE: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}

def _collect_needed_symbols(pairs: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """Devuelve (directos, inversos, usd_legs) en formato Yahoo '=X'."""
    directs = [to_yahoo(p) for p in pairs]
    inverses = [to_yahoo(p[3:] + p[:3]) for p in pairs]
    # patas USD: para triangulación A/B = (USD/B)/(USD/A) => necesitamos A=X y B=X
    ccys = set()
    for p in pairs:
        a, b = split_pair(p)
        ccys.add(a); ccys.add(b)
    usd_legs = [f"{c}=X" for c in ccys]  # en Yahoo, 'EUR=X' significa USD/EUR
    return directs, inverses, usd_legs

def _extract_close_from_batch(batch_df: pd.DataFrame, symbol: str) -> Optional[np.ndarray]:
    """Extrae 'Close' de un símbolo en df multiindex group_by='ticker'."""
    if batch_df is None or batch_df.empty:
        return None
    if not isinstance(batch_df.columns, pd.MultiIndex):
        return None
    lvl0 = batch_df.columns.get_level_values(0)
    if symbol not in set(lvl0):
        return None
    sub = batch_df[symbol]
    if "Close" not in sub.columns:
        return None
    ser = sub["Close"].dropna()
    if ser.empty:
        return None
    return ser.to_numpy(dtype=float)

def _inverse_series(close: np.ndarray) -> Optional[np.ndarray]:
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / close
    inv = inv[np.isfinite(inv)]
    return inv if inv.size >= 2 else None

def _align_last(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    return a[-n:], b[-n:]

def _build_pair_series_from_maps(pair: str, sym_map: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    """
    Construye A/B (Close) usando:
      1) directo A B =X
      2) inverso 1/(B A =X)
      3) triangulación USD: (B=X) / (A=X)
    """
    direct = to_yahoo(pair)
    inv = to_yahoo(pair[3:] + pair[:3])
    a, b = split_pair(pair)
    leg_a = f"{a}=X"  # USD/A
    leg_b = f"{b}=X"  # USD/B

    # 1) directo
    cdir = sym_map.get(direct)
    if cdir is not None and len(cdir) >= 2:
        return cdir

    # 2) inverso
    cinv = sym_map.get(inv)
    if cinv is not None and len(cinv) >= 2:
        invc = _inverse_series(cinv)
        if invc is not None and len(invc) >= 2:
            return invc

    # 3) triangulación USD
    ca = sym_map.get(leg_a)  # USD/A
    cb = sym_map.get(leg_b)  # USD/B
    if ca is None or cb is None or len(ca) < 2 or len(cb) < 2:
        return None
    ca, cb = _align_last(ca, cb)
    with np.errstate(divide="ignore", invalid="ignore"):
        cross = cb / ca  # (USD/B)/(USD/A) = A/B
    cross = cross[np.isfinite(cross)]
    return cross if len(cross) >= 2 else None

def _batch_download(period: str, interval: str, pairs: List[str]) -> Dict[str, np.ndarray]:
    """
    Descarga **todos** los símbolos requeridos (directos, inversos, patas USD) en una sola llamada
    para (period, interval). Devuelve un map symbol->Close ndarray.
    """
    key = (period, interval)
    if key in _BATCH_CACHE:
        return _BATCH_CACHE[key]

    directs, inverses, usd_legs = _collect_needed_symbols(pairs)
    all_syms = sorted(set(directs + inverses + usd_legs))

    # Intento principal con el periodo indicado; si algunos vienen vacíos,
    # NO repetimos llamadas por símbolo: la reconstrucción por par usará inversa/triangulación.
    df = yf.download(
        tickers=all_syms,
        period=period,
        interval=interval,
        auto_adjust=False,
        group_by="ticker",
        progress=False,
        threads=False,  # más estable
    )

    sym_map: Dict[str, np.ndarray] = {}
    for sym in all_syms:
        ser = _extract_close_from_batch(df, sym)
        if ser is not None:
            sym_map[sym] = ser

    _BATCH_CACHE[key] = sym_map
    return sym_map

def _download_series(pairs: List[str], period: str, interval: str, set_progress):
    """
    Usa batch + reconstrucción:
      - directo -> inverso -> triangulación USD
    Devuelve dict par -> np.ndarray | None
    """
    closes: Dict[str, Optional[np.ndarray]] = {}
    sym_map = _batch_download(period, interval, pairs)
    total = len(pairs)

    for i, pair in enumerate(pairs, 1):
        try:
            series = _build_pair_series_from_maps(pair, sym_map)
            closes[pair] = series if (series is not None and len(series) >= 2) else None
        except Exception:
            closes[pair] = None
        if set_progress:
            set_progress(i / total * 0.5)  # 50% del progreso es la descarga
    return closes

# ============================== Workers ================================
def _analyze_block(pairs: List[str],
                   closes: Dict[str, np.ndarray],
                   cfg: BlockConfig,
                   posneg: Dict[str, dict],
                   set_progress=None):
    """Analiza correlaciones para el universo 'pairs' usando mapa posneg (ya filtrado)."""
    results: List[Dict] = []
    total = len(pairs)

    for j, pair in enumerate(pairs, 1):
        c_main = closes.get(pair)
        if c_main is None or len(c_main) < (cfg.N + 1):
            results.append({
                "pair": pair, "positives": [], "negatives": [],
                "global_signal": "na", "reason": "sin datos suficientes"
            })
            if set_progress: set_progress(0.5 + j / total * 0.5)
            continue

        cm = c_main[-(cfg.N + 1):]  # +1 por diff
        if len(cm) < 2:
            results.append({
                "pair": pair, "positives": [], "negatives": [],
                "global_signal": "na", "reason": "series insuficientes"
            })
            if set_progress: set_progress(0.5 + j / total * 0.5)
            continue

        pos_items, neg_items = [], []
        comps_pos = posneg.get(pair, {}).get("positives", [])
        comps_neg = posneg.get(pair, {}).get("negatives", [])

        for o in comps_pos:
            co = closes.get(o)
            if co is None or len(co) < 2:
                continue
            cn = co[-len(cm):]
            rho, sync, dirs = correlation_and_sync(cm, cn, cfg.K, wins=cfg.winsorize)
            if rho is None:
                continue
            pos_items.append({
                "pair": o,
                "rho": round(rho, 4),
                "sync": round(sync * 100, 1),
                "dir_main": dirs[0],
                "dir_other": dirs[1],
                "strength": classify_strength(rho, sync, cfg.rho_thr, cfg.sync_thr)
            })

        for o in comps_neg:
            co = closes.get(o)
            if co is None or len(co) < 2:
                continue
            cn = co[-len(cm):]
            rho, sync, dirs = correlation_and_sync(cm, cn, cfg.K, wins=cfg.winsorize)
            if rho is None:
                continue
            neg_items.append({
                "pair": o,
                "rho": round(rho, 4),
                "sync": round(sync * 100, 1),
                "dir_main": dirs[0],
                "dir_other": dirs[1],
                "strength": classify_strength(rho, sync, cfg.rho_thr, cfg.sync_thr)
            })

        # Señal global
               # ===== Señal global + explicación enriquecida =====
        def _best(items, key="rho", absval=True):
            if not items:
                return None
            if absval:
                return max(items, key=lambda x: abs(x.get(key, 0.0)))
            return max(items, key=lambda x: x.get(key, 0.0))

        # Separar fuertes por lado respetando el signo
        strong_pos_items = [x for x in pos_items if x["strength"] == "strong" and x["rho"] > 0]
        strong_neg_items = [x for x in neg_items if x["strength"] == "strong" and x["rho"] < 0]

        # Mejores ejemplos
        best_pos_strong = _best(strong_pos_items)               # mejor positiva fuerte
        best_neg_strong = _best(strong_neg_items, absval=True)  # mejor negativa fuerte
        best_pos_any    = _best(pos_items)                      # mejor positiva (aunque no sea fuerte)
        best_neg_any    = _best(neg_items, absval=True)         # mejor negativa (aunque no sea fuerte)

        thr_rho  = cfg.rho_thr
        thr_sync = cfg.sync_thr

        # Determinar señal y razón detallada
        if best_pos_strong and best_neg_strong:
            glb = "strong"
            rsn = (
                f"positiva y negativa fuertes. "
                f"Ej pos: {best_pos_strong['pair']} (ρ={best_pos_strong['rho']:.2f}, "
                f"sync={best_pos_strong['sync']:.1f}%), "
                f"ej neg: {best_neg_strong['pair']} (ρ={best_neg_strong['rho']:.2f}, "
                f"sync={best_neg_strong['sync']:.1f}%). "
                f"Umbrales ≥|ρ| {thr_rho:.2f} y Sync {thr_sync:.0f}%."
            )
        elif best_pos_strong or best_neg_strong:
            # Solo un lado fuerte => medium
            glb = "medium"
            if best_pos_strong:
                side = "positivas"
                ex   = best_pos_strong
                opp  = best_neg_any
            else:
                side = "negativas"
                ex   = best_neg_strong
                opp  = best_pos_any
            rsn = (
                f"solo {side} fuertes: {ex['pair']} "
                f"(ρ={ex['rho']:.2f}, sync={ex['sync']:.1f}%). "
                f"El lado opuesto no alcanzó fuerte"
            )
            if opp:
                rsn += f" (mejor opuesto: {opp['pair']} ρ={opp['rho']:.2f}, sync={opp['sync']:.1f}%). "
            rsn += f"Buscamos Sync≥{thr_sync:.0f}% como mínimo."
        elif pos_items or neg_items:
            glb = "weak"
            # Mencionar el mejor candidato de cada lado para contexto
            parts = []
            if best_pos_any:
                parts.append(f"mejor positiva: {best_pos_any['pair']} (ρ={best_pos_any['rho']:.2f}, sync={best_pos_any['sync']:.1f}%)")
            if best_neg_any:
                parts.append(f"mejor negativa: {best_neg_any['pair']} (ρ={best_neg_any['rho']:.2f}, sync={best_neg_any['sync']:.1f}%)")
            rsn = "sin fuertes; " + (", ".join(parts) if parts else "comparables débiles/medias") + f". Umbrales fuertes: |ρ|≥{thr_rho:.2f} y Sync≥{thr_sync:.0f}%."
        else:
            glb, rsn = "na", "sin comparables válidos"

        # Paquete de explicación estructurada (útil para UI)
        explain = {
            "thresholds": {"rho_thr": thr_rho, "sync_thr": thr_sync},
            "sides": {
                "positives": {
                    "strong_examples": strong_pos_items[:3],  # top 3 para no inflar payload
                    "best_any": best_pos_any
                },
                "negatives": {
                    "strong_examples": strong_neg_items[:3],
                    "best_any": best_neg_any
                }
            }
        }

        results.append({
            "pair": pair,
            "positives": pos_items,
            "negatives": neg_items,
            "global_signal": glb,
            "reason": rsn,
            "explain": explain
        })


        if set_progress:
            set_progress(0.5 + j / total * 0.5)
    return results

def _run_all():
    try:
        pairs = PAIRS

        # -------- INTRADAY --------
        icfg = CONFIG.intraday
        def ip(p): JOB["progress"]["intraday"] = round(p, 3)
        closes_i = _download_series(pairs, period=icfg.period, interval=icfg.interval, set_progress=ip)

        # Universo disponible intradía (con N+1 barras)
        avail_i = [p for p, c in closes_i.items() if c is not None and len(c) >= (icfg.N + 1)]
        posneg_i = build_pos_neg_map(avail_i)
        res_i = _analyze_block(avail_i, closes_i, cfg=icfg, posneg=posneg_i, set_progress=ip)

        # -------- MACRO H4 --------
        h4cfg = CONFIG.macro.H4
        def mp4(p): JOB["progress"]["macro"] = round(p * 0.5, 3)
        closes_h4 = _download_series(pairs, period=h4cfg.period, interval=h4cfg.interval, set_progress=mp4)
        avail_h4 = [p for p, c in closes_h4.items() if c is not None and len(c) >= (h4cfg.N + 1)]
        posneg_h4 = build_pos_neg_map(avail_h4)
        res_h4 = _analyze_block(avail_h4, closes_h4, cfg=h4cfg, posneg=posneg_h4, set_progress=mp4)

        # -------- MACRO D1 --------
        d1cfg = CONFIG.macro.D1
        def mp1(p): JOB["progress"]["macro"] = round(0.5 + p * 0.5, 3)
        closes_d1 = _download_series(pairs, period=d1cfg.period, interval=d1cfg.interval, set_progress=mp1)
        avail_d1 = [p for p, c in closes_d1.items() if c is not None and len(c) >= (d1cfg.N + 1)]
        posneg_d1 = build_pos_neg_map(avail_d1)
        res_d1 = _analyze_block(avail_d1, closes_d1, cfg=d1cfg, posneg=posneg_d1, set_progress=mp1)

        JOB["results"] = {
            "intraday": {
                "timeframe": icfg.interval,
                "params": icfg.dict(),
                "unavailable": sorted([p for p in pairs if p not in avail_i]),
                "results": res_i
            },
            "macro": {
                "H4": {
                    "params": h4cfg.dict(),
                    "unavailable": sorted([p for p in pairs if p not in avail_h4]),
                    "results": res_h4
                },
                "D1": {
                    "params": d1cfg.dict(),
                    "unavailable": sorted([p for p in pairs if p not in avail_d1]),
                    "results": res_d1
                }
            }
        }
        JOB["status"] = "done"
        JOB["progress"] = {"intraday": 1.0, "macro": 1.0}
    except Exception as e:
        JOB["status"] = "error"
        JOB["error"] = str(e)
