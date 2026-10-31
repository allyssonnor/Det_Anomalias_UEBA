# tests/test_causality.py
#
# Auditoria programática dos invariantes metodológicos do detector.
# Rode com: pytest tests/test_causality.py -v
#
# Se algum teste falhar, o detector NÃO está pronto para a defesa.

import copy
import pytest
import numpy as np
import pandas as pd

from core.features.feature_processor import FeatureProcessor


def _config():
    return {
        "data": {"target_col": "Is_Anomaly"},
        "features": {"enabled": True},
    }


def _df(rows):
    return pd.DataFrame(rows)


def _benign(user, t, event=4624, host="h", extra=None):
    row = {"UserID": user, "Time": t, "EventID": event,
           "LogHost": host, "Is_Anomaly": 0}
    if extra:
        row.update(extra)
    return row


# ---------- TESTE 1: transform não pode reajustar ----------
def test_transform_requires_fit():
    fp = FeatureProcessor(_config(), strict=True)
    df = _df([_benign("u", 0)])
    with pytest.raises(RuntimeError, match=r"sem \.fit\(\)"):
        fp.transform(df)


# ---------- TESTE 2: inferência não muda o estado ajustado ----------
def test_transform_does_not_mutate_fitted_state():
    train = _df([_benign("u", t, host="h") for t in range(10)])
    # dataset "novo" com padrão totalmente diferente — se o processor reajustar,
    # hour_mean/host_freq vão mudar e o teste detecta.
    test = _df([_benign("u", t, host="zzz", extra={"EventID": 4625}) for t in range(100, 110)])

    fp = FeatureProcessor(_config())
    fp.fit(train)
    baseline_before = float(fp.historical_state["hour_mean"].iloc[0])
    host_set_before = copy.deepcopy(fp.historical_state["user_hosts_seen"]["u"])

    fp.transform(test)

    assert float(fp.historical_state["hour_mean"].iloc[0]) == baseline_before
    assert fp.historical_state["user_hosts_seen"]["u"] == host_set_before
    assert "zzz" not in fp.historical_state["user_hosts_seen"]["u"]


# ---------- TESTE 3: eventos futuros não alteram features passadas ----------
def test_future_events_do_not_affect_past_features():
    prefix = [_benign("u", t, event=4625, host="h") for t in range(10)]
    suffix_a = [_benign("u", t, host="h") for t in range(10, 20)]
    suffix_b = [_benign("u", t, host="g", extra={"EventID": 4625}) for t in range(10, 20)]

    fp = FeatureProcessor(_config())
    fp.fit(_df(prefix))

    out_a = fp.transform(_df(prefix + suffix_a))
    out_b = fp.transform(_df(prefix + suffix_b))

    prefix_a = out_a[out_a["Time"] < 10].reset_index(drop=True)
    prefix_b = out_b[out_b["Time"] < 10].reset_index(drop=True)

    # Todo feature dos eventos com Time<10 deve ser idêntico em ambos os futuros.
    pd.testing.assert_frame_equal(prefix_a, prefix_b)


# ---------- TESTE 4: mudança de host é detectada mesmo com frequências iguais ----------
def test_host_change_detected_with_equal_frequencies():
    rows = [
        _benign("u", 1, host="A"),
        _benign("u", 2, host="B"),
        _benign("u", 3, host="A"),
        _benign("u", 4, host="B"),
    ]
    fp = FeatureProcessor(_config())
    fp.fit(_df(rows))
    out = fp.transform(_df(rows))

    # Frequências de A e B são 0.5 cada. Se host_changed usasse _enc,
    # veria diff=0 e detectaria 0 mudanças. A versão corrigida vê 3.
    assert int(out["host_changed"].sum()) >= 3


# ---------- TESTE 5: rajada de falhas supera falhas esparsas ----------
def test_burst_detection():
    dense = [_benign("u", t * 60, event=4625) for t in range(5)]           # 5 falhas em 5 min
    sparse = [_benign("u", t * 86400, event=4625) for t in range(5)]       # 5 falhas em 5 dias

    fp = FeatureProcessor(_config())
    fp.fit(_df(dense + sparse))

    out_dense = fp.transform(_df(dense))
    out_sparse = fp.transform(_df(sparse))

    assert out_dense["fail_count_5m"].max() >= 4, "Rajada não ativou fail_count_5m"
    assert out_sparse["fail_count_5m"].max() <= 1, "Falhas esparsas inflaram fail_count_5m"


# ---------- TESTE 6: determinismo ----------
def test_transform_is_deterministic():
    rows = [_benign("u", t, host="h" if t % 2 else "g") for t in range(20)]
    fp = FeatureProcessor(_config())
    fp.fit(_df(rows))
    out1 = fp.transform(_df(rows))
    out2 = fp.transform(_df(rows))
    pd.testing.assert_frame_equal(out1, out2)


# ---------- TESTE 7: user_host_new dispara apenas na primeira vez ----------
def test_user_host_new():
    train = [_benign("u", t, host="HOST_A") for t in range(10)]
    test = [_benign("u", t, host="HOST_A") for t in range(10, 15)] + \
           [_benign("u", t, host="HOST_B") for t in range(15, 20)]

    fp = FeatureProcessor(_config())
    fp.fit(_df(train))
    out = fp.transform(_df(test))

    on_b = out[out["LogHost"].astype(str) == "HOST_B"] if "LogHost" in out.columns else None
    # LogHost é meta; recorre ao índice
    idx_b = [i for i, r in enumerate(test) if r["LogHost"] == "HOST_B"]
    assert all(out["user_host_new"].iloc[i] == 1 for i in idx_b)
    idx_a = [i for i, r in enumerate(test) if r["LogHost"] == "HOST_A"]
    assert all(out["user_host_new"].iloc[i] == 0 for i in idx_a)