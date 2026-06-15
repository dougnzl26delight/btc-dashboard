"""Warm the running Streamlit server so the FIRST real visitor gets the fast
(~3s) render instead of the ~27s cold one.

The 27s cold load is a fresh Streamlit process: one-time Python imports
(ccxt/yfinance/plotly/pandas + the core modules) + first disk-cache reads +
building the Plotly charts. It only happens on the first page load after the
server (re)starts. Streamlit runs the script when a session connects, so we
open a websocket to the server right after a restart — the server eats the cold
render for THIS throwaway session, and every real user after lands on the warm
path.

Dependency: websocket-client (already installed). Best-effort: any failure exits
0 so it never blocks the restart. Called by restart_dashboard.ps1 (background).
"""
import sys
import time

URL = "ws://localhost:8511/_stcore/stream"
TOTAL_BUDGET = 90       # hard cap (cold render + slack)
QUIET_SECS = 12         # no render message for this long => render settled


def main() -> int:
    try:
        import websocket
    except Exception:
        print("warm: websocket-client not available; skipping")
        return 0

    # Server may still be booting after a restart — retry the connect briefly.
    ws = None
    t_start = time.time()
    while time.time() - t_start < 40:
        try:
            ws = websocket.create_connection(URL, timeout=10)
            break
        except Exception:
            time.sleep(3)
    if ws is None:
        print("warm: could not connect (server not up?); skipping")
        return 0

    # Streamlit waits for the client to REQUEST the first run — send a rerun BackMsg
    # (empty ClientState => run the main script). This is what triggers the render.
    try:
        from streamlit.proto.BackMsg_pb2 import BackMsg
        bm = BackMsg()
        bm.rerun_script.SetInParent()
        ws.send_binary(bm.SerializeToString())
    except Exception as e:
        print(f"warm: could not send rerun request ({type(e).__name__})")

    ws.settimeout(QUIET_SECS)
    msgs, last = 0, time.time()
    while time.time() - t_start < TOTAL_BUDGET:
        try:
            ws.recv()
            msgs += 1
            last = time.time()
        except Exception:
            break  # quiet for QUIET_SECS => the cold render finished
    try:
        ws.close()
    except Exception:
        pass
    print(f"warm: drained {msgs} render messages in {time.time()-t_start:.0f}s "
          f"(server now warm)" if msgs else
          f"warm: 0 messages in {time.time()-t_start:.0f}s (connect didn't trigger a run)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
