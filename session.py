"""
HTTP Session 管理 —— 复用连接池，避免每次请求都建连
"""

import requests

_session = None


def get_session():
    global _session
    if _session is None:
        _session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=1,
        )
        _session.mount('http://', adapter)
        _session.mount('https://', adapter)
    return _session
