"""
Send each Angel One account's API traffic out of its own internet connection.

Angel One accepts orders only from the static IP registered for each SmartAPI app, and one IP can't be
registered for two apps - so Vivek's app is registered with the Jio AirFiber IP and Sethu's with the
mobile (USB tethering) IP.  In .env, per account, either the network adapter's name or its local IPv4:

    ANGEL_VIVEK_INTERFACE=Wi-Fi                    (name as in Windows "Network Connections")
    ANGEL_SETHU_INTERFACE=Ethernet 3               (the USB-tethered phone)
    # or instead:  ANGEL_SETHU_SOURCE_IP=192.168.42.129

Every request for that account is made from a socket bound to that adapter's address, so Windows sends it
out through that connection (each connection needs its own default gateway, which Windows sets up when
both are connected).  If the adapter is missing or down - phone unplugged - the account's requests are
refused instead of going out through the other connection.  Nothing set = the computer's normal route.

How: the SmartAPI SDK calls the module-level requests.request(); install() swaps that module reference in
SmartApi.smartConnect for a shim that uses the bound requests.Session set for the current thread by via().
"""
import os
import threading
import time as _time
from contextlib import contextmanager

import requests
from requests.adapters import HTTPAdapter

IPIFY = "https://api.ipify.org"
PUBLIC_IP_SECONDS = 600

_local = threading.local()
_lock = threading.Lock()
_sessions = {}           # local ip -> requests.Session bound to it
_public = {}             # local ip -> (fetched at, public ip)
_installed = {"done": False}


class RouteError(Exception):
    pass


class SourceAddressAdapter(HTTPAdapter):
    """HTTPAdapter whose connections are made from a fixed local address."""

    def __init__(self, ip, **kw):
        self.ip = ip
        super().__init__(**kw)

    def init_poolmanager(self, *args, **kw):
        kw["source_address"] = (self.ip, 0)
        return super().init_poolmanager(*args, **kw)

    def proxy_manager_for(self, *args, **kw):
        kw["source_address"] = (self.ip, 0)
        return super().proxy_manager_for(*args, **kw)


def adapters():
    """{adapter name: {"up": bool, "ipv4": [addresses]}} for this computer."""
    import psutil
    stats = psutil.net_if_stats()
    out = {}
    for name, addrs in psutil.net_if_addrs().items():
        v4 = [a.address for a in addrs if a.family.name == "AF_INET" and not a.address.startswith(("127.", "169.254."))]
        if v4:
            out[name] = {"up": bool(stats.get(name) and stats[name].isup), "ipv4": v4}
    return out


def setting(account):
    """("interface", name) / ("ip", address) / (None, None) from .env."""
    acc = account.upper()
    name = (os.environ.get(f"ANGEL_{acc}_INTERFACE") or "").strip()
    ip = (os.environ.get(f"ANGEL_{acc}_SOURCE_IP") or "").strip()
    return ("interface", name) if name else ("ip", ip) if ip else (None, None)


def source_ip(account):
    """The local address this account's requests leave from, or None for the normal route."""
    kind, value = setting(account)
    if kind is None:
        return None
    ads = adapters()
    if kind == "interface":
        a = ads.get(value)
        if not a:
            raise RouteError(f"Network adapter '{value}' (ANGEL_{account.upper()}_INTERFACE) is not connected - "
                             f"connected now: {', '.join(ads) or 'none'}.")
        if not a["up"]:
            raise RouteError(f"Network adapter '{value}' is down.")
        return a["ipv4"][0]
    if not any(value in a["ipv4"] and a["up"] for a in ads.values()):
        raise RouteError(f"No connected adapter has the address {value} (ANGEL_{account.upper()}_SOURCE_IP).")
    return value


def session_for(ip):
    with _lock:
        if ip not in _sessions:
            s = requests.Session()
            ad = SourceAddressAdapter(ip)
            s.mount("https://", ad)
            s.mount("http://", ad)
            _sessions[ip] = s
        return _sessions[ip]


class _RequestsShim:
    """Stands in for the `requests` module inside the SDK: request() goes through this thread's bound
    session when one is set (see via()), everything else is the real requests module."""

    def __getattr__(self, name):
        return getattr(requests, name)

    def request(self, method, url, **kw):
        s = getattr(_local, "session", None)
        return (s or requests).request(method, url, **kw)


def install():
    if _installed["done"]:
        return
    import SmartApi.smartConnect as sc_mod
    sc_mod.requests = _RequestsShim()
    _installed["done"] = True


@contextmanager
def via(account):
    """Within this block, SDK requests made by this thread leave through the account's connection."""
    install()
    ip = source_ip(account)
    prev = getattr(_local, "session", None)
    _local.session = session_for(ip) if ip else None
    try:
        yield ip
    finally:
        _local.session = prev


def public_ip(account, use_cache=True):
    """The public IP Angel One sees for this account (what must be registered in its SmartAPI app)."""
    ip = source_ip(account)
    key = ip or "default"
    hit = _public.get(key)
    if use_cache and hit and _time.time() - hit[0] < PUBLIC_IP_SECONDS:
        return hit[1]
    r = (session_for(ip) if ip else requests).get(IPIFY, timeout=10)
    r.raise_for_status()
    _public[key] = (_time.time(), r.text.strip())
    return _public[key][1]


def describe(account, fresh=False):
    """For the page: which connection the account uses and the public IP it shows (fresh: asked again now)."""
    kind, value = setting(account)
    out = {"setting": f"{'adapter' if kind == 'interface' else 'address'} {value}" if kind else "normal route (not set)"}
    try:
        out["local_ip"] = source_ip(account)
        out["public_ip"] = public_ip(account, use_cache=not fresh)
    except Exception as exc:
        out["error"] = str(exc)
    return out
