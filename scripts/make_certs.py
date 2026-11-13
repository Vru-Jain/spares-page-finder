"""The certificates that let the tool be opened over HTTPS, so Chrome trusts its downloads.

    .venv\\Scripts\\python.exe scripts\\make_certs.py [more names or addresses ...]

Makes, in certs\\ (never in git: it holds private keys):
  ca.key, ca.crt     the office's own certificate authority, made once and kept for ten years.
                     Each PC that opens the tool trusts it once (README.md, "HTTPS");
                     SparesFinder-CA.crt is the same certificate, for handing out
  server.key/.crt    the server's certificate, signed by it, for this PC's name, its addresses
                     and localhost. Made again each time this runs (it lasts 825 days, or when
                     the PC's name or address changes); the authority stays, so nobody has to
                     trust anything again.
OpenSSL comes with Git for Windows; it is run directly, not from Git Bash, which would
rewrite the "/CN=..." arguments as paths.
"""
from __future__ import annotations

import ipaddress
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

CERTS = Path(__file__).resolve().parent.parent / "certs"
OPENSSL_PLACES = [r"C:\Program Files\Git\mingw64\bin\openssl.exe", r"C:\Program Files\Git\usr\bin\openssl.exe"]


def openssl() -> str:
    found = shutil.which("openssl") or next((p for p in OPENSSL_PLACES if Path(p).exists()), None)
    if not found:
        sys.exit("OpenSSL was not found: install Git for Windows, or put openssl.exe on the PATH")
    return found


def names(extra: list[str]) -> tuple[list[str], list[str]]:
    """(names, addresses) the server answers to: this PC's name and full name, localhost, its
    IPv4 addresses, and any given."""
    dns, ips = {socket.gethostname(), socket.getfqdn(), "localhost"}, {"127.0.0.1"}
    try:
        ips.update(a[4][0] for a in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET))
    except OSError:
        pass
    for item in extra:
        try:
            ips.add(str(ipaddress.ip_address(item)))
        except ValueError:
            dns.add(item)
    ips = {ip for ip in ips if not ip.startswith("169.254.")}   # self-assigned, no network
    return sorted(dns, key=str.lower), sorted(ips)


def run(*args: str) -> None:
    subprocess.run([openssl(), *args], check=True, cwd=CERTS, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def main(extra: list[str]) -> None:
    CERTS.mkdir(exist_ok=True)
    dns, ips = names(extra)
    if not (CERTS / "ca.crt").exists():
        # It may vouch only for this PC's names and the office's private addresses: were its key
        # ever copied, it could not be used to pass off any other website.
        allowed = [f"permitted;DNS:{n}" for n in dns] + [f"permitted;IP:{net}" for net in (
            "127.0.0.0/255.0.0.0", "10.0.0.0/255.0.0.0", "172.16.0.0/255.240.0.0", "192.168.0.0/255.255.0.0")]
        run("req", "-x509", "-newkey", "rsa:3072", "-sha256", "-days", "3650", "-nodes",
            "-keyout", "ca.key", "-out", "ca.crt", "-subj", "/CN=Spares Page Finder office CA",
            "-addext", "basicConstraints=critical,CA:TRUE", "-addext", "keyUsage=critical,keyCertSign,cRLSign",
            "-addext", "nameConstraints=critical," + ",".join(allowed))
        print("Made the office certificate authority: certs\\ca.crt")
    shutil.copy(CERTS / "ca.crt", CERTS / "SparesFinder-CA.crt")
    (CERTS / "server.ext").write_text(
        "basicConstraints=critical,CA:FALSE\n"
        "keyUsage=critical,digitalSignature,keyEncipherment\n"
        "extendedKeyUsage=serverAuth\n"
        "subjectAltName=" + ",".join([f"DNS:{n}" for n in dns] + [f"IP:{ip}" for ip in ips]) + "\n", encoding="ascii")
    run("req", "-newkey", "rsa:2048", "-nodes", "-keyout", "server.key", "-out", "server.csr",
        "-subj", f"/CN={socket.gethostname()}")
    run("x509", "-req", "-in", "server.csr", "-CA", "ca.crt", "-CAkey", "ca.key", "-CAcreateserial",
        "-out", "server.crt", "-days", "825", "-sha256", "-extfile", "server.ext")
    for leftover in ("server.csr", "server.ext"):
        (CERTS / leftover).unlink(missing_ok=True)
    print(f"Made the server's certificate for {', '.join(dns + ips)}: certs\\server.crt (825 days)")


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except subprocess.CalledProcessError as exc:
        sys.exit(f"OpenSSL failed: {exc.stderr.decode(errors='replace').strip()}")
