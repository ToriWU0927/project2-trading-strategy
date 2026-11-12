"""Minimal pure-Python Parquet reader (flat schemas only).

Written because neither pyarrow nor fastparquet is installed and the
environment has no network. Supports: thrift compact footer, data page
V1/V2, dictionary pages, PLAIN / PLAIN_DICTIONARY / RLE_DICTIONARY
encodings, UNCOMPRESSED / SNAPPY (via system libsnappy) / GZIP codecs,
optional (nullable) flat columns. Returns a pandas DataFrame and restores
the pandas index from the 'pandas' key-value metadata if present.
"""
import ctypes, ctypes.util, gzip, json, struct, zlib
import numpy as np
import pandas as pd

# ---------------- thrift compact protocol ----------------
class TReader:
    def __init__(self, buf, pos=0):
        self.b, self.p = buf, pos

    def byte(self):
        v = self.b[self.p]; self.p += 1; return v

    def varint(self):
        r = s = 0
        while True:
            x = self.byte(); r |= (x & 0x7F) << s; s += 7
            if not x & 0x80: return r

    def zz(self):
        n = self.varint(); return (n >> 1) ^ -(n & 1)

    def read(self, t):
        if t == 1: return True
        if t == 2: return False
        if t == 3: return struct.unpack("b", bytes([self.byte()]))[0]
        if t in (4, 5, 6): return self.zz()
        if t == 7:
            v = struct.unpack("<d", self.b[self.p:self.p + 8])[0]; self.p += 8; return v
        if t == 8:
            n = self.varint(); v = bytes(self.b[self.p:self.p + n]); self.p += n; return v
        if t in (9, 10):
            h = self.byte(); n = h >> 4; et = h & 0x0F
            if n == 15: n = self.varint()
            if et in (1, 2):
                return [self.byte() == 1 for _ in range(n)]
            return [self.read(et) for _ in range(n)]
        if t == 11:
            n = self.varint()
            if n == 0: return {}
            h = self.byte(); kt, vt = h >> 4, h & 0x0F
            return {self.read(kt): self.read(vt) for _ in range(n)}
        if t == 12: return self.struct()
        raise ValueError(f"unknown thrift type {t}")

    def struct(self):
        out, last = {}, 0
        while True:
            h = self.byte()
            if h == 0: return out
            d, t = h >> 4, h & 0x0F
            fid = last + d if d else self.zz()
            out[fid] = self.read(t); last = fid

# ---------------- codecs ----------------
_snappy = ctypes.CDLL(ctypes.util.find_library("snappy") or "libsnappy.so.1")

def snappy_decompress(data):
    n = ctypes.c_size_t()
    if _snappy.snappy_uncompressed_length(data, ctypes.c_size_t(len(data)), ctypes.byref(n)) != 0:
        raise ValueError("snappy length error")
    out = ctypes.create_string_buffer(n.value)
    if _snappy.snappy_uncompress(data, ctypes.c_size_t(len(data)), out, ctypes.byref(n)) != 0:
        raise ValueError("snappy decompress error")
    return out.raw[:n.value]

def decompress(codec, data):
    if codec == 0: return data
    if codec == 1: return snappy_decompress(data)
    if codec == 2: return gzip.decompress(data)
    raise NotImplementedError(f"codec {codec}")

# ---------------- encodings ----------------
def rle_hybrid(buf, bw, n):
    """RLE / bit-packed hybrid decoder -> np.int64 array of length n."""
    out = np.empty(n, dtype=np.int64); i = 0; r = TReader(buf)
    bytew = (bw + 7) // 8
    while i < n and r.p < len(buf):
        h = r.varint()
        if h & 1:
            groups = h >> 1; nb = groups * bw
            chunk = np.frombuffer(bytes(buf[r.p:r.p + nb]), dtype=np.uint8); r.p += nb
            bits = np.unpackbits(chunk, bitorder="little")
            cnt = groups * 8
            vals = bits[:cnt * bw].reshape(cnt, bw) @ (1 << np.arange(bw, dtype=np.int64)) if bw else np.zeros(cnt, np.int64)
            take = min(cnt, n - i); out[i:i + take] = vals[:take]; i += take
        else:
            cnt = h >> 1
            v = int.from_bytes(bytes(buf[r.p:r.p + bytew]), "little"); r.p += bytew
            take = min(cnt, n - i); out[i:i + take] = v; i += take
    return out

NP = {1: "<i4", 2: "<i8", 4: "<f4", 5: "<f8"}

def plain(buf, ptype, n):
    if ptype in NP:
        dt = np.dtype(NP[ptype]); return np.frombuffer(bytes(buf[:n * dt.itemsize]), dtype=dt).copy()
    if ptype == 0:
        return np.unpackbits(np.frombuffer(bytes(buf), np.uint8), bitorder="little")[:n].astype(bool)
    if ptype == 6:
        out, p = [], 0
        for _ in range(n):
            L = struct.unpack("<I", buf[p:p + 4])[0]; p += 4
            out.append(bytes(buf[p:p + L]).decode("utf-8")); p += L
        return np.array(out, dtype=object)
    raise NotImplementedError(f"physical type {ptype}")

# ---------------- reader ----------------
def read_column(f, cmeta, max_def):
    ptype, codec, nvals = cmeta[1], cmeta[4], cmeta[5]
    start = cmeta.get(11, cmeta[9]) if cmeta.get(11) else cmeta[9]
    f.seek(start); raw = f.read(cmeta[7])
    r = TReader(raw); dictionary = None; vals_all, mask_all = [], []
    got = 0
    while got < nvals:
        ph = r.struct(); csz, usz = ph[3], ph[2]
        body = raw[r.p:r.p + csz]; r.p += csz
        if ph[1] == 2:  # dictionary page
            dictionary = plain(decompress(codec, body), ptype, ph[7][1]); continue
        if ph[1] == 0:
            dh = ph[5]; n, enc = dh[1], dh[2]
            data = decompress(codec, body); p = 0
            if max_def > 0:
                L = struct.unpack("<I", data[:4])[0]
                defs = rle_hybrid(data[4:4 + L], 1, n); p = 4 + L
            else:
                defs = np.ones(n, np.int64)
        elif ph[1] == 3:
            dh = ph[8]; n, enc = dh[1], dh[4]
            dl, rl = dh.get(5, 0), dh.get(6, 0)
            levels = body[:rl + dl]; rest = body[rl + dl:]
            defs = rle_hybrid(levels[rl:rl + dl], 1, n) if max_def > 0 else np.ones(n, np.int64)
            data = decompress(codec, rest) if dh.get(7, True) else rest; p = 0
        else:
            raise NotImplementedError(f"page type {ph[1]}")
        nn = int(defs.sum()); payload = data[p:]
        if enc == 0:
            v = plain(payload, ptype, nn)
        elif enc in (2, 8):
            bw = payload[0]; idx = rle_hybrid(payload[1:], bw, nn); v = dictionary[idx]
        else:
            raise NotImplementedError(f"encoding {enc}")
        vals_all.append(v); mask_all.append(defs.astype(bool)); got += n
    vals = np.concatenate(vals_all); mask = np.concatenate(mask_all)
    return vals, mask, ptype

def read_parquet(path):
    with open(path, "rb") as f:
        f.seek(-8, 2); flen = struct.unpack("<I", f.read(4))[0]
        assert f.read(4) == b"PAR1"
        f.seek(-8 - flen, 2); meta = TReader(f.read(flen)).struct()
        schema = meta[2][1:]
        names = [s[4].decode() for s in schema]
        rep = [s.get(3, 0) for s in schema]          # 0 REQUIRED, 1 OPTIONAL
        logical = [s.get(10) for s in schema]
        conv = [s.get(6) for s in schema]
        kv = {d[1].decode(): d.get(2, b"").decode() for d in meta.get(5, [])}
        cols = {nm: [] for nm in names}
        for rg in meta[4]:
            for j, cc in enumerate(rg[1]):
                vals, mask, ptype = read_column(f, cc[3], 1 if rep[j] == 1 else 0)
                cols[names[j]].append((vals, mask, ptype))
    data = {}
    for j, nm in enumerate(names):
        parts = cols[nm]; ptype = parts[0][2]
        mask = np.concatenate([m for _, m, _ in parts])
        vals = np.concatenate([v for v, _, _ in parts])
        lg = logical[j] or {}
        if 8 in lg or conv[j] in (9, 10):          # TIMESTAMP
            unit = "ns"
            if 8 in lg:
                u = lg[8].get(2, {})
                unit = "ms" if 1 in u else "us" if 2 in u else "ns"
            elif conv[j] == 9: unit = "ms"
            elif conv[j] == 10: unit = "us"
            full = np.full(len(mask), np.datetime64("NaT", unit), dtype=f"datetime64[{unit}]")
            full[mask] = vals.astype(f"datetime64[{unit}]")
            data[nm] = pd.to_datetime(full)
        elif 6 in lg or conv[j] == 6 or ptype == 6:  # DATE or string
            if (6 in lg or conv[j] == 6) and ptype == 1:
                full = np.full(len(mask), np.datetime64("NaT", "D"), dtype="datetime64[D]")
                full[mask] = vals.astype("datetime64[D]"); data[nm] = pd.to_datetime(full)
            else:
                full = np.full(len(mask), None, dtype=object); full[mask] = vals; data[nm] = full
        elif ptype in (4, 5):
            full = np.full(len(mask), np.nan); full[mask] = vals; data[nm] = full
        elif ptype in (1, 2):
            if mask.all(): data[nm] = vals
            else:
                full = np.full(len(mask), np.nan); full[mask] = vals; data[nm] = full
        else:
            full = np.full(len(mask), None, dtype=object); full[mask] = vals; data[nm] = full
    df = pd.DataFrame(data)
    if "pandas" in kv:
        pm = json.loads(kv["pandas"])
        idx = [c for c in pm.get("index_columns", []) if isinstance(c, str) and c in df.columns]
        if idx:
            df = df.set_index(idx)
            df.index.names = [None if n.startswith("__index_level_") else n for n in df.index.names]
    return df, meta, kv

if __name__ == "__main__":
    import sys
    df, meta, kv = read_parquet(sys.argv[1])
    print(df.shape); print(df.head()); print(df.dtypes)
