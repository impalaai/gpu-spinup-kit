#!/usr/bin/env python3
# nonloop_socks.py <pid> — print "fd ip" for sockets whose LOCAL address is a specific
# non-loopback IP (they pin the dump-time box IP and kill cross-boot CRIU restore).
# Reconstruction of the nebius golden-disk helper (original lost; logic per its call site).
import sys, os

pid = sys.argv[1]

def parse_proc_net(path, v6):
    m = {}
    try:
        with open(path) as f:
            next(f)
            for line in f:
                p = line.split()
                local, inode = p[1], p[9]
                addr, _ = local.split(":")
                m[inode] = (addr, v6)
    except FileNotFoundError:
        pass
    return m

inodes = {}
for f, v6 in (("tcp", False), ("tcp6", True), ("udp", False), ("udp6", True)):
    inodes.update(parse_proc_net(f"/proc/net/{f}", v6))

def is_loop_or_any(addr, v6):
    if not v6:
        # little-endian hex IPv4
        b = bytes.fromhex(addr)[::-1]
        return b == b"\x00\x00\x00\x00" or b[0] == 127
    # IPv6: 4 little-endian 32-bit groups
    raw = bytes.fromhex(addr)
    groups = [raw[i:i+4][::-1] for i in range(0, 16, 4)]
    b = b"".join(groups)
    if b == b"\x00" * 16:            # ::
        return True
    if b == b"\x00" * 15 + b"\x01":  # ::1
        return True
    if b[:10] == b"\x00" * 10 and b[10:12] == b"\xff\xff":  # v4-mapped
        return b[12] == 127 or b[12:16] == b"\x00\x00\x00\x00"
    return False

def fmt(addr, v6):
    if not v6:
        return ".".join(str(x) for x in bytes.fromhex(addr)[::-1])
    raw = bytes.fromhex(addr)
    groups = [raw[i:i+4][::-1] for i in range(0, 16, 4)]
    b = b"".join(groups)
    return ":".join(f"{(b[i]<<8)|b[i+1]:x}" for i in range(0, 16, 2))

fddir = f"/proc/{pid}/fd"
for fd in os.listdir(fddir):
    try:
        link = os.readlink(os.path.join(fddir, fd))
    except OSError:
        continue
    if not link.startswith("socket:["):
        continue
    inode = link[8:-1]
    if inode not in inodes:
        continue
    addr, v6 = inodes[inode]
    if not is_loop_or_any(addr, v6):
        print(fd, fmt(addr, v6))
