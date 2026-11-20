#!/usr/bin/env python3
"""Bounded binary array transport with explicit metadata and exact reads."""
import json
import numpy as np


def write_all(stream,raw):
    view=memoryview(raw).cast('B');done=0
    while done<len(view):
        n=stream.write(view[done:])
        if not n:raise ValueError('Incomplete feature stream write')
        done+=n


def read_exact(stream,count):
    result=bytearray(count);view=memoryview(result);done=0
    while done<count:
        n=stream.readinto(view[done:])
        if not n:raise ValueError('Truncated original feature stream')
        done+=n
    return result


def write_json(stream,value):
    raw=json.dumps(value,separators=(',',':')).encode()
    if len(raw)>2**20:raise ValueError('Bounded stream metadata required')
    write_all(stream,len(raw).to_bytes(8,'little'));write_all(stream,raw)


def read_json(stream):
    size=int.from_bytes(read_exact(stream,8),'little')
    if not 0<size<=2**20:raise ValueError('Invalid stream metadata length')
    return json.loads(read_exact(stream,size))


def write_array(stream,array):
    value=np.ascontiguousarray(array)
    write_all(stream,value)


def read_array(stream,shape,dtype):
    dtype=np.dtype(dtype);size=int(np.prod(shape))*dtype.itemsize
    return np.frombuffer(read_exact(stream,size),dtype=dtype).reshape(shape)
