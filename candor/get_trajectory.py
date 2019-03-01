#!/usr/bin/env python
import sys
import bz2
import json

import h5py

from .dump_stream import pretty

def get_traj_from_nexus(filename):
    file = h5py.File(filename)
    for name, value in file.items():
        traj_str = value['DAS_logs/trajectory/config'][()][0]
        break
    traj = json.loads(traj_str)
    return traj

def get_traj_from_stream(filename):
    with bz2.BZ2File(filename) as file_handle:
        for line in file_handle:
            line = line.decode('utf-8')
            break
    record = json.loads(line)
    traj_str = record['data']['trajectory.config']
    traj = json.loads(record['data']['trajectory.config'])
    return traj

def main():
    filename = sys.argv[1]
    if '.nxs.' in filename:
        traj = get_traj_from_nexus(filename)
    elif filename.endswith('.bz2'):
        traj = get_traj_from_stream(filename)
    else:
        raise ValueError("need stream or nexus file")
    pretty(traj)

if __name__ == "__main__":
    main()
