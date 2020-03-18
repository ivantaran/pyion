#!/usr/bin/python3

from rnxobs import *
from rnxhdr import RinexHeader
# from os import listdir
from os.path import isfile, join, dirname
import locale
import sys
import os

options = ['--combine PATH', '--decimate', '--split SECONDS', '--output PATH']


class RinexReader(object):

    def __init__(self):
        self._hdr = None  # TODO : RinexHeader
        self._obs = None  # TODO : RinexObservationsReader

    def open(self, file_name):
        line_index = 0
        data_lines = []

        with open(file_name, 'r') as f:
            self._hdr = RinexHeader()
            for line in f:
                line_index += 1
                if self._hdr.append(line) == RinexHeader.END_OF_HEADER:
                    break
            if not self._hdr.is_valid_for_obs():
                return
            for line in f:
                if not RinexHeader.is_comment(line):
                    data_lines.append(line)
                line_index += 1
            print('lines: header %d, data %d, total %d' % (len(self._hdr._lines), len(data_lines), line_index))

        if self._hdr.is_valid_for_obs():
            self._obs = RinexObservationsReader(self._hdr, data_lines)
        for name, obj in self._obs.objects().items():
            print('observations: %s %d' % (name, len(obj.observations())))

    def save_txt(self):
        if self._obs is not None:
            self._obs.save_txt()

    def save_rinex(self, path, makesubdirs=False):
        if self._obs is not None:
            self._obs.save_rinex(path, makesubdirs)

    def header(self):
        return self._hdr

    def observations(self):
        return self._obs


def combine(input_path: str, decimate: bool = False, output_path: str = '.', split: int = 0):
    rnx_list1 = []
    rnx_list2 = []
    locale.setlocale(locale.LC_ALL, '')

    if os.path.isdir(input_path):
        for root, subdirs, files in os.walk(input_path):
            for f in files:
                input_path = os.path.join(root, f)
                if os.path.isfile(input_path) and input_path.endswith('.rnx'):
                    rnx = RinexReader()
                    print(input_path)
                    rnx.open(input_path)
                    hdr = rnx.header()
                    if hdr is not None:
                        mn = hdr.marker_number()[1:2]
                        if mn == '0':
                            rnx_list1.append(rnx)
                        elif mn == '1':
                            rnx_list2.append(rnx)
                    print()
        print(len(rnx_list1), len(rnx_list2))
    else:
        print("[%s] is not directory" % input_path)

    for rnx_list in [rnx_list1, rnx_list2]:
        if len(rnx_list) <= 0:
            print('empty set')
            continue

        rnx = rnx_list[0]
        sys_obs_types = set()
        for r in rnx_list[1:]:
            for name, objout in r.observations().objects().items():
                if name in rnx.observations().objects().keys():
                    objin = rnx.observations().objects().get(name)  # TODO : ObservationObject
                    for time in objout.observations():
                        if time in objin.observations().keys():
                            objin.observations()[time].update(objout.observations()[time])
                        else:
                            objin.observations()[time] = objout.observations()[time]
                else:
                    rnx.observations().objects()[name] = objout
            for sys, types in r.header().sys_obs_types().items():
                s = set()
                s.update(types)
                s.update(rnx.header().sys_obs_types()[sys])
                rnx.header().sys_obs_types()[sys] = list(s)
            rnx.header().glonass_slot_frq().update(r.header().glonass_slot_frq())
            rnx.header().sys_phase_shift().update(r.header().sys_phase_shift())
            rnx.header().comment_lines().update(r.header().comment_lines())

        if decimate:
            rnx.observations().decimate()

        if split > 0:
            rnxsplit = rnx.observations().split(split)
            for r in rnxsplit:
                r.save_rinex(output_path, make_subdirs=True)
        else:
            rnx.save_rinex(output_path, make_subdirs=True)


def main(argv):
    locale.setlocale(locale.LC_ALL, '')
    print(argv)
    if len(argv) >= 3:
        if '--combine' in argv[1:-1]:
            index = argv.index('--combine')
            if len(argv) > index + 1:
                input_path = argv[index + 1]
                decimate = '--decimate' in argv[1:-1]
                if '--output' in argv[1:-1]:
                    index = argv.index('--output')
                    if len(argv) > index + 1:
                        output_path = argv[index + 1]
                    else:
                        output_path = '.'
                else:
                    output_path = '.'

                if '--split' in argv[1:-1]:
                    index = argv.index('--split')
                    if len(argv) > index + 1:
                        split = int(argv[index + 1])
                    else:
                        split = 900
                else:
                    split = 0

                combine(input_path, decimate, output_path, split)
            else:
                print('usage: python3 rnxreader.py %s [path]' % options)
    else:
        print('usage: python3 rnxreader.py %s [path]' % options)

    exit(0)
    # if len(argv) > 1 and isfile(argv[-1]):
    #     path = argv[-1]
    #     rnx = RinexReader()
    #     print(path)
    #     rnx.open(path)
    #     rnx.save_rinex(dirname(path))
    #     print()
    # else:
    #     print('usage: python3 rnxreader.py [filename]')
    # exit(0)

    # rnx = RinexReader()
    # for f in listdir('./data'):
    #     path = join('./data', f)
    #     if isfile(path):
    #         print(path)
    #         rnx.open(path)
    #         rnx.save_obs()
    #


if __name__ == '__main__':
    main(sys.argv)
