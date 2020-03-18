
from datetime import datetime, timedelta
from datetime import timezone
from os.path import join
from os.path import isdir
from os import makedirs
import math
import copy
from rnxhdr import RinexHeader


class ObservationObject(object):

    def __init__(self):
        self._observations = {}

    def put_observations(self, time, types, obs):  # TODO replace to update dict
        line = self._observations[time] if time in self._observations else {}
        for i, t in zip(range(len(obs)), types):
            line[t] = obs[i]
        self._observations[time] = line

    def observations(self):
        return self._observations

    def save(self, file_name):
        types = set('')

        for sample in self._observations.values():
            types.update(sample.keys())

        types = sorted(types)

        with open(file_name, 'w') as f:

            line = '\tTime'
            for t in types:
                line += '\t' + t
            f.write(line + '\n')

            for time, sample in sorted(self._observations.items()):
                line = '\t%f' % time
                for t in types:
                    line += '\t%f' % sample.get(t, 0.0)
                f.write(line + '\n')


class ObservationSample(object):

    def __init__(self, name, time, data):
        self._name = name
        self._time = time
        self._data = data

    def set_value(self, index, value):
        try:
            self._data[index] = value
        except LookupError:
            print('ObservationSample LookupError')

    def get_name(self):
        return self._name

    def get_time(self):
        return self._time

    def get_data(self):
        return self._data


class RinexObservationsReader(object):

    FLAG_OK = 0

    T31 = [
        -0.5,
        -0.4666666666666667,
        -0.43333333333333335,
        -0.4,
        -0.36666666666666664,
        -0.3333333333333333,
        -0.3,
        -0.26666666666666666,
        -0.23333333333333334,
        -0.2,
        -0.16666666666666666,
        -0.13333333333333333,
        -0.1,
        -0.06666666666666667,
        -0.03333333333333333,
        0.0,
        0.03333333333333333,
        0.06666666666666667,
        0.1,
        0.13333333333333333,
        0.16666666666666666,
        0.2,
        0.23333333333333334,
        0.26666666666666666,
        0.3,
        0.3333333333333333,
        0.36666666666666664,
        0.4,
        0.43333333333333335,
        0.4666666666666667,
        0.5
    ]

    C0 = [
        -0.04105571847507332,
        -0.02639296187683285,
        -0.012741429871574478,
        -1.0112245929821961E-4,
        0.011527960359995967,
        0.022145818586308028,
        0.03175245221963799,
        0.04034786125998585,
        0.047932045707351614,
        0.05450500556173527,
        0.06006674082313683,
        0.06461725149155628,
        0.06815653756699365,
        0.0706845990494489,
        0.07220143593892205,
        0.0727070482354131,
        0.07220143593892205,
        0.0706845990494489,
        0.06815653756699365,
        0.06461725149155628,
        0.06006674082313683,
        0.05450500556173527,
        0.047932045707351614,
        0.04034786125998585,
        0.03175245221963799,
        0.022145818586308028,
        0.011527960359995967,
        -1.0112245929821961E-4,
        -0.012741429871574478,
        -0.02639296187683285,
        -0.04105571847507332
    ]

    C1 = [
        - 0.18145161290322578,
        -0.1693548387096774,
        -0.157258064516129,
        -0.14516129032258063,
        -0.13306451612903222,
        -0.12096774193548387,
        -0.10887096774193548,
        -0.0967741935483871,
        -0.08467741935483873,
        -0.07258064516129034,
        -0.06048387096774195,
        -0.04838709677419356,
        -0.03629032258064518,
        -0.024193548387096784,
        -0.012096774193548399,
        -1.209647630676673E-17,
        0.012096774193548375,
        0.024193548387096763,
        0.03629032258064515,
        0.04838709677419353,
        0.06048387096774192,
        0.07258064516129031,
        0.0846774193548387,
        0.0967741935483871,
        0.10887096774193548,
        0.12096774193548387,
        0.13306451612903228,
        0.14516129032258068,
        0.15725806451612906,
        0.16935483870967746,
        0.18145161290322584
    ]

    C2 = [
        0.8247800586510264,
        0.6598240469208213,
        0.5062443118616645,
        0.36404085347355664,
        0.233213671756497,
        0.11376276671048635,
        0.005688138335524229,
        -0.09101021336838921,
        -0.17633228840125398,
        -0.25027808676307006,
        -0.31284760845383763,
        -0.3640408534735565,
        -0.4038578218222268,
        -0.4322985134998484,
        -0.44936292850642134,
        -0.4550510668419457,
        -0.44936292850642134,
        -0.4322985134998484,
        -0.4038578218222268,
        -0.3640408534735565,
        -0.31284760845383763,
        -0.25027808676307006,
        -0.17633228840125387,
        -0.0910102133683891,
        0.00568813833552434,
        0.11376276671048646,
        0.2332136717564971,
        0.36404085347355675,
        0.5062443118616646,
        0.6598240469208214,
        0.8247800586510265
    ]

    def __init__(self, header: RinexHeader, lines):
        self._header = header
        self._index = 0
        self._lines = lines
        self._objects = {}
        if self._header.rinex_version() in [2.00, 2.01, 2.10, 2.11]:
            self._parse()
        elif self._header.rinex_version() in [3.03]:
            self._parse3xx()
        else:
            print('invalid rinex version: %4.2f' % self._header.rinex_version())

    def objects(self):
        return self._objects

    def header(self):
        return self._header

    def _eod(self):
        return self._index >= len(self._lines)

    def _get_line(self):
        line = ''
        if not self._eod():
            line = self._lines[self._index]
            self._index += 1
        return line

    def _parse(self):
        t0 = 0.0
        self._index = 0
        while not self._eod():
            samples = []
            t0 = self._parse_header(samples, t0)
            if t0 > 0.0:
                self._parse_observations(samples)
                self._add_observations(samples)
            elif t0 < 0.0:
                break
        # self.decimate()

    def _parse3xx(self):
        t0 = 0.0
        self._index = 0
        while not self._eod():
            samples = []
            t0 = self._parse_header3xx(samples, t0)
            if t0 > 0.0:
                self._add_observations3xx(samples)
            elif t0 < 0.0:
                break
        # self.decimate()

    def _parse_header3xx(self, samples, t0):
        time = 0.0
        # samples.clear()
        del samples[:]  # TODO path for python 3.2
        line = self._get_line()

        if line[0] == '>':
            flag = self._get_flag3xx(line)
        else:
            flag = -1

        if flag == self.FLAG_OK:

            time = self._get_time3xx(line)
            if time - t0 > 0.0 and time > 0.0:  # TODO replace dt > 0.0 by Obs Interval
                count = self._get_object_count3xx(line)
                index_object = 0

                while index_object < count:
                    try:
                        line = self._get_line()
                        name = line[0:3]
                        types = self._header.sys_obs_types()[name[0]]
                        data = [0.0] * len(types)
                        samples.append(ObservationSample(name, time, data))

                        for i in range(len(data)):
                            value_position = 3 + i * 16
                            try:
                                value = float(line[value_position: value_position + 14])
                            except (LookupError, ValueError):
                                value = 0.0

                            value_position += 14
                            try:
                                lli = int(line[value_position])
                            except (LookupError, ValueError):
                                lli = 0

                            value_position += 1
                            try:
                                ps = int(line[value_position])
                            except (LookupError, ValueError):
                                ps = 0

                            # TODO lli and ps
                            data[i] = value

                        index_object += 1
                    except LookupError:
                        time = -1.0
                        print('bad name at data line {}'.format(self._index))
                        print(line)
            else:
                time = -1.0
                print('reverse time at data line {}'.format(self._index))
                print(line)

        else:
            # print('flag', flag)
            pass

        return time

    def _parse_header(self, samples, t0):
        time = 0.0
        # samples.clear()
        del samples[:]  # TODO path for python 3.2
        line = self._get_line()
        flag = self._get_flag(line)

        if flag == self.FLAG_OK:

            time = self._get_time(line)
            if time - t0 > 0.0:  # TODO replace 0.0 by Obs Interval
                count = self._get_object_count(line)
                index_object = 0

                while index_object < count:

                    for i in range(12):
                        if index_object < count:
                            object_name_position = 32 + i * 3
                            try:
                                name = line[object_name_position: object_name_position + 3]
                                data = [0.0] * len(self._header.types_of_observ())
                                samples.append(ObservationSample(name, time, data))
                                index_object += 1
                            except LookupError:
                                time = -1.0
                                print('bad name at data line {}'.format(self._index))
                                print(line)
                        else:
                            break
                    if index_object < count:
                        line = self._get_line()
            else:
                time = -1.0
                print('reverse time at data line {}'.format(self._index))
                print(line)

        else:
            # print('flag', flag)
            pass

        return time

    def _get_flag(self, line):
        flag = -1
        try:
            flag = int(line[28:29])
        except (LookupError, ValueError):
            print('bad flag at data line {}'.format(self._index))
            print(line)
        return flag

    def _get_flag3xx(self, line):
        flag = -1
        try:
            flag = int(line[31])
        except (LookupError, ValueError):
            print('bad flag at data line {}'.format(self._index))
            print(line)
        return flag

    def _get_time(self, line):
        result = 0.0

        if len(line) > 25:
            try:
                year = int(line[1:3]) + 2000  # TODO write base year from time of first obs
                month = int(line[4:6])
                day = int(line[7:9])
                hour = int(line[10:12])
                minute = int(line[13:15])
                fsecond = float(line[15:26])
                second = int(fsecond)
                fsecond -= second
                dt = datetime(year, month, day, hour, minute, second, int(fsecond * 1000000.0), tzinfo=timezone.utc)
                # result = dt.timestamp()  #  TODO leapSeconds and path for python 3.2
                result = (dt - datetime(1970, 1, 1, tzinfo=timezone.utc)) / timedelta(seconds=1)
            except ValueError:
                print('bad time at data line {}'.format(self._index))
                print(line)

        else:
            print('bad time at data line {}'.format(self._index))
            print(line)

        return result

    def _get_time3xx(self, line):
        result = 0.0

        if len(line) > 25:
            try:
                year = int(line[2:6])
                month = int(line[7:9])
                day = int(line[10:12])
                hour = int(line[13:15])
                minute = int(line[16:18])
                fsecond = float(line[18:29])
                second = int(fsecond)
                fsecond -= second
                dt = datetime(year, month, day, hour, minute, second, int(fsecond * 1000000.0), tzinfo=timezone.utc)
                # result = dt.timestamp()  #  TODO leapSeconds and path for python 3.2
                result = (dt - datetime(1970, 1, 1, tzinfo=timezone.utc)) / timedelta(seconds=1)
                # TODO Receiver clock offset Field insert here
            except ValueError:
                print('bad time at data line {}'.format(self._index))
                print(line)

        else:
            print('bad time at data line {}'.format(self._index))
            print(line)

        return result

    def _get_object_count(self, line):
        result = 0
        try:
            result = int(line[29:32])
        except (LookupError, ValueError):
            print('bad count at data line {}'.format(self._index))
            print(line)
        return result

    def _get_object_count3xx(self, line):
        result = 0
        try:
            result = int(line[32:35])
        except (LookupError, ValueError):
            print('bad count at data line {}'.format(self._index))
            print(line)
        return result

    def _parse_observations(self, samples):
        obserbations_line_count = (len(self._header.types_of_observ()) - 1) // 5 + 1
        value = 0.0
        lli = 0
        ps = 0

        for sample in samples:
            index = 0
            for j in range(obserbations_line_count):
                line = self._get_line()

                for i in range(5):
                    if not index < len(self._header.types_of_observ()):
                        continue

                    value_position = i * 16
                    try:
                        value = float(line[value_position : value_position + 14])
                    except (LookupError, ValueError):
                        value = 0.0

                    value_position += 14
                    try:
                        lli = int(line[value_position : value_position + 1])
                    except (LookupError, ValueError):
                        lli = 0

                    value_position += 1
                    try:
                        ps = int(line[value_position : value_position + 1])
                    except (LookupError, ValueError):
                        ps = 0

                    # TODO lli and ps
                    sample.set_value(index, value)
                    index = index + 1

    def _add_observations(self, samples):
        obj = None

        for sample in samples:
            name = sample.get_name()
            if name in self._objects:
                obj = self._objects[name]
            else:
                obj = ObservationObject()
                self._objects[name] = obj
            obj.put_observations(sample.get_time(), self._header.types_of_observ(), sample.get_data())

    def _add_observations3xx(self, samples):
        obj = None

        for sample in samples:
            name = sample.get_name()
            if name in self._objects:
                obj = self._objects[name]
            else:
                obj = ObservationObject()
                self._objects[name] = obj
            obj.put_observations(sample.get_time(), self._header.sys_obs_types()[name[0]], sample.get_data())

    def _get_obs_lines(self):
        if self._header.rinex_version() in [2.00, 2.01, 2.10, 2.11]:
            return self._get_obs_lines2xx()
        elif self._header.rinex_version() in [3.03]:
            return self._get_obs_lines3xx()
        else:
            return []

    def _get_obs_lines2xx(self):
        lines = []
        times = self._get_times()

        for time in times:
            names = []
            dt = datetime.fromtimestamp(time, tz=timezone.utc)
            line = ' %02d %2d %2d %2d %2d%11.7f  %1d' % (
                dt.year % 100, dt.month, dt.day, dt.hour, dt.minute, dt.second + dt.microsecond / 1000000.0, 0)

            for name, values in self._objects.items():
                if time in values.observations():
                    names.append(name)
            names = sorted(names)

            line += '%3d' % len(names)
            index_object = 0
            while index_object < len(names):
                for i in range(12):
                    if index_object < len(names):
                        line += names[index_object][0:3]
                        index_object += 1
                    else:
                        break
                lines.append(line[0:80].rstrip())
                line = 32 * ' '

            index_object = 0
            while index_object < len(names):
                index_observ = 0
                types = self._header.types_of_observ()
                while index_observ < len(types):
                    line = ''
                    for i in range(5):
                        if index_observ < len(types):
                            value = self._objects[names[index_object]].observations()[time][types[index_observ]]
                            if value == 0.0:
                                line += 16 * ' '
                            else:
                                line += '%14.3f ' % value
                                if types[index_observ][0] in 'L':
                                    t = types[index_observ]
                                    t = 'S' + t[1:]
                                    if t in types:
                                        value = self._objects[names[index_object]].observations()[time][t]
                                        value = min(max(int(value / 6), 0), 9)
                                        line += '%1d' % value
                                    else:
                                        line += ' '
                                else:
                                    line += ' '
                            index_observ += 1
                        else:
                            break
                    lines.append(line[0:80].rstrip())
                index_object += 1

        return lines

    def _get_obs_lines3xx(self):
        lines = []
        times = self._get_times()

        for time in times:
            names = []
            dt = datetime.fromtimestamp(time, tz=timezone.utc)
            line = '> %4d %2d %2d %2d %2d%11.7f  %1d' % (
                dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second + dt.microsecond / 1000000.0, 0)

            for name, values in self._objects.items():
                if time in values.observations():
                    names.append(name)
            names = sorted(names)

            line += '%3d' % len(names)
            lines.append(line.rstrip())

            for name in names:
                line = name[0:3]
                types = self._header.sys_obs_types()[name[0]]
                for type in types:
                    if type in self._objects[name].observations()[time].keys():
                        value = self._objects[name].observations()[time][type]
                    else:
                        value = 0.0
                    if value == 0.0:
                        line += 16 * ' '
                    else:
                        line += '%14.3f ' % value
                        if 'L' == type[0]:
                            t = type
                            t = 'S' + t[1:]
                            if t in types:
                                value = self._objects[name].observations()[time][t]
                                value = min(max(int(value / 6), 0), 9)
                                line += '%1d' % value
                            else:
                                line += ' '
                        else:
                            line += ' '
                lines.append(line.rstrip())

        return lines

    def save_txt(self):
        for k, obj in self._objects.items():
            obj.save('%s.txt' % k)

    def save_rinex(self, path, make_subdirs=False, header_comments=False):
        if len(self.objects()) > 0:
            if isdir(path):
                self._update_header()
                file_name, subdir = self._header.get_file_name_rapid_obs()
                if make_subdirs:
                    path = join(path, subdir)
                    makedirs(path, exist_ok=True)
                    path = join(path, file_name)
                else:
                    path = join(path, file_name)
                print('save rinex: ', path)
                with open(path, 'w') as f:
                    f.write('\n'.join(self._header.get_header_lines_obs(header_comments)))
                    f.write('\n')
                    f.write('\n'.join(self._get_obs_lines()))
                    f.write('\n')
            else:
                print('[%s] is not directory' % path)

    def decimate(self):
        self._medianfilter()
        # if self._header.interval() != 1.0:
        #     return
        # return
        #
        # for name, obj in self._objects.items():
        #     obs = {}
        #     for time in obj.observations():
        #         if int(time) % 30 == 0:
        #             sub = []
        #             for i in range(int(time) - len(self.C0) // 2, int(time) + len(self.C0) // 2 + 1):
        #                 if i in obj.observations():
        #                     sub.append(obj.observations()[i])
        #             if len(sub) == len(self.C0):
        #                 dec = {}
        #                 for k in sub[0]:
        #                     value = 0.0
        #                     for i in range(len(self.C0)):
        #                         if sub[i][k] != 0.0:
        #                             value += sub[i][k] * self.C0[i]
        #                         else:
        #                             value = 0.0
        #                             break
        #                     dec[k] = value
        #                     # dec[k] = sub[15][k] #TODO simple downsampling
        #                 obs[time] = dec
        #     obj.observations().clear()
        #     obj.observations().update(obs)
        #
        # self._header.set_interval(30.0)
        # print('observations decimated from interval 1.0 to 30.0 seconds')

    def _update_header(self):
        times = self._get_times()
        if len(times) > 0:
            self._header.set_time_of_first_obs(times[0])
        names = self._objects.keys()
        slots = {name: self.header().glonass_slot_frq()[name]
                 for name in names if name in self.header().glonass_slot_frq()}
        self.header().set_glonass_slot_frq(slots)
        self.header().sort_sys_obs_types()

    def _get_times(self):
        times = set()
        for obj in self._objects.values():
            times.update(obj.observations().keys())
        if len(times) <= 0:
            return []
        times = sorted(times)
        return times

    def split(self, period: int):

        times = self._get_times()
        if len(times) <= 0:
            return []

        rnxlist = []

        rnxobs = RinexObservationsReader(self.header(), [])
        period = rnxobs.header().set_period(period)
        objects = rnxobs.objects()
        index0 = times[0] // period
        index = index0

        for time in times:

            index = time // period

            if index != index0:
                if len(rnxobs.objects()) > 0:
                    rnxlist.append(rnxobs)
                rnxobs = RinexObservationsReader(copy.copy(self.header()), [])
                period = rnxobs.header().set_period(period)
                objects = rnxobs.objects()
                index0 = time // period

            for name, obj in self._objects.items():
                if time in obj.observations():
                    if name not in objects:
                        objects[name] = ObservationObject()
                    else:
                        pass
                    objects[name].observations()[time] = obj.observations()[time]

        if len(rnxobs.objects()) > 0:
            rnxlist.append(rnxobs)

        return rnxlist

    def _medianfilter(self):

        if self._header.interval() != 1.0:
            return

        for name, obj in self._objects.items():
            obs = {}
            for time in obj.observations():
                if int(time) % 30 == 0:
                    sub = []
                    for i in range(int(time) - len(self.C0) // 2, int(time) + len(self.C0) // 2 + 1):
                        if i in obj.observations():
                            sub.append(obj.observations()[i])
                    if len(sub) == len(self.C0):
                        dec = {}
                        for k in sub[0]:
                            column = []
                            value = 0.0
                            for i in range(len(self.C0)):
                                if k in sub[i] and sub[i][k] != 0.0:
                                    column.append(sub[i][k])
                                else:
                                    break
                            if len(column) == len(self.C0):
                                resids, value = self._quadresiduals31(column)
                                if k[0] != 'S':
                                    med = self._median(resids)
                                    mad = self._mad(med, resids) * 10.0
                                    for i in range(len(self.C0)):
                                        if abs(resids[i]) > mad:
                                            print('bad observation: ', i, resids[i], mad, k)
                                            value = 0.0
                                            break
                            dec[k] = value
                        obs[time] = dec
            obj.observations().clear()
            obj.observations().update(obs)

        self._header.set_interval(30.0)
        print('observations decimated from interval 1.0 to 30.0 seconds')

    def _quadresiduals31(self, din):

        k0 = 0.0
        k1 = 0.0
        k2 = 0.0
        r = []

        for i in range(len(self.C0)):
            k0 += din[i] * self.C0[i]
            k1 += din[i] * self.C1[i]
            k2 += din[i] * self.C2[i]

        for i in range(len(self.C0)):
            r.append(din[i] - k0 - k1 * self.T31[i] - k2 * self.T31[i] * self.T31[i])

        return r, k0

    def _mean(self, data):
        value = 0.0
        for d in data:
            value += d
        return value / len(data)

    def _stdev(self, mean, data):
        value = 0.0
        for d in data:
            d -= mean
            value += d * d
        return math.sqrt(value / (len(data) - 1.0))

    def _median(self, data):
        values = sorted(data)
        m = len(values)
        if m % 2 > 0:
            return values[m // 2]
        else:
            return (values[m // 2] + values[m // 2 + 1]) * 0.5

    def _mad(self, m, data):
        r = []
        for d in data:
            r.append(abs(d - m))
        return self._median(r)
