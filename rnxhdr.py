from datetime import datetime, timedelta
from datetime import timezone


class RinexHeader(object):

    END_OF_HEADER = True

    MARKER_LINE_INDEX = 60

    MARKER_RINEX_VERSION = \
        'RINEX VERSION / TYPE'

    MARKER_PGM_RUN_BY_DATE = \
        'PGM / RUN BY / DATE'

    MARKER_NAME = \
        'MARKER NAME'

    MARKER_NUMBER = \
        'MARKER NUMBER'

    MARKER_TYPE = \
        'MARKER TYPE'

    MARKER_OBSERVER_AGENCY = \
        'OBSERVER / AGENCY'

    MARKER_REC_NUM_TYPE_VERS = \
        'REC # / TYPE / VERS'

    MARKER_ANT_NUM_TYPE = \
        'ANT # / TYPE'

    MARKER_APPROX_POSITION_XYZ = \
        'APPROX POSITION XYZ'

    MARKER_ANTENNA_DELTA_HEN = \
        'ANTENNA: DELTA H/E/N'

    MARKER_TYPES_OF_OBSERV = \
        '# / TYPES OF OBSERV'

    MARKER_SYS_NUM_OBS_TYPES = \
        'SYS / # / OBS TYPES'

    MARKER_SIGNAL_STRENGTH_UNIT = \
        'SIGNAL STRENGTH UNIT'

    MARKER_INTERVAL = \
        'INTERVAL'

    MARKER_TIME_OF_FIRST_OBS = \
        'TIME OF FIRST OBS'

    MARKER_SYS_PHASE_SHIFT = \
        'SYS / PHASE SHIFT'

    MARKER_GLONASS_SLOT_FRQ = \
        'GLONASS SLOT / FRQ #'

    MARKER_GLONASS_COD_PHS_BIS = \
        'GLONASS COD/PHS/BIS'

    MARKER_COMMENT = \
        'COMMENT'

    MARKER_LEAP_SECONDS = \
        'LEAP SECONDS'

    MARKER_END_OF_HEADER = \
        'END OF HEADER'

    def __init__(self):
        self._rinex_version = 0.0
        self._type = ''
        self._system = ''
        self._pgm = ''
        self._run_by = ''
        self._date = ''
        self._marker_name = ''
        self._marker_number = ''
        self._marker_type = ''
        self._observer = ''
        self._agency = ''
        self._rec_num = ''
        self._rec_type = ''
        self._rec_vers = ''
        self._ant_num = ''
        self._ant_type = ''
        self._approx_position_xyz = [0.0, 0.0, 0.0]
        self._ant_hen = [0.0, 0.0, 0.0]
        self._types_of_observ = []
        self._sys_obs_types = {}
        self._signal_strength_unit = ''
        self._interval = 1.0
        self._period = '00U'
        self._time_of_first_obs = 0.0
        self._time_system = ''
        self._sys_phase_shift = set()
        self._glonass_slot_frq = {}
        self._glonass_cod_phs_bis = []
        self._leap_seconds = 0
        self._lines = []
        self._comment_lines = set()

    def is_valid_for_obs(self):
        result = True

        if self._rinex_version not in [2.00, 2.01, 2.10, 2.11, 3.03]:
            print('invalid rinex version: %4.2f' % self._rinex_version)
            result = False
        if len(self._type) < 1 or self._type[0] != 'O':
            print('rinex type is not O type:', self._type)
            result = False
        if len(self._marker_name) < 1:
            print('empty marker name')
            result = False
        if self._time_of_first_obs == 0.0:
            print('invalid time of first obs')
            result = False
        if (self._rinex_version in [2.00, 2.01, 2.10, 2.11] and not self._types_of_observ) \
                or (self._rinex_version in [3.03] and not self._sys_obs_types):  # TODO check dictionary items
            print('empty types of observations')
            result = False

        return result

    def append(self, line):
        self._lines.append(line)
        return self.END_OF_HEADER if self._eoh(line) else not self.END_OF_HEADER

    def _step_line(self, index):

        line = self._lines[index]

        if len(line) <= self.MARKER_LINE_INDEX:
            return 1
        elif self.MARKER_RINEX_VERSION in line[self.MARKER_LINE_INDEX:]:
            return self._parse_rinex_version(line)
        elif self.MARKER_PGM_RUN_BY_DATE in line[self.MARKER_LINE_INDEX:]:
            return self._parse_pgm_run_by_date(line)
        elif self.MARKER_NAME in line[self.MARKER_LINE_INDEX:]:
            return self._parse_marker_name(line)
        elif self.MARKER_NUMBER in line[self.MARKER_LINE_INDEX:]:
            return self._parse_marker_number(line)
        elif self.MARKER_TYPE in line[self.MARKER_LINE_INDEX:]:
            return self._parse_marker_type(line)
        elif self.MARKER_OBSERVER_AGENCY in line[self.MARKER_LINE_INDEX:]:
            return self._parse_observer_agency(line)
        elif self.MARKER_REC_NUM_TYPE_VERS in line[self.MARKER_LINE_INDEX:]:
            return self._parse_rec_num_type_vers(line)
        elif self.MARKER_ANT_NUM_TYPE in line[self.MARKER_LINE_INDEX:]:
            return self._parse_ant_num_type(line)
        elif self.MARKER_APPROX_POSITION_XYZ in line[self.MARKER_LINE_INDEX:]:
            return self._parse_approx_position_xyz(line)
        elif self.MARKER_ANTENNA_DELTA_HEN in line[self.MARKER_LINE_INDEX:]:
            return self._parse_antenna_delta_hen(line)
        elif self.MARKER_TYPES_OF_OBSERV in line[self.MARKER_LINE_INDEX:]:
            return self._parse_types_of_observ(index)
        elif self.MARKER_SYS_NUM_OBS_TYPES in line[self.MARKER_LINE_INDEX:]:
            return self._parse_sys_obs_types(index)
        elif self.MARKER_SIGNAL_STRENGTH_UNIT in line[self.MARKER_LINE_INDEX:]:
            return self._parse_signal_strength_unit(line)
        elif self.MARKER_INTERVAL in line[self.MARKER_LINE_INDEX:]:
            return self._parse_interval(line)
        elif self.MARKER_TIME_OF_FIRST_OBS in line[self.MARKER_LINE_INDEX:]:
            return self._parse_time_of_first_obs(line)
        elif self.MARKER_SYS_PHASE_SHIFT in line[self.MARKER_LINE_INDEX:]:
            return self._parse_sys_phase_shift(index)
        elif self.MARKER_GLONASS_SLOT_FRQ in line[self.MARKER_LINE_INDEX:]:
            return self._parse_glonass_slot_frq(index)
        elif self.MARKER_GLONASS_COD_PHS_BIS in line[self.MARKER_LINE_INDEX:]:
            return self._parse_glonass_cod_phs_bis(index)
        elif self.MARKER_LEAP_SECONDS in line[self.MARKER_LINE_INDEX:]:
            return self._parse_leap_seconds(line)
        elif self.MARKER_COMMENT in line[self.MARKER_LINE_INDEX:]:
            self._comment_lines.add(line[0:80].rstrip())

        return 1

    @staticmethod
    def is_comment(line):
        return RinexHeader.MARKER_COMMENT in line[RinexHeader.MARKER_LINE_INDEX:]

    def _eoh(self, line):
        result = False
        if self.MARKER_END_OF_HEADER in line[self.MARKER_LINE_INDEX:]:
            index = 0
            while index < len(self._lines):
                step = self._step_line(index)
                if step > 0:
                    index += step
                else:
                    break
            result = True
        return result

    def _parse_rinex_version(self, line):
        try:
            self._rinex_version = float(line[0:9])
            self._type = line[20:40]
            self._system = line[40:60]
        except (LookupError, ValueError):
            print('error in line\n%s' % line)
            return 0
        return 1

    def _parse_pgm_run_by_date(self, line):
        try:
            self._pgm = line[0:20]
            self._run_by = line[20:40]
            self._date = line[40:60]
        except LookupError:
            print('error in line\n%s' % line)
            return 0
        return 1

    def _parse_marker_name(self, line):
        try:
            self._marker_name = line[0:60].strip()
        except LookupError:
            print('error in line\n%s' % line)
            return 0
        return 1

    def _parse_marker_number(self, line):
        try:
            self._marker_number = line[0:20]
        except LookupError:
            print('error in line\n%s' % line)
            return 0
        return 1

    def _parse_marker_type(self, line):
        try:
            self._marker_type = line[0:20]
        except LookupError:
            print('error in line\n%s' % line)
            return 0
        return 1

    def _parse_observer_agency(self, line):
        try:
            self._observer = line[0:20]
            self._agency = line[20:40]
        except LookupError:
            print('error in line\n%s' % line)
            return 0
        return 1

    def _parse_rec_num_type_vers(self, line):
        try:
            self._rec_num = line[0:20]
            self._rec_type = line[20:40]
            self._rec_vers = line[40:60]
        except LookupError:
            print('error in line\n%s' % line)
            return 0
        return 1

    def _parse_ant_num_type(self, line):
        try:
            self._ant_num = line[0:20]
            self._ant_type = line[20:40]
        except LookupError:
            print('error in line\n%s' % line)
            return 0
        return 1

    def _parse_approx_position_xyz(self, line):
        try:
            self._approx_position_xyz[0] = float(line[0:14])
            self._approx_position_xyz[1] = float(line[14:28])
            self._approx_position_xyz[2] = float(line[28:42])
        except (LookupError, ValueError):
            print('error in line\n%s' % line)
            return 0
        return 1

    def _parse_antenna_delta_hen(self, line):
        try:
            self._ant_hen[0] = float(line[0:14])
            self._ant_hen[1] = float(line[14:28])
            self._ant_hen[2] = float(line[28:42])
        except (LookupError, ValueError):
            print('error in line\n%s' % line)
            return 0
        return 1

    def _parse_sys_obs_types(self, index):
        count_lines = 0
        types = []
        try:
            line = self._lines[index]
            sys = line[0].strip()
            count = int(line[3:6])
            count_lines = (count - 1) // 13 + 1

            for j in range(count_lines):
                line = self._lines[index + j]
                if self.MARKER_SYS_NUM_OBS_TYPES in line[self.MARKER_LINE_INDEX:]:
                    for i in range(13):
                        if len(types) < count:
                            pos = 7 + i * 4
                            t = line[pos: pos + 3].strip()
                            if len(t) > 0:
                                types.append(t)
                        else:
                            break
                else:
                    print('error in line\n%s' % line)
                    return 0
                if sys:
                    self._sys_obs_types[sys] = types
        except (LookupError, ValueError):
            print('error in line\n%s' % line)
            return 0
        return count_lines

    def _parse_signal_strength_unit(self, line):
        try:
            self._signal_strength_unit = line[0:20].strip()
        except LookupError:
            print('error in line\n%s' % line)
            return 0
        return 1

    def _parse_types_of_observ(self, index):
        count_lines = 0
        # self._types_of_observ.clear()
        del self._types_of_observ[:]  # TODO path for python 3.2
        try:
            line = self._lines[index]
            count = int(line[0:6])
            count_lines = (count - 1) // 9 + 1

            for j in range(count_lines):
                line = self._lines[index + j]
                if self.MARKER_TYPES_OF_OBSERV in line[self.MARKER_LINE_INDEX:]:
                    for i in range(9):
                        if len(self._types_of_observ) < count:
                            pos = 10 + i * 6
                            t = line[pos: pos + 2].strip()
                            if len(t) > 0:
                                self._types_of_observ.append(t)
                        else:
                            break
                else:
                    print('error in line\n%s' % line)
                    return 0
            if len(self._types_of_observ) != count:
                print('error: types of observ found %d of %d\n' % len(self._types_of_observ), count)
        except (LookupError, ValueError):
            print('error in line\n%s' % line)
            return 0
        return count_lines

    def _parse_interval(self, line):
        try:
            self._interval = float(line[0:10])
            pass
        except (LookupError, ValueError):
            print('error in line\n%s' % line)
            return 0
        return 1

    def _parse_time_of_first_obs(self, line):
        try:
            year = int(line[0:6])
            month = int(line[6:12])
            day = int(line[12:18])
            hour = int(line[18:24])
            minute = int(line[24:30])
            fsecond = float(line[30:43])
            second = int(fsecond)
            fsecond -= second
            self._time_system = line[48:51]
            dt = datetime(year, month, day, hour, minute, second, int(fsecond * 1000000.0), tzinfo=timezone.utc)
            # self._time_of_first_obs = dt.timestamp()  # TODO path for python 3.2
            self._time_of_first_obs = (dt - datetime(1970, 1, 1, tzinfo=timezone.utc)) / timedelta(seconds=1)
        except (LookupError, ValueError):
            print('error in line\n%s' % line)
            return 0
        return 1

    def _parse_sys_phase_shift(self, index):
        count_lines = 0
        try:
            while self.MARKER_SYS_PHASE_SHIFT in self._lines[index][self.MARKER_LINE_INDEX:]:
                self._sys_phase_shift.add(self._lines[index][0:80])
                index += 1
                count_lines += 1
        except LookupError:
            print('error in line\n%s' % self._lines[index])
            return 0
        return count_lines

    def _parse_glonass_slot_frq(self, index):
        count_lines = 0
        try:
            if self.MARKER_GLONASS_SLOT_FRQ in self._lines[index][self.MARKER_LINE_INDEX:]:
                count = int(self._lines[index][0:3])
                count_lines = count // 8 + 1
                for j in range(count_lines):
                    line = self._lines[index + j]
                    if self.MARKER_GLONASS_SLOT_FRQ in line[self.MARKER_LINE_INDEX:]:
                        for i in range(8):
                            if len(self._glonass_slot_frq) < count:
                                pos = 4 + i * 7
                                slot = line[pos:pos + 3].strip()
                                frq = int(line[pos + 4:pos + 6])
                                self._glonass_slot_frq[slot] = frq
                            else:
                                break
        except (LookupError, ValueError):
            print('error in line\n%s' % self._lines[index])
        return count_lines

    def _parse_glonass_cod_phs_bis(self, index):
        count_lines = 0
        try:
            while self.MARKER_GLONASS_COD_PHS_BIS in self._lines[index][self.MARKER_LINE_INDEX:]:
                self._glonass_cod_phs_bis.append(self._lines[index][0:80])
                index += 1
                count_lines += 1
        except LookupError:
            print('error in line\n%s' % self._lines[index])
            return 0
        return count_lines

    def _parse_leap_seconds(self, line):
        try:
            self._leap_seconds = int(line[0:6])
        except (LookupError, ValueError):
            print('error in line\n%s' % line)
            return 0
        return 1

    def set_interval(self, value):
        if 0.0 < value < 1000000.0:
            self._interval = value
        else:
            print('can\'t set invalid interval: %f' % value)

    def rinex_version(self):
        return self._rinex_version

    def types_of_observ(self):
        return self._types_of_observ

    def sys_obs_types(self):
        return self._sys_obs_types

    def interval(self):
        return self._interval

    def name(self):
        return self._marker_name

    def marker_number(self):
        return self._marker_number

    def time_of_first_obs(self):
        return self._time_of_first_obs

    def glonass_slot_frq(self):
        return self._glonass_slot_frq

    def sys_phase_shift(self):
        return self._sys_phase_shift

    def comment_lines(self):
        return self._comment_lines

    def get_file_name_rapid_obs(self):
        if self._rinex_version in [2.00, 2.01, 2.10, 2.11]:
            return self.get_file_name_rapid_obs2xx()
        elif self._rinex_version in [3.03]:
            return self.get_file_name_rapid_obs3xx()
        else:
            return '', ''

    def get_file_name_rapid_obs3xx(self):
        file_name = self._marker_name.upper()

        try:
            mr = int(self._marker_number[0:20]) % 100
        except (LookupError, ValueError):
            mr = 0

        file_name += '%02dRUS_R_' % mr
        dt = datetime.fromtimestamp(self._time_of_first_obs, tz=timezone.utc)
        file_name += '%04d%03d%02d%02d' % \
                     (dt.year % 10000, dt.timetuple().tm_yday % 1000, dt.hour % 100, dt.minute % 100)
        file_name += '_%3s_%02dS_' % (self._period, self._interval)
        if len(self._sys_obs_types) == 1:
            file_name += '%cO.rnx' % list(self._sys_obs_types.keys())[0]
        else:
            file_name += 'MO.rnx'

        subdir = '%02d%03d/%02do' % (dt.year % 100, dt.timetuple().tm_yday % 1000, dt.year % 100)

        return file_name, subdir

    def get_file_name_rapid_obs2xx(self):
        file_name = self._marker_name.lower()
        dt = datetime.fromtimestamp(self._time_of_first_obs, tz=timezone.utc)
        file_name += '%03d' % (dt.timetuple().tm_yday % 1000)
        file_name += chr(dt.hour + ord('a'))
        file_name += '%02d' % (dt.minute % 100)
        file_name += '.%02do' % (dt.year % 100)
        subdir = '%02d%03d/%02do' % (dt.year % 100, dt.timetuple().tm_yday % 1000, dt.year % 100)
        return file_name, subdir

    def _get_lines_glonass_slot_frq3xx(self):
        lines = []
        if len(self._glonass_slot_frq) > 0:
            index = 0
            line = '%3d ' % (len(self._glonass_slot_frq) % 1000)
            for slot, frq in self._glonass_slot_frq.items():
                line += '%3s %2d ' % (slot, frq)
                if index == 7:
                    line = '%-60s%-20s' % (line[0:60], RinexHeader.MARKER_GLONASS_SLOT_FRQ)
                    lines.append(line[0:80].rstrip())
                    line = ' ' * 4
                    index = 0
                else:
                    index += 1
            if index > 0:
                line = '%-60s%-20s' % (line[0:60], RinexHeader.MARKER_GLONASS_SLOT_FRQ)
                lines.append(line[0:80].rstrip())

        return lines

    def sort_sys_obs_types(self):
        for sys, types in self._sys_obs_types.items():
            types = sorted(map(lambda t: t[1:] + t[0], types))
            types = list(map(lambda t: t[-1] + t[0:-1], types))
            self._sys_obs_types[sys] = types

    def _get_lines_types_of_observ3xx(self):
        lines = []
        for sys, types in self._sys_obs_types.items():
            index = 0
            line = '%c  %3d' % (sys[0], len(types) % 1000)
            for type in types:
                line += ' %3s' % type[0:3]
                if index == 12:
                    line = '%-60s%-20s' % (line[0:60], RinexHeader.MARKER_SYS_NUM_OBS_TYPES)
                    lines.append(line[0:80].rstrip())
                    line = ' ' * 6
                    index = 0
                else:
                    index += 1
            if index > 0:
                line = '%-60s%-20s' % (line[0:60], RinexHeader.MARKER_SYS_NUM_OBS_TYPES)
                lines.append(line[0:80].rstrip())

        return lines

    def _get_lines_types_of_observ2xx(self):
        lines = []
        line = '%6d' % len(self._types_of_observ)
        index = 0
        while index < len(self._types_of_observ):
            for i in range(9):
                if index < len(self._types_of_observ):
                    line += '    %2s' % self._types_of_observ[index][0:2]
                    index += 1
                else:
                    break
            line = '%-60s%-20s' % (line[0:60], RinexHeader.MARKER_TYPES_OF_OBSERV)
            lines.append(line[0:80].rstrip())
            line = '      '

        return lines

    def get_header_lines_obs(self, header_comments=False):
        if self._rinex_version in [2.00, 2.01, 2.10, 2.11]:
            return self._get_header_lines_obs2xx(header_comments)
        elif self._rinex_version in [3.03]:
            return self._get_header_lines_obs3xx(header_comments)
        else:
            return []

    def _get_header_lines_obs3xx(self, header_comments=False):
        lines = []

        if self.is_valid_for_obs():

            line = '%9.2f%-11s%-20s%-20s%-20s' % (
                self._rinex_version, '', self._type[0:20],
                self._system[0:20], RinexHeader.MARKER_RINEX_VERSION)
            lines.append(line[0:80].rstrip())

            line = '%-20s%-20s%-20s%-20s' % (
                self._pgm[0:20], self._run_by[0:20], self._date[0:20], RinexHeader.MARKER_PGM_RUN_BY_DATE)
            lines.append(line[0:80].rstrip())

            line = '%-60s%-20s' % (self._marker_name[0:60], RinexHeader.MARKER_NAME)
            lines.append(line[0:80].rstrip())

            if not self._marker_number.isspace():
                line = '%-20s%-40s%-20s' % (self._marker_number[0:20], '', RinexHeader.MARKER_NUMBER)
                lines.append(line[0:80].rstrip())

            line = '%-20s%-40s%-20s' % (self._marker_type[0:20], '', RinexHeader.MARKER_TYPE)
            lines.append(line[0:80].rstrip())

            line = '%-20s%-40s%-20s' % (self._observer[0:20], self._agency[0:40], RinexHeader.MARKER_OBSERVER_AGENCY)
            lines.append(line[0:80].rstrip())

            line = '%-20s%-20s%-20s%-20s' % (
                self._rec_num[0:20], self._rec_type[0:20], self._rec_vers[0:20], RinexHeader.MARKER_REC_NUM_TYPE_VERS)
            lines.append(line[0:80].rstrip())

            line = '%-20s%-20s%-20s%-20s' % (
                self._ant_num[0:20], self._ant_type[0:20], '', RinexHeader.MARKER_ANT_NUM_TYPE)
            lines.append(line[0:80].rstrip())

            line = '%14.4f%14.4f%14.4f%-18s%-20s' % (
                self._approx_position_xyz[0], self._approx_position_xyz[1], self._approx_position_xyz[2], '',
                RinexHeader.MARKER_APPROX_POSITION_XYZ)
            lines.append(line[0:80].rstrip())

            line = '%14.4f%14.4f%14.4f%-18s%-20s' % (
                self._ant_hen[0], self._ant_hen[1], self._ant_hen[2], '', RinexHeader.MARKER_ANTENNA_DELTA_HEN)
            lines.append(line[0:80].rstrip())

            lines.extend(self._get_lines_types_of_observ3xx())

            if not self._signal_strength_unit.isspace():
                line = '%-20s%-40s%-20s' % (self._signal_strength_unit[0:20], '',
                                            RinexHeader.MARKER_SIGNAL_STRENGTH_UNIT)
                lines.append(line[0:80].rstrip())

            if self._interval > 0.0:
                line = '%10.3f%-50s%-20s' % (self._interval, '', RinexHeader.MARKER_INTERVAL)
                lines.append(line[0:80].rstrip())

            dt = datetime.fromtimestamp(self._time_of_first_obs, tz=timezone.utc)
            line = '%6d%6d%6d%6d%6d%13.7f%-5s%-3s%-9s%-20s' % (
                dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second + dt.microsecond / 1000000.0,
                '', self._time_system[0:3], '', RinexHeader.MARKER_TIME_OF_FIRST_OBS)
            lines.append(line[0:80].rstrip())

            lines.extend(sorted(self._sys_phase_shift))
            lines.extend(self._get_lines_glonass_slot_frq3xx())

            for line in self._glonass_cod_phs_bis:
                lines.append(line[0:80].rstrip())

            if self._leap_seconds > 0:
                line = '%6d%-54s%-20s' % (self._leap_seconds, '', RinexHeader.MARKER_LEAP_SECONDS)
                lines.append(line[0:80].rstrip())

            if header_comments:
                lines.extend(sorted(self._comment_lines))

            line = '%-60s%-20s' % ('', RinexHeader.MARKER_END_OF_HEADER)
            lines.append(line[0:80].rstrip())

        return lines

    def _get_header_lines_obs2xx(self, header_comments=False):
        lines = []

        if self.is_valid_for_obs():

            line = '%9.2f%-11s%-20s%-20s%-20s' % (
                self._rinex_version, '', self._type[0:20],
                self._system[0:20], RinexHeader.MARKER_RINEX_VERSION)
            lines.append(line[0:80].rstrip())

            line = '%-20s%-20s%-20s%-20s' % (
                self._pgm[0:20], self._run_by[0:20], self._date[0:20], RinexHeader.MARKER_PGM_RUN_BY_DATE)
            lines.append(line[0:80].rstrip())

            line = '%-60s%-20s' % (self._marker_name[0:60], RinexHeader.MARKER_NAME)
            lines.append(line[0:80].rstrip())

            if not self._marker_number.isspace():
                line = '%-20s%-40s%-20s' % (self._marker_number[0:20], '', RinexHeader.MARKER_NUMBER)
                lines.append(line[0:80].rstrip())

            line = '%-20s%-40s%-20s' % (self._observer[0:20], self._agency[0:40], RinexHeader.MARKER_OBSERVER_AGENCY)
            lines.append(line[0:80].rstrip())

            line = '%-20s%-20s%-20s%-20s' % (
                self._rec_num[0:20], self._rec_type[0:20], self._rec_vers[0:20], RinexHeader.MARKER_REC_NUM_TYPE_VERS)
            lines.append(line[0:80].rstrip())

            line = '%-20s%-20s%-20s%-20s' % (
                self._ant_num[0:20], self._ant_type[0:20], '', RinexHeader.MARKER_ANT_NUM_TYPE)
            lines.append(line[0:80].rstrip())

            line = '%14.4f%14.4f%14.4f%-18s%-20s' % (
                self._approx_position_xyz[0], self._approx_position_xyz[1], self._approx_position_xyz[2], '',
                RinexHeader.MARKER_APPROX_POSITION_XYZ)
            lines.append(line[0:80].rstrip())

            line = '%14.4f%14.4f%14.4f%-18s%-20s' % (
                self._ant_hen[0], self._ant_hen[1], self._ant_hen[2], '', RinexHeader.MARKER_ANTENNA_DELTA_HEN)
            lines.append(line[0:80].rstrip())

            lines.extend(self._get_lines_types_of_observ2xx())

            if self._interval > 0.0:
                line = '%10.3f%-50s%-20s' % (self._interval, '', RinexHeader.MARKER_INTERVAL)
                lines.append(line[0:80].rstrip())

            dt = datetime.fromtimestamp(self._time_of_first_obs, tz=timezone.utc)
            line = '%6d%6d%6d%6d%6d%13.7f%-5s%-3s%-9s%-20s' % (
                dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second + dt.microsecond / 1000000.0,
                '', self._time_system[0:3], '', RinexHeader.MARKER_TIME_OF_FIRST_OBS)
            lines.append(line[0:80].rstrip())

            if self._leap_seconds > 0:
                line = '%6d%-54s%-20s' % (self._leap_seconds, '', RinexHeader.MARKER_LEAP_SECONDS)
                lines.append(line[0:80].rstrip())

            if header_comments:
                lines.extend(sorted(self._comment_lines))

            line = '%-60s%-20s' % ('', RinexHeader.MARKER_END_OF_HEADER)
            lines.append(line[0:80].rstrip())

        return lines

    def set_time_of_first_obs(self, value: float):
        self._time_of_first_obs = value

    def set_glonass_slot_frq(self, values: dict):
        self._glonass_slot_frq = values

    def set_period(self, seconds: int):
        m = 'U'  # TODO : chr
        t = 0  # TODO : int
        if seconds <= 900:
            m = 'M'
            t = 15
            seconds = t * 60
        elif seconds < 3600:
            m = 'M'
            t = seconds // 60
            seconds = t * 60
        elif seconds < 86400:
            m = 'H'
            t = seconds // 3600
            seconds = t * 3600
        elif seconds < 8640000:
            m = 'D'
            t = seconds // 86400
            seconds = t * 86400
        else:
            m = 'U'
            t = 0
        self._period = '%02d%c' % (t, m)
        return seconds
