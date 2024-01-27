import os
import re
import time
import warnings
from typing import (
    TYPE_CHECKING,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
    cast,
)
from jinja2 import Environment, FileSystemLoader, PackageLoader, select_autoescape

from cantools import __version__

if TYPE_CHECKING:
    from cantools.database.can import Database, Message, Signal


_T1 = TypeVar("_T1")
_T2 = TypeVar("_T2")
THelperKind = Tuple[str, int]


MESSAGE_DEFINITION_INIT_FMT = '''\
int {database_name}_{message_name}_init(struct {database_name}_{message_name}_t *msg_p)
{{
    if (msg_p == NULL) return -1;

    memset(msg_p, 0, sizeof(struct {database_name}_{message_name}_t));
{init_body}
    return 0;
}}
'''

DEFINITION_PACK_FMT = '''\
int {database_name}_{message_name}_pack(
    uint8_t *dst_p,
    const struct {database_name}_{message_name}_t *src_p,
    size_t size)
{{
{pack_unused}\
{pack_variables}\
    if (size < {message_length}u) {{
        return (-EINVAL);
    }}

    memset(&dst_p[0], 0, {message_length});
{pack_body}
    return ({message_length});
}}

'''

DEFINITION_UNPACK_FMT = '''\
int {database_name}_{message_name}_unpack(
    struct {database_name}_{message_name}_t *dst_p,
    const uint8_t *src_p,
    size_t size)
{{
{unpack_unused}\
{unpack_variables}\
    if (size < {message_length}u) {{
        return (-EINVAL);
    }}
{unpack_body}
    return (0);
}}

'''

SIGNAL_DEFINITION_ENCODE_FMT = '''\
{type_name} {database_name}_{message_name}_{signal_name}_encode({floating_point_type} value)
{{
    return ({type_name})({encode});
}}

'''

SIGNAL_DEFINITION_DECODE_FMT = '''\
{floating_point_type} {database_name}_{message_name}_{signal_name}_decode({type_name} value)
{{
    return ({decode});
}}

'''

SIGNAL_DEFINITION_IS_IN_RANGE_FMT = '''\
bool {database_name}_{message_name}_{signal_name}_is_in_range({type_name} value)
{{
{unused}\
    return ({check});
}}
'''

EMPTY_DEFINITION_FMT = '''\
int {database_name}_{message_name}_pack(
    uint8_t *dst_p,
    const struct {database_name}_{message_name}_t *src_p,
    size_t size)
{{
    (void)dst_p;
    (void)src_p;
    (void)size;

    return (0);
}}

int {database_name}_{message_name}_unpack(
    struct {database_name}_{message_name}_t *dst_p,
    const uint8_t *src_p,
    size_t size)
{{
    (void)dst_p;
    (void)src_p;
    (void)size;

    return (0);
}}
'''

SIGN_EXTENSION_FMT = '''
    if (({name} & (1{suffix} << {shift})) != 0{suffix}) {{
        {name} |= 0x{mask:x}{suffix};
    }}

'''

INIT_SIGNAL_BODY_TEMPLATE_FMT = '''\
    msg_p->{signal_name} = {signal_initial};
'''


class CodeGenSignal:

    def __init__(self, signal: "Signal") -> None:
        self.signal: "Signal" = signal
        self.snake_name = camel_to_snake_case(signal.name)

    @property
    def unit(self) -> str:
        return _get(self.signal.unit, '-')

    @property
    def type_length(self) -> int:
        if self.signal.length <= 8:
            return 8
        elif self.signal.length <= 16:
            return 16
        elif self.signal.length <= 32:
            return 32
        else:
            return 64

    @property
    def type_name(self) -> str:
        if self.signal.conversion.is_float:
            if self.signal.length == 32:
                type_name = 'float'
            else:
                type_name = 'double'
        else:
            type_name = f'int{self.type_length}_t'

            if not self.signal.is_signed:
                type_name = 'u' + type_name

        return type_name

    @property
    def type_suffix(self) -> str:
        try:
            return {
                'uint8_t': 'u',
                'uint16_t': 'u',
                'uint32_t': 'u',
                'int64_t': 'll',
                'uint64_t': 'ull',
                'float': 'f'
            }[self.type_name]
        except KeyError:
            return ''

    @property
    def conversion_type_suffix(self) -> str:
        try:
            return {
                8: 'u',
                16: 'u',
                32: 'u',
                64: 'ull'
            }[self.type_length]
        except KeyError:
            return ''

    @property
    def unique_choices(self) -> Dict[int, str]:
        """Make duplicated choice names unique by first appending its value
        and then underscores until unique.

        """
        if self.signal.choices is None:
            return {}

        items = {
            value: camel_to_snake_case(str(name)).upper()
            for value, name in self.signal.choices.items()
        }
        names = list(items.values())
        duplicated_names = [
            name
            for name in set(names)
            if names.count(name) > 1
        ]
        unique_choices = {
            value: name
            for value, name in items.items()
            if names.count(name) == 1
        }

        for value, name in items.items():
            if name in duplicated_names:
                name += _canonical(f'_{value}')

                while name in unique_choices.values():
                    name += '_'

                unique_choices[value] = name

        return unique_choices

    @property
    def minimum_ctype_value(self) -> Optional[int]:
        if self.type_name == 'int8_t':
            return -2**7
        elif self.type_name == 'int16_t':
            return -2**15
        elif self.type_name == 'int32_t':
            return -2**31
        elif self.type_name == 'int64_t':
            return -2**63
        elif self.type_name.startswith('u'):
            return 0
        else:
            return None

    @property
    def maximum_ctype_value(self) -> Optional[int]:
        if self.type_name == 'int8_t':
            return 2**7 - 1
        elif self.type_name == 'int16_t':
            return 2**15 - 1
        elif self.type_name == 'int32_t':
            return 2**31 - 1
        elif self.type_name == 'int64_t':
            return 2**63 - 1
        elif self.type_name == 'uint8_t':
            return 2**8 - 1
        elif self.type_name == 'uint16_t':
            return 2**16 - 1
        elif self.type_name == 'uint32_t':
            return 2**32 - 1
        elif self.type_name == 'uint64_t':
            return 2**64 - 1
        else:
            return None

    @property
    def minimum_can_raw_value(self) -> Optional[int]:
        if self.signal.conversion.is_float:
            return None
        elif self.signal.is_signed:
            return cast(int, -(2 ** (self.signal.length - 1)))
        else:
            return 0

    @property
    def maximum_can_raw_value(self) -> Optional[int]:
        if self.signal.conversion.is_float:
            return None
        elif self.signal.is_signed:
            return cast(int, (2 ** (self.signal.length - 1)) - 1)
        else:
            return cast(int, (2 ** self.signal.length) - 1)

    def segments(self, invert_shift: bool) -> Iterator[Tuple[int, int, str, int]]:
        index, pos = divmod(self.signal.start, 8)
        left = self.signal.length

        while left > 0:
            if self.signal.byte_order == 'big_endian':
                if left >= (pos + 1):
                    length = (pos + 1)
                    pos = 7
                    shift = -(left - length)
                    mask = ((1 << length) - 1)
                else:
                    length = left
                    shift = (pos - length + 1)
                    mask = ((1 << length) - 1)
                    mask <<= (pos - length + 1)
            else:
                shift = (left - self.signal.length) + pos

                if left >= (8 - pos):
                    length = (8 - pos)
                    mask = ((1 << length) - 1)
                    mask <<= pos
                    pos = 0
                else:
                    length = left
                    mask = ((1 << length) - 1)
                    mask <<= pos

            if invert_shift:
                if shift < 0:
                    shift = -shift
                    shift_direction = 'left'
                else:
                    shift_direction = 'right'
            else:
                if shift < 0:
                    shift = -shift
                    shift_direction = 'right'
                else:
                    shift_direction = 'left'

            yield index, shift, shift_direction, mask

            left -= length
            index += 1


class CodeGenMessage:

    def __init__(self, message: "Message") -> None:
        self.message = message
        self.snake_name = camel_to_snake_case(message.name)
        self.cg_signals = [CodeGenSignal(signal) for signal in message.signals]

    def get_signal_by_name(self, name: str) -> "CodeGenSignal":
        for cg_signal in self.cg_signals:
            if cg_signal.signal.name == name:
                return cg_signal
        raise KeyError(f"Signal {name} not found.")


def _canonical(value: str) -> str:
    """Replace anything but 'a-z', 'A-Z' and '0-9' with '_'.

    """

    return re.sub(r'[^a-zA-Z0-9]', '_', value)


def camel_to_snake_case(value: str) -> str:
    value = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', value)
    value = re.sub(r'(_+)', '_', value)
    value = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', value).lower()
    value = _canonical(value)

    return value


def _strip_blank_lines(lines: List[str]) -> List[str]:
    try:
        while lines[0] == '':
            lines = lines[1:]

        while lines[-1] == '':
            lines = lines[:-1]
    except IndexError:
        pass

    return lines


def _get(value: Optional[_T1], default: _T2) -> Union[_T1, _T2]:
    if value is None:
        return default
    return value


def _format_comment(comment: Optional[str]) -> str:
    return f'{comment}'


def _format_range(cg_signal: "CodeGenSignal") -> str:
    minimum = cg_signal.signal.minimum
    maximum = cg_signal.signal.maximum

    def phys_to_raw(x: Union[int, float]) -> Union[int, float]:
        raw_val = cg_signal.signal.scaled_to_raw(x)
        if cg_signal.signal.is_float:
            return float(raw_val)
        return round(raw_val)

    if minimum is not None and maximum is not None:
        return \
            f'{phys_to_raw(minimum)}..' \
            f'{phys_to_raw(maximum)} ' \
            f'({round(minimum, 5)}..{round(maximum, 5)} {cg_signal.unit})'
    elif minimum is not None:
        return f'{phys_to_raw(minimum)}.. ({round(minimum, 5)}.. {cg_signal.unit})'
    elif maximum is not None:
        return f'..{phys_to_raw(maximum)} (..{round(maximum, 5)} {cg_signal.unit})'
    else:
        return '-'


def _generate_signal(cg_signal: "CodeGenSignal", bit_fields: bool) -> str:
    comment = _format_comment(cg_signal.signal.comment)
    range_ = _format_range(cg_signal)
    scale = _get(cg_signal.signal.conversion.scale, '-')
    offset = _get(cg_signal.signal.conversion.offset, '-')

    if cg_signal.signal.conversion.is_float or not bit_fields:
        length = ''
    else:
        length = f' : {cg_signal.signal.length}'

    return {'comment': comment,
            'range': range_,
            'scale': scale,
            'offset': offset,
            'type_name': cg_signal.type_name,
            'name': cg_signal.snake_name,
            'length': length}


def _format_pack_code_mux(cg_message: "CodeGenMessage",
                          mux: Dict[str, Dict[int, List[str]]],
                          body_lines_per_index: List[str],
                          variable_lines: List[str],
                          helper_kinds: Set[THelperKind]) -> List[str]:
    signal_name, multiplexed_signals = list(mux.items())[0]
    _format_pack_code_signal(cg_message,
                             signal_name,
                             body_lines_per_index,
                             variable_lines,
                             helper_kinds)
    multiplexed_signals_per_id = sorted(multiplexed_signals.items())
    signal_name = camel_to_snake_case(signal_name)

    lines = [
        '',
        f'switch (src_p->{signal_name}) {{'
    ]

    for multiplexer_id, signals_of_multiplexer_id in multiplexed_signals_per_id:
        body_lines = _format_pack_code_level(cg_message,
                                             signals_of_multiplexer_id,
                                             variable_lines,
                                             helper_kinds)
        lines.append('')
        lines.append(f'case {multiplexer_id}:')

        if body_lines:
            lines.extend(body_lines[1:-1])

        lines.append('    break;')

    lines.extend([
        '',
        'default:',
        '    break;',
        '}'])

    return [('    ' + line).rstrip() for line in lines]


def _format_pack_code_signal(cg_message: "CodeGenMessage",
                             signal_name: str,
                             body_lines: List[str],
                             variable_lines: List[str],
                             helper_kinds: Set[THelperKind]) -> None:
    cg_signal = cg_message.get_signal_by_name(signal_name)

    if cg_signal.signal.conversion.is_float or cg_signal.signal.is_signed:
        variable = f'    uint{cg_signal.type_length}_t {cg_signal.snake_name};'

        if cg_signal.signal.conversion.is_float:
            conversion = '    memcpy(&{0}, &src_p->{0}, sizeof({0}));'.format(
                cg_signal.snake_name)
        else:
            conversion = '    {0} = (uint{1}_t)src_p->{0};'.format(
                cg_signal.snake_name,
                cg_signal.type_length)

        variable_lines.append(variable)
        body_lines.append(conversion)

    for index, shift, shift_direction, mask in cg_signal.segments(invert_shift=False):
        if cg_signal.signal.conversion.is_float or cg_signal.signal.is_signed:
            fmt = '    dst_p[{}] |= pack_{}_shift_u{}({}, {}u, 0x{:02x}u);'
        else:
            fmt = '    dst_p[{}] |= pack_{}_shift_u{}(src_p->{}, {}u, 0x{:02x}u);'

        line = fmt.format(index,
                          shift_direction,
                          cg_signal.type_length,
                          cg_signal.snake_name,
                          shift,
                          mask)
        body_lines.append(line)
        helper_kinds.add((shift_direction, cg_signal.type_length))


def _format_pack_code_level(cg_message: "CodeGenMessage",
                            signal_names: Union[List[str], List[Dict[str, Dict[int, List[str]]]]],
                            variable_lines: List[str],
                            helper_kinds: Set[THelperKind]) -> List[str]:
    """Format one pack level in a signal tree.

    """

    body_lines: List[str] = []
    muxes_lines: List[str] = []

    for signal_name in signal_names:
        if isinstance(signal_name, dict):
            mux_lines = _format_pack_code_mux(cg_message,
                                              signal_name,
                                              body_lines,
                                              variable_lines,
                                              helper_kinds)
            muxes_lines += mux_lines
        else:
            _format_pack_code_signal(cg_message,
                                     signal_name,
                                     body_lines,
                                     variable_lines,
                                     helper_kinds)

    body_lines = body_lines + muxes_lines

    if body_lines:
        body_lines = [''] + body_lines + ['']

    return body_lines


def _format_pack_code(cg_message: "CodeGenMessage",
                      helper_kinds: Set[THelperKind]
                      ) -> Tuple[str, str]:
    variable_lines: List[str] = []
    body_lines = _format_pack_code_level(cg_message,
                                         cg_message.message.signal_tree,
                                         variable_lines,
                                         helper_kinds)

    if variable_lines:
        variable_lines = sorted(set(variable_lines)) + ['', '']

    return '\n'.join(variable_lines), '\n'.join(body_lines)


def _format_unpack_code_mux(cg_message: "CodeGenMessage",
                            mux: Dict[str, Dict[int, List[str]]],
                            body_lines_per_index: List[str],
                            variable_lines: List[str],
                            helper_kinds: Set[THelperKind],
                            node_name: Optional[str]) -> List[str]:
    signal_name, multiplexed_signals = list(mux.items())[0]
    _format_unpack_code_signal(cg_message,
                               signal_name,
                               body_lines_per_index,
                               variable_lines,
                               helper_kinds)
    multiplexed_signals_per_id = sorted(multiplexed_signals.items())
    signal_name = camel_to_snake_case(signal_name)

    lines = [
        f'switch (dst_p->{signal_name}) {{'
    ]

    for multiplexer_id, signals_of_multiplexer_id in multiplexed_signals_per_id:
        body_lines = _format_unpack_code_level(cg_message,
                                               signals_of_multiplexer_id,
                                               variable_lines,
                                               helper_kinds,
                                               node_name)
        lines.append('')
        lines.append(f'case {multiplexer_id}:')
        lines.extend(_strip_blank_lines(body_lines))
        lines.append('    break;')

    lines.extend([
        '',
        'default:',
        '    break;',
        '}'])

    return [('    ' + line).rstrip() for line in lines]


def _format_unpack_code_signal(cg_message: "CodeGenMessage",
                               signal_name: str,
                               body_lines: List[str],
                               variable_lines: List[str],
                               helper_kinds: Set[THelperKind]) -> None:
    cg_signal = cg_message.get_signal_by_name(signal_name)
    conversion_type_name = f'uint{cg_signal.type_length}_t'

    if cg_signal.signal.conversion.is_float or cg_signal.signal.is_signed:
        variable = f'    {conversion_type_name} {cg_signal.snake_name};'
        variable_lines.append(variable)

    segments = cg_signal.segments(invert_shift=True)

    for i, (index, shift, shift_direction, mask) in enumerate(segments):
        if cg_signal.signal.conversion.is_float or cg_signal.signal.is_signed:
            fmt = '    {} {} unpack_{}_shift_u{}(src_p[{}], {}u, 0x{:02x}u);'
        else:
            fmt = '    dst_p->{} {} unpack_{}_shift_u{}(src_p[{}], {}u, 0x{:02x}u);'

        line = fmt.format(cg_signal.snake_name,
                          '=' if i == 0 else '|=',
                          shift_direction,
                          cg_signal.type_length,
                          index,
                          shift,
                          mask)
        body_lines.append(line)
        helper_kinds.add((shift_direction, cg_signal.type_length))

    if cg_signal.signal.conversion.is_float:
        conversion = '    memcpy(&dst_p->{0}, &{0}, sizeof(dst_p->{0}));'.format(
            cg_signal.snake_name)
        body_lines.append(conversion)
    elif cg_signal.signal.is_signed:
        mask = ((1 << (cg_signal.type_length - cg_signal.signal.length)) - 1)

        if mask != 0:
            mask <<= cg_signal.signal.length
            formatted = SIGN_EXTENSION_FMT.format(name=cg_signal.snake_name,
                                                  shift=cg_signal.signal.length - 1,
                                                  mask=mask,
                                                  suffix=cg_signal.conversion_type_suffix)
            body_lines.extend(formatted.splitlines())

        conversion = '    dst_p->{0} = (int{1}_t){0};'.format(cg_signal.snake_name,
                                                              cg_signal.type_length)
        body_lines.append(conversion)


def _format_unpack_code_level(cg_message: "CodeGenMessage",
                              signal_names: Union[List[str], List[Dict[str, Dict[int, List[str]]]]],
                              variable_lines: List[str],
                              helper_kinds: Set[THelperKind],
                              node_name: Optional[str]) -> List[str]:
    """Format one unpack level in a signal tree.

    """

    body_lines: List[str] = []
    muxes_lines: List[str] = []

    for signal_name in signal_names:
        if isinstance(signal_name, dict):
            mux_lines = _format_unpack_code_mux(cg_message,
                                                signal_name,
                                                body_lines,
                                                variable_lines,
                                                helper_kinds,
                                                node_name)

            if muxes_lines:
                muxes_lines.append('')

            muxes_lines += mux_lines
        else:
            if not _is_receiver(cg_message.get_signal_by_name(signal_name), node_name):
                continue

            _format_unpack_code_signal(cg_message,
                                       signal_name,
                                       body_lines,
                                       variable_lines,
                                       helper_kinds)

    if body_lines:
        if body_lines[-1] != '':
            body_lines.append('')

    if muxes_lines:
        muxes_lines.append('')

    body_lines = body_lines + muxes_lines

    if body_lines:
        body_lines = [''] + body_lines

    return body_lines


def _format_unpack_code(cg_message: "CodeGenMessage",
                        helper_kinds: Set[THelperKind],
                        node_name: Optional[str]) -> Tuple[str, str]:
    variable_lines: List[str] = []
    body_lines = _format_unpack_code_level(cg_message,
                                           cg_message.message.signal_tree,
                                           variable_lines,
                                           helper_kinds,
                                           node_name)

    if variable_lines:
        variable_lines = sorted(set(variable_lines)) + ['', '']

    return '\n'.join(variable_lines), '\n'.join(body_lines)


def _generate_struct(cg_message: "CodeGenMessage", bit_fields: bool) -> Tuple[str, List[str]]:
    members = []

    for cg_signal in cg_message.cg_signals:
        members.append(_generate_signal(cg_signal, bit_fields))

    if cg_message.message.comment is None:
        comment = ''
    else:
        comment = cg_message.message.comment

    return comment, members


def _generate_encode_decode(cg_signal: "CodeGenSignal", use_float: bool) -> Tuple[str, str]:
    floating_point_type = _get_floating_point_type(use_float)

    scale = cg_signal.signal.scale
    offset = cg_signal.signal.offset

    scale_literal = f"{scale}{'.0' if isinstance(scale, int) else ''}{'f' if use_float else ''}"
    offset_literal = f"{offset}{'.0' if isinstance(offset, int) else ''}{'f' if use_float else ''}"

    if offset == 0 and scale == 1:
        encoding = 'value'
        decoding = f'({floating_point_type})value'
    elif offset != 0 and scale != 1:
        encoding = f'(value - {offset_literal}) / {scale_literal}'
        decoding = f'(({floating_point_type})value * {scale_literal}) + {offset_literal}'
    elif offset != 0:
        encoding = f'value - {offset_literal}'
        decoding = f'({floating_point_type})value + {offset_literal}'
    else:
        encoding = f'value / {scale_literal}'
        decoding = f'({floating_point_type})value * {scale_literal}'

    return encoding, decoding


def _generate_is_in_range(cg_signal: "CodeGenSignal") -> str:
    """Generate range checks for all signals in given message.

    """
    minimum = cg_signal.signal.minimum
    maximum = cg_signal.signal.maximum

    if minimum is not None:
        minimum = cg_signal.signal.scaled_to_raw(minimum)

    if maximum is not None:
        maximum = cg_signal.signal.scaled_to_raw(maximum)

    if minimum is None and cg_signal.minimum_can_raw_value is not None:
        if cg_signal.minimum_ctype_value is None:
            minimum = cg_signal.minimum_can_raw_value
        elif cg_signal.minimum_can_raw_value > cg_signal.minimum_ctype_value:
            minimum = cg_signal.minimum_can_raw_value

    if maximum is None and cg_signal.maximum_can_raw_value is not None:
        if cg_signal.maximum_ctype_value is None:
            maximum = cg_signal.maximum_can_raw_value
        elif cg_signal.maximum_can_raw_value < cg_signal.maximum_ctype_value:
            maximum = cg_signal.maximum_can_raw_value

    suffix = cg_signal.type_suffix
    check = []

    if minimum is not None:
        if not cg_signal.signal.conversion.is_float:
            minimum = round(minimum)
        else:
            minimum = float(minimum)

        minimum_ctype_value = cg_signal.minimum_ctype_value

        if (minimum_ctype_value is None) or (minimum > minimum_ctype_value):
            check.append(f'(value >= {minimum}{suffix})')

    if maximum is not None:
        if not cg_signal.signal.conversion.is_float:
            maximum = round(maximum)
        else:
            maximum = float(maximum)

        maximum_ctype_value = cg_signal.maximum_ctype_value

        if (maximum_ctype_value is None) or (maximum < maximum_ctype_value):
            check.append(f'(value <= {maximum}{suffix})')

    if not check:
        check = ['true']
    elif len(check) == 1:
        check = [check[0][1:-1]]

    return ' && '.join(check)


def _generate_frame_defines(cg_messages: List["CodeGenMessage"],
                               node_name: Optional[str]) -> list:
    return [cg_message for cg_message in cg_messages if _is_sender_or_receiver(cg_message, node_name)]


def _generate_structs(cg_messages: List["CodeGenMessage"],
                      bit_fields: bool,
                      node_name: Optional[str]) -> str:
    structs = []

    for cg_message in cg_messages:
        if _is_sender_or_receiver(cg_message, node_name):
            comment, members = _generate_struct(cg_message, bit_fields)
            structs.append({'comment': comment,
                            'database_message_name': cg_message.message.name,
                            'message_name': cg_message.snake_name,
                            'members': members})

    return structs


def _is_sender(cg_message: "CodeGenMessage", node_name: Optional[str]) -> bool:
    return node_name is None or node_name in cg_message.message.senders


def _is_receiver(cg_signal: "CodeGenSignal", node_name: Optional[str]) -> bool:
    return node_name is None or node_name in cg_signal.signal.receivers


def _is_sender_or_receiver(cg_message: "CodeGenMessage", node_name: Optional[str]) -> bool:
    if _is_sender(cg_message, node_name):
        return True
    return any(_is_receiver(cg_signal, node_name) for cg_signal in cg_message.cg_signals)


def _get_floating_point_type(use_float: bool) -> str:
    return 'float' if use_float else 'double'


def _generate_declarations(cg_messages: List["CodeGenMessage"],
                           floating_point_numbers: bool,
                           use_float: bool,
                           node_name: Optional[str]) -> str:
    declarations = []

    for cg_message in cg_messages:
        signal_declarations = []
        is_sender = _is_sender(cg_message, node_name)
        is_receiver = node_name is None

        for cg_signal in cg_message.cg_signals:
            signal_declaration = {
                'signal_name': cg_signal.snake_name,
                'type_name': cg_signal.type_name,
                'floating_point_type': _get_floating_point_type(use_float),
            }

            if is_sender or _is_receiver(cg_signal, node_name):
                signal_declarations.append(signal_declaration)

        declaration = {
            'database_message_name': cg_message.message.name,
            'message_name': cg_message.snake_name,
            'sender': is_sender,
            'receiver': is_receiver,
            'signal_declarations': signal_declarations,
            'floating_point_numbers': floating_point_numbers,
            'node_name': node_name,
        }

        if declaration:
            declarations.append(declaration)

    return declarations


def _generate_definitions(database_name: str,
                          cg_messages: List["CodeGenMessage"],
                          floating_point_numbers: bool,
                          use_float: bool,
                          node_name: Optional[str],
                          ) -> Tuple[str, Tuple[Set[THelperKind], Set[THelperKind]]]:
    definitions = []
    pack_helper_kinds: Set[THelperKind] = set()
    unpack_helper_kinds: Set[THelperKind] = set()

    for cg_message in cg_messages:
        signal_definitions = []
        is_sender = _is_sender(cg_message, node_name)
        is_receiver = node_name is None
        signals_init_body = ''

        for cg_signal in cg_message.cg_signals:
            if use_float and cg_signal.type_name == "double":
                warnings.warn(f"User selected `--use-float`, but database contains "
                              f"signal with data type `double`: "
                              f"\"{cg_message.message.name}::{cg_signal.signal.name}\"",
                              stacklevel=2)
                _use_float = False
            else:
                _use_float = use_float

            encode, decode = _generate_encode_decode(cg_signal, _use_float)
            check = _generate_is_in_range(cg_signal)

            if _is_receiver(cg_signal, node_name):
                is_receiver = True

            if check == 'true':
                unused = '    (void)value;\n\n'
            else:
                unused = ''

            signal_definition = ''

            if floating_point_numbers:
                if is_sender:
                    signal_definition += SIGNAL_DEFINITION_ENCODE_FMT.format(
                        database_name=database_name,
                        message_name=cg_message.snake_name,
                        signal_name=cg_signal.snake_name,
                        type_name=cg_signal.type_name,
                        encode=encode,
                        floating_point_type=_get_floating_point_type(_use_float))
                if node_name is None or _is_receiver(cg_signal, node_name):
                    signal_definition += SIGNAL_DEFINITION_DECODE_FMT.format(
                        database_name=database_name,
                        message_name=cg_message.snake_name,
                        signal_name=cg_signal.snake_name,
                        type_name=cg_signal.type_name,
                        decode=decode,
                        floating_point_type=_get_floating_point_type(_use_float))

            if is_sender or _is_receiver(cg_signal, node_name):
                signal_definition += SIGNAL_DEFINITION_IS_IN_RANGE_FMT.format(
                    database_name=database_name,
                    message_name=cg_message.snake_name,
                    signal_name=cg_signal.snake_name,
                    type_name=cg_signal.type_name,
                    unused=unused,
                    check=check)

                signal_definitions.append(signal_definition)

            if cg_signal.signal.initial:
                signals_init_body += INIT_SIGNAL_BODY_TEMPLATE_FMT.format(signal_initial=cg_signal.signal.raw_initial,
                                                                          signal_name=cg_signal.snake_name)

        if cg_message.message.length > 0:
            pack_variables, pack_body = _format_pack_code(cg_message,
                                                          pack_helper_kinds)
            unpack_variables, unpack_body = _format_unpack_code(cg_message,
                                                                unpack_helper_kinds,
                                                                node_name)
            pack_unused = ''
            unpack_unused = ''

            if not pack_body:
                pack_unused += '    (void)src_p;\n\n'

            if not unpack_body:
                unpack_unused += '    (void)dst_p;\n'
                unpack_unused += '    (void)src_p;\n\n'

            definition = ""
            if is_sender:
                definition += DEFINITION_PACK_FMT.format(database_name=database_name,
                                                         database_message_name=cg_message.message.name,
                                                         message_name=cg_message.snake_name,
                                                         message_length=cg_message.message.length,
                                                         pack_unused=pack_unused,
                                                         pack_variables=pack_variables,
                                                         pack_body=pack_body)
            if is_receiver:
                definition += DEFINITION_UNPACK_FMT.format(database_name=database_name,
                                                           database_message_name=cg_message.message.name,
                                                           message_name=cg_message.snake_name,
                                                           message_length=cg_message.message.length,
                                                           unpack_unused=unpack_unused,
                                                           unpack_variables=unpack_variables,
                                                           unpack_body=unpack_body)

            definition += MESSAGE_DEFINITION_INIT_FMT.format(database_name=database_name,
                                                             database_message_name=cg_message.message.name,
                                                             message_name=cg_message.snake_name,
                                                             init_body=signals_init_body)

        else:
            definition = EMPTY_DEFINITION_FMT.format(database_name=database_name,
                                                     message_name=cg_message.snake_name)

        if signal_definitions:
            definition += '\n' + '\n'.join(signal_definitions)

        if definition:
            definitions.append(definition)

    return '\n'.join(definitions), (pack_helper_kinds, unpack_helper_kinds)


def _generate_fuzzer_source(jinja_env: Environment,
                            database_name: str,
                            cg_messages: List["CodeGenMessage"],
                            date: str,
                            header_name: str,
                            source_name: str,
                            fuzzer_source_name: str) -> Tuple[str, str]:
    tests = []

    for cg_message in cg_messages:
        name = f'{database_name}_{camel_to_snake_case(cg_message.message.name)}'

        tests.append(name)

    fuzzer_source_template = jinja_env.get_template('fuzzer.c.jinja')
    source = fuzzer_source_template.render(version=__version__,
                                           date=date,
                                           header=header_name,
                                           tests=tests)

    makefile_template = jinja_env.get_template('fuzzer.mk.jinja')
    makefile = makefile_template.render(version=__version__,
                                        date=date,
                                        source=source_name,
                                        fuzzer_source=fuzzer_source_name)

    return source, makefile


def generate(database: "Database",
             database_name: str,
             header_name: str,
             source_name: str,
             fuzzer_source_name: str,
             floating_point_numbers: bool = True,
             bit_fields: bool = False,
             use_float: bool = False,
             node_name: Optional[str] = None,
             ) -> Tuple[str, str, str, str]:
    """Generate C source code from given CAN database `database`.

    `database_name` is used as a prefix for all defines, data
    structures and functions.

    `header_name` is the file name of the C header file, which is
    included by the C source file.

    `source_name` is the file name of the C source file, which is
    needed by the fuzzer makefile.

    `fuzzer_source_name` is the file name of the C source file, which
    is needed by the fuzzer makefile.

    Set `floating_point_numbers` to ``True`` to allow floating point
    numbers in the generated code.

    Set `bit_fields` to ``True`` to generate bit fields in structs.

    Set `use_float` to ``True`` to prefer the `float` type instead
    of the `double` type for floating point numbers.

    `node_name` specifies the node for which message packers will be generated.
    For all other messages, unpackers will be generated. If `node_name` is not
    provided, both packers and unpackers will be generated.

    This function returns a tuple of the C header and source files as
    strings.

    """

    date=time.ctime()
    cg_messages = [CodeGenMessage(message) for message in database.messages]
    frame_defines = _generate_frame_defines(cg_messages, node_name)

    structs = _generate_structs(cg_messages, bit_fields, node_name)
    declarations = _generate_declarations(cg_messages,
                                          floating_point_numbers,
                                          use_float,
                                          node_name)
    definitions, helper_kinds = _generate_definitions(database_name,
                                                      cg_messages,
                                                      floating_point_numbers,
                                                      use_float,
                                                      node_name)
    pack_helper_kinds, unpack_helper_kinds = helper_kinds

    dir_path = os.path.dirname(os.path.realpath(__file__))
    loader = FileSystemLoader(os.path.join(dir_path, 'templates'))
    jinja_env = Environment(loader=loader)

    def prefix_lines(value, prefix):
        return '\n'.join(f'{prefix}{line}' for line in value.splitlines())

    # Add the filter to the environment
    jinja_env.filters['prefix_lines'] = prefix_lines

    header_template = jinja_env.get_template('header.h.jinja')
    header = header_template.render(version=__version__,
                                    date=date,
                                    database_name=database_name,
                                    frame_defines=frame_defines,
                                    structs=structs,
                                    declarations=declarations)

    source_template = jinja_env.get_template('source.c.jinja')
    source = source_template.render(version=__version__,
                                    date=date,
                                    header=header_name,
                                    pack_helpers=pack_helper_kinds,
                                    unpack_helpers=unpack_helper_kinds,
                                    definitions=definitions)

    fuzzer_source, fuzzer_makefile = _generate_fuzzer_source(
        jinja_env,
        database_name,
        cg_messages,
        date,
        header_name,
        source_name,
        fuzzer_source_name)

    return header, source, fuzzer_source, fuzzer_makefile
