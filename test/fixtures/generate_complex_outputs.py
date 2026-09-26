"""Generate tiny ONNX fixtures using only Python's standard library.

Wire fields follow https://github.com/onnx/onnx/blob/v1.14.0/onnx/onnx.proto.
Fixtures use IR 8 / opset 13 or 16 (sequence Identity requires opset 16).
Run: python3 test/fixtures/generate_complex_outputs.py
"""
from pathlib import Path


def varint(value):
    result = bytearray()
    while value > 127:
        result.append((value & 127) | 128)
        value >>= 7
    return bytes(result + bytes([value]))


def integer(field, value):
    return varint(field << 3) + varint(value)


def message(field, value):
    if isinstance(value, str):
        value = value.encode('utf-8')
    return varint((field << 3) | 2) + varint(len(value)) + value


def tensor_type(shape):
    dimensions = b''.join(message(1, integer(1, d)) for d in shape)
    return message(1, integer(1, 1) + message(2, dimensions))  # FLOAT


def sequence_type(element_type):
    return message(4, message(1, element_type))


def value_info(name, value_type):
    return message(1, name) + message(2, value_type)


def node(op, inputs, output, attributes=b'', domain=''):
    return (b''.join(message(1, name) for name in inputs)
            + message(2, output) + message(4, op) + attributes
            + (message(7, domain) if domain else b''))


def model(name, nodes, outputs, ml=False, metadata=(), inputs=None, opset=13):
    if inputs is None:
        inputs = [('input', tensor_type([1, 2]))]
    graph = (b''.join(message(1, n) for n in nodes) + message(2, name)
             + b''.join(message(11, value_info(n, t)) for n, t in inputs)
             + b''.join(message(12, value_info(n, t)) for n, t in outputs))
    result = integer(1, 8) + message(7, graph) + message(8, integer(2, opset))
    if ml:
        result += message(8, message(1, 'ai.onnx.ml') + integer(2, 1))
    for key, value in metadata:
        result += message(14, message(1, key) + message(2, value))
    Path(__file__).with_name(name + '.onnx').write_bytes(result)


model('tensor_sequence', [
    node('SequenceConstruct', ['input', 'input'], 'sequence'),
    node('Identity', ['input'], 'tensor'),
], [('sequence', sequence_type(tensor_type([1, 2]))),
    ('tensor', tensor_type([1, 2]))])

# ZipMap maps each input row to class IDs 10 and 20.
labels = (message(1, 'classlabels_int64s') + integer(20, 7)
          + integer(8, 10) + integer(8, 20))
map_type = message(5, integer(1, 7) + message(2, tensor_type([])))
model('map_sequence', [
    node('ZipMap', ['input'], 'maps', message(5, labels), 'ai.onnx.ml'),
], [('maps', sequence_type(map_type))], ml=True)

model('metadata', [node('Identity', ['input'], 'output')],
      [('output', tensor_type([1, 2]))], metadata=[
          ('author', 'onnxruntime_flutter'), ('empty', ''),
          ('说明🧠', '中文元数据🧠'), ('nul', 'prefix\0suffix'),
      ])

# Mixed complex/tensor inputs exercise worker restoration and output ordering.
sequence = sequence_type(tensor_type([1, 2]))
model('sequence_input', [
    node('Identity', ['input'], 'sequence'),
    node('Identity', ['tensor_input'], 'tensor'),
], [('sequence', sequence), ('tensor', tensor_type([1, 2]))],
    inputs=[('input', sequence), ('tensor_input', tensor_type([1, 2]))], opset=16)

# A graph input can also be an output without requiring a map-specific operator.
model('map_input', [node('Identity', ['tensor_input'], 'tensor')],
      [('input', map_type), ('tensor', tensor_type([1, 2]))],
      inputs=[('input', map_type), ('tensor_input', tensor_type([1, 2]))])

sparse = message(8, integer(1, 1) + message(2,
    message(1, integer(1, 2)) + message(1, integer(1, 2))))
model('sparse_input', [], [('input', sparse)], inputs=[('input', sparse)])

model('empty_sequence', [
    node('SequenceEmpty', [], 'sequence'),
    node('Identity', ['input'], 'tensor'),
], [('sequence', sequence), ('tensor', tensor_type([1, 2]))])

# Symbolic, anonymous dynamic, scalar and empty dimensions for model inspection.
dynamic_type = message(1, integer(1, 1) + message(2,
    message(1, message(2, 'batch')) + message(1, integer(1, 2))))
model('dynamic_identity', [node('Identity', ['input'], 'output')],
      [('output', dynamic_type)], inputs=[('input', dynamic_type)])
model('scalar_identity', [node('Identity', ['input'], 'output')],
      [('output', tensor_type([]))], inputs=[('input', tensor_type([]))])

# Half precision Identity fixtures retain tensor bits through native/Web inference.
for name, dtype in [('float16', 10), ('bfloat16', 16)]:
    half_type = message(1, integer(1, dtype) + message(2, message(1, integer(1, 2))))
    model(name + '_identity', [node('Identity', ['input'], 'output')],
          [('output', half_type)], inputs=[('input', half_type)])
