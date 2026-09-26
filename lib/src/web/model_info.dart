import 'dart:convert';
import 'dart:typed_data';

/// Reads only graph names and custom metadata from the ONNX protobuf. Weights
/// are skipped without copying. This keeps synchronous constructor/getter APIs
/// while ONNX Runtime Web creates its actual session asynchronously on first run.
class ModelInfo {
  final inputs = <String>[];
  final outputs = <String>[];
  final metadata = <String, String>{};
  ModelInfo._();

  factory ModelInfo.read(Uint8List bytes) {
    final info = ModelInfo._();
    final reader = _Proto(bytes);
    var hasGraph = false;
    while (reader.next()) {
      if (reader.field == 7 && reader.wire == 2) {
        if (hasGraph) throw const FormatException('Duplicate ONNX graph.');
        hasGraph = true;
        info._graph(reader.message());
      } else if (reader.field == 14 && reader.wire == 2) {
        final entry = _Proto(reader.message());
        var key = '', value = '';
        while (entry.next()) {
          if (entry.field == 1 && entry.wire == 2) {
            key = utf8.decode(entry.message());
          } else if (entry.field == 2 && entry.wire == 2) {
            value = utf8.decode(entry.message());
          } else {
            entry.skip();
          }
        }
        info.metadata[key] = value;
      } else {
        reader.skip();
      }
    }
    if (!hasGraph) {
      throw const FormatException(
        'Web requires an ONNX protobuf model with a graph.',
      );
    }
    return info;
  }

  void _graph(Uint8List bytes) {
    final graph = _Proto(bytes);
    final initializers = <String>{};
    while (graph.next()) {
      if (graph.wire == 2) {
        switch (graph.field) {
          case 11:
            inputs.add(_name(graph.message(), 1));
            continue;
          case 12:
            outputs.add(_name(graph.message(), 1));
            continue;
          case 5:
            initializers.add(_name(graph.message(), 8));
            continue;
          case 15:
            final sparse = _Proto(graph.message());
            while (sparse.next()) {
              if (sparse.field == 1 && sparse.wire == 2) {
                initializers.add(_name(sparse.message(), 8));
              } else {
                sparse.skip();
              }
            }
            continue;
        }
      }
      graph.skip();
    }
    inputs.removeWhere(initializers.contains);
  }

  static String _name(Uint8List bytes, int field) {
    final reader = _Proto(bytes);
    while (reader.next()) {
      if (reader.field == field && reader.wire == 2) {
        return utf8.decode(reader.message());
      }
      reader.skip();
    }
    throw const FormatException('Missing ONNX value name.');
  }
}

class _Proto {
  final Uint8List bytes;
  int offset = 0;
  int field = 0;
  int wire = 0;
  _Proto(this.bytes);

  int _byte() {
    if (offset >= bytes.length) {
      throw const FormatException('Truncated ONNX model.');
    }
    return bytes[offset++];
  }

  // Tags and lengths are bounded to uint32; avoid JavaScript's 32-bit shifts.
  int _uint32() {
    var value = 0, multiplier = 1;
    for (var i = 0; i < 5; i++) {
      final byte = _byte();
      value += (byte & 127) * multiplier;
      if (value > 0xffffffff) {
        throw const FormatException('Invalid protobuf length/tag.');
      }
      if (byte < 128) return value;
      multiplier *= 128;
    }
    throw const FormatException('Invalid protobuf varint.');
  }

  bool next() {
    if (offset == bytes.length) return false;
    final tag = _uint32();
    field = tag ~/ 8;
    wire = tag % 8;
    if (field == 0) throw const FormatException('Invalid protobuf field.');
    return true;
  }

  void _advance(int length) {
    if (length < 0 || length > bytes.length - offset) {
      throw const FormatException('Truncated ONNX model.');
    }
    offset += length;
  }

  Uint8List message() {
    final length = _uint32();
    final start = offset;
    _advance(length);
    return Uint8List.sublistView(bytes, start, offset);
  }

  void skip() {
    switch (wire) {
      case 0:
        for (var i = 0; i < 10; i++) {
          final byte = _byte();
          if (i == 9 && byte > 1) {
            throw const FormatException('Invalid protobuf varint.');
          }
          if (byte < 128) return;
        }
        throw const FormatException('Invalid protobuf varint.');
      case 1:
        _advance(8);
        return;
      case 2:
        _advance(_uint32());
        return;
      case 5:
        _advance(4);
        return;
      default:
        throw const FormatException('Unsupported protobuf wire type.');
    }
  }
}
