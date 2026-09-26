import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:onnxruntime/src/web/model_info.dart';

List<int> _varint(int value) {
  final bytes = <int>[];
  while (value > 127) {
    bytes.add((value & 127) | 128);
    value >>= 7;
  }
  return [...bytes, value];
}

List<int> _message(int field, List<int> bytes) => [
  ..._varint(field * 8 + 2),
  ..._varint(bytes.length),
  ...bytes,
];
List<int> _text(int field, String text) => _message(field, utf8.encode(text));
Uint8List _model(List<int> graph) => Uint8List.fromList(_message(7, graph));

void main() {
  test('model parser reads real names and Unicode metadata', () {
    final info = ModelInfo.read(
      File('test/fixtures/metadata.onnx').readAsBytesSync(),
    );
    expect(info.inputs, ['input']);
    expect(info.outputs, ['output']);
    expect(info.metadata['说明🧠'], '中文元数据🧠');
    expect(info.metadata['empty'], '');
    expect(info.metadata['nul'], 'prefix\u0000suffix');
  });

  test('initializers are excluded regardless of graph field order', () {
    final info = ModelInfo.read(
      _model([
        ..._message(11, _text(1, 'input')),
        ..._message(11, _text(1, 'weight')),
        ..._message(11, _text(1, 'sparse')),
        ..._message(12, _text(1, 'second')),
        ..._message(12, _text(1, 'first')),
        ..._message(5, [
          ..._text(8, 'weight'),
          ..._message(9, List.filled(1024, 0)),
        ]),
        ..._message(15, _message(1, _text(8, 'sparse'))),
      ]),
    );
    expect(info.inputs, ['input']);
    expect(info.outputs, ['second', 'first']);
  });

  test('unknown protobuf fields are skipped at each supported wire width', () {
    final info = ModelInfo.read(
      Uint8List.fromList([
        8, ...List.filled(9, 255), 1, // valid 64-bit varint
        17, ...List.filled(8, 0),
        29, ...List.filled(4, 0),
        ..._text(2, 'producer'),
        ..._model(_message(11, _text(1, 'input'))),
      ]),
    );
    expect(info.inputs, ['input']);
  });

  test(
    'truncated and overflowing protobuf data is rejected deterministically',
    () {
      final valid = _model(_message(11, _text(1, 'input')));
      for (var length = 0; length < valid.length; length++) {
        expect(
          () => ModelInfo.read(Uint8List.sublistView(valid, 0, length)),
          throwsFormatException,
        );
      }
      for (final bytes in <List<int>>[
        [0],
        [58, 255, 255, 255, 255, 31],
        [8, ...List.filled(10, 255)],
        [11],
        [..._model([]), ..._model([])],
        _model(_message(11, [])),
      ]) {
        expect(
          () => ModelInfo.read(Uint8List.fromList(bytes)),
          throwsFormatException,
        );
      }
    },
  );
}
