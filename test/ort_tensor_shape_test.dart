import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:onnxruntime/onnxruntime.dart';

void main() {
  test('inferred shapes reject ragged nesting even with matching total size',
      () {
    for (final data in <List>[
      [
        [1],
        [2, 3]
      ],
      [
        [1, 2],
        [3],
        [4, 5, 6]
      ],
      [
        [1],
        2
      ],
      [
        1,
        [2]
      ],
      [
        [1],
        []
      ],
      [
        ['a'],
        ['b', 'c']
      ],
      [
        [true],
        [false, true]
      ],
      [
        Float32List.fromList([1]),
        Float32List.fromList([2, 3])
      ],
    ]) {
      expect(() => OrtValueTensor.createTensorWithDataList(data),
          throwsArgumentError,
          reason: '$data');
    }
  });

  test('explicit numeric and bool shapes cannot discard extra elements', () {
    for (final data in <List>[
      [1, 2],
      [1.5, 2.5],
      [true, false],
      Uint8List.fromList([1, 2]),
      Int8List.fromList([1, 2]),
      Uint16List.fromList([1, 2]),
      Int16List.fromList([1, 2]),
      Uint32List.fromList([1, 2]),
      Int32List.fromList([1, 2]),
      Uint64List.fromList([1, 2]),
      Int64List.fromList([1, 2]),
      Float32List.fromList([1, 2]),
      Float64List.fromList([1, 2]),
    ]) {
      for (final shape in <List<int>>[
        [],
        [1],
        [0],
        [1, 1]
      ]) {
        expect(() => OrtValueTensor.createTensorWithDataList(data, shape),
            throwsArgumentError,
            reason: '${data.runtimeType}: $shape');
      }
    }
  });

  test('string shapes cannot leave unfilled elements', () {
    for (final shape in <List<int>>[
      [2],
      [1, 2],
      [2, 2]
    ]) {
      expect(() => OrtValueTensor.createTensorWithDataList(['a'], shape),
          throwsArgumentError);
    }
    // Multiplication must not wrap an enormous shape to a small valid count.
    expect(
        () =>
            OrtValueTensor.createTensorWithDataList(['a'], [1 << 32, 1 << 32]),
        throwsArgumentError);
  });

  test('explicit shapes still flatten and reshape nested inputs in order', () {
    for (final data in <List>[
      [
        [1],
        [2, 3]
      ],
      [
        ['a'],
        ['b', 'c']
      ],
      [
        [true],
        [false, true]
      ],
    ]) {
      final tensor = OrtValueTensor.createTensorWithDataList(data, [1, 3]);
      try {
        expect(tensor.value, [
          [...data[0], ...data[1]]
        ]);
      } finally {
        tensor.release();
      }
    }
  });

  test('rectangular inferred shapes and scalar representations are unchanged',
      () {
    for (final data in <List>[
      [
        [1, 2],
        [3, 4]
      ],
      [
        ['a', ''],
        ['中文', '🧠']
      ],
      [
        [true, false],
        [false, true]
      ],
    ]) {
      final tensor = OrtValueTensor.createTensorWithDataList(data);
      try {
        expect(tensor.value, data);
      } finally {
        tensor.release();
      }
    }
    for (final data in [1, 1.5, true, 'a']) {
      final tensor = OrtValueTensor.createTensorWithData(data);
      try {
        expect(tensor.value, data);
      } finally {
        tensor.release();
      }
    }
  });

  test('previous native shape failures retain their exception behavior', () {
    for (final data in <List>[
      [1, 2],
      ['a', 'b']
    ]) {
      final shape = data.first is String ? [1] : [3];
      expect(() => OrtValueTensor.createTensorWithDataList(data, shape),
          throwsException);
      expect(() => OrtValueTensor.createTensorWithDataList(data, [-1]),
          throwsException);
    }
  });
}
