import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:onnxruntime/onnxruntime.dart';

void checkTypedData({required bool web}) {
  final cases = <List>[
    Uint8List.fromList([0, 255]),
    Int8List.fromList([-128, 127]),
    Uint16List.fromList([0, 65535]),
    Int16List.fromList([-32768, 32767]),
    Uint32List.fromList([0, 4294967295]),
    Int32List.fromList([-2147483648, 2147483647]),
    Float32List.fromList([-1.25, 3.5]),
    Float64List.fromList([-1.25, 1e100]),
    if (!web)
      Int64List.fromList([
        int.parse('-9223372036854775808'),
        int.parse('9223372036854775807'),
      ]),
    if (!web) Uint64List.fromList([0, -1]),
  ];
  for (final data in cases) {
    final tensor = OrtValueTensor.createTensorWithDataList(data, [1, 2]);
    try {
      final copy = tensor.toTypedData();
      expect(copy.runtimeType, data.runtimeType);
      expect(copy.lengthInBytes, (data as TypedData).lengthInBytes);
      expect(copy, data);
      final other = tensor.toTypedData();
      if (copy is Float32List) {
        copy[0] = 0.0;
      } else if (copy is Float64List) {
        copy[0] = 0.0;
      } else {
        (copy as dynamic)[0] = 0;
      }
      expect(other, data, reason: 'Each extraction owns its buffer');
      expect(tensor.value, [data], reason: 'Extraction must not mutate value');
      tensor.release();
      expect(other, data, reason: 'Copy remains valid after tensor release');
      expect(() => tensor.toTypedData(), throwsStateError);
    } finally {
      tensor.release();
    }
  }
  for (final data in <List>[Float32List(0), Int32List(0)]) {
    final tensor = OrtValueTensor.createTensorWithDataList(data, [0, 2]);
    try {
      expect(tensor.toTypedData().lengthInBytes, 0);
      expect(tensor.shape, [0, 2]);
    } finally {
      tensor.release();
    }
  }
  final scalar = OrtValueTensor.createTensorWithDataList(
    Float32List.fromList([2.5]),
    [],
  );
  try {
    expect(scalar.toTypedData(), Float32List.fromList([2.5]));
    expect(scalar.value, 2.5);
  } finally {
    scalar.release();
  }
  for (final data in <List>[
    [true, false],
    ['hello', '世界'],
    if (web) [1, 2],
  ]) {
    final tensor = OrtValueTensor.createTensorWithDataList(data);
    try {
      expect(() => tensor.toTypedData(), throwsUnsupportedError);
      expect(tensor.value, data, reason: 'Legacy extraction remains available');
    } finally {
      tensor.release();
    }
  }
}

Future<void> checkTypedInference(OrtSession session) async {
  final options = OrtRunOptions();
  final input = OrtValueTensor.createTensorWithDataList(
    Float32List.fromList([1, 2]),
    [1, 2],
  );
  try {
    final outputs = await session.runAsyncOrThrow(options, {'input': input});
    late TypedData copy;
    try {
      final output = outputs.single as OrtValueTensor;
      copy = output.toTypedData();
      expect(copy, isA<Float32List>());
      expect(copy, [1, 2]);
      expect(output.value, [
        [1.0, 2.0],
      ]);
    } finally {
      for (final value in outputs) {
        value?.release();
      }
    }
    expect(copy, [1, 2]);
  } finally {
    options.release();
    input.release();
    session.release();
  }
}
