import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:onnxruntime/onnxruntime.dart';

void main() {
  const rows = [
    ['first', '中文'],
    ['', 'emoji🧠']
  ];
  for (final explicitShape in [false, true]) {
    for (final dimensions in [2, 3]) {
      test('$dimensions-D strings retain order (explicit: $explicitShape)', () {
        final data = dimensions == 2 ? rows : [rows, rows];
        final shape = dimensions == 2 ? [2, 2] : [2, 2, 2];
        final tensor = OrtValueTensor.createTensorWithDataList(
            data, explicitShape ? shape : null);
        try {
          expect(tensor.value, data);
        } finally {
          tensor.release();
        }
      });
    }
  }

  test('explicit shape can reshape nested strings', () {
    final tensor = OrtValueTensor.createTensorWithDataList(rows, [1, 4]);
    try {
      expect(tensor.value, [
        ['first', '中文', '', 'emoji🧠']
      ]);
    } finally {
      tensor.release();
    }
  });

  test('scalar and flat string inputs retain their value representations', () {
    final scalar = OrtValueTensor.createTensorWithData('中文🧠');
    final flat = OrtValueTensor.createTensorWithDataList(['first', '']);
    final reshaped =
        OrtValueTensor.createTensorWithDataList(['first', ''], [1, 2]);
    try {
      expect(scalar.value, '中文🧠');
      expect(flat.value, ['first', '']);
      expect(reshaped.value, [
        ['first', '']
      ]);
    } finally {
      scalar.release();
      flat.release();
      reshaped.release();
    }
  });

  test('flat mixed string inputs still throw a type error', () {
    for (final invalid in [1, null, true]) {
      expect(() => OrtValueTensor.createTensorWithDataList(['first', invalid]),
          throwsA(isA<TypeError>()));
    }
  });

  test('nested strings match flat input in synchronous and async inference',
      () async {
    final options = OrtSessionOptions()..setIntraOpNumThreads(1);
    final session = OrtSession.fromFile(
        File('example/assets/models/test_types_STRING.pb'), options);
    final runOptions = OrtRunOptions();
    const strings = ['first', '中文', '', 'emoji🧠', 'last'];
    final flat = OrtValueTensor.createTensorWithDataList(strings, [1, 5]);
    final outputs = <OrtValue?>[];
    OrtValueTensor? nested;
    try {
      nested = OrtValueTensor.createTensorWithDataList([strings]);
      final reference = session.run(runOptions, {'input': flat});
      outputs.addAll(reference);
      final sync = session.run(runOptions, {'input': nested});
      outputs.addAll(sync);
      final async = await session.runAsync(runOptions, {'input': nested})!;
      outputs.addAll(async);
      expect(reference, isNotEmpty);
      final expected = reference.map((value) => value!.value).toList();
      expect(sync.map((value) => value!.value).toList(), expected);
      expect(async.map((value) => value!.value).toList(), expected);
    } finally {
      final worker = session.isolateSession;
      session.release();
      await worker?.release();
      for (final output in outputs) {
        output?.release();
      }
      nested?.release();
      flat.release();
      runOptions.release();
      options.release();
    }
  });
}
