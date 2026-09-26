import 'dart:io';
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'half_checks.dart';

void main() {
  test('all non-NaN half bit patterns roundtrip exactly', checkHalfRoundtrip);
  test('half precision exact bits, edge cases and explicit decoding', () {
    checkHalf(web: false);
  });
  for (final bfloat in [false, true]) {
    test('half precision real native inference ($bfloat)', () async {
      final options = OrtSessionOptions()..setIntraOpNumThreads(1);
      final session = OrtSession.fromFile(
        File('test/fixtures/${bfloat ? 'bfloat16' : 'float16'}_identity.onnx'),
        options,
      );
      options.release();
      await checkHalfInference(session, bfloat: bfloat);
    });
  }
  test(
    'composite copies survive source release, children survive parent release',
    () {
      final source = OrtValueTensor.createTensorWithDataList(
        Float32List.fromList([1, 2]),
      );
      final sequence = OrtValueSequence.fromTensors([source, source]);
      source.release();
      final children = sequence.elements;
      sequence.release();
      try {
        expect(children.map((e) => e.value), [
          [1.0, 2.0],
          [1.0, 2.0],
        ]);
      } finally {
        for (final child in children) {
          child.release();
        }
      }
      final half = OrtValueTensor.fromFloat16([1.5], [1]);
      final halfSequence = OrtValueSequence.fromTensors([half]);
      half.release();
      final halfChildren = halfSequence.elements;
      halfSequence.release();
      try {
        expect((halfChildren.single as OrtValueTensor).toFloat32List(), [1.5]);
      } finally {
        halfChildren.single.release();
      }
      for (final keys in <List>[
        ['中文', 'emoji🧠'],
        Int64List.fromList([10, 20]),
      ]) {
        final keyTensor = OrtValueTensor.createTensorWithDataList(keys);
        final valueTensor = OrtValueTensor.createTensorWithDataList(
          Float32List.fromList([1.5, 2.5]),
        );
        final map = OrtValueMap.fromTensors(keyTensor, valueTensor);
        keyTensor.release();
        valueTensor.release();
        try {
          expect(map.value, {keys[0]: 1.5, keys[1]: 2.5});
        } finally {
          map.release();
        }
      }
      expect(() => OrtValueSequence.fromTensors([]), throwsArgumentError);
    },
  );
  for (final csr in [false, true]) {
    test(
      'owned sparse ${csr ? 'CSR' : 'COO'} data and inference snapshots',
      () async {
        final data = Float32List.fromList([3, 4]);
        final indices = Int64List.fromList(csr ? [0, 1] : [0, 3]);
        final sparse =
            csr
                ? OrtValueSparseTensor.fromCsr(
                  data,
                  [2, 2],
                  indices,
                  Int64List.fromList([0, 1, 2]),
                )
                : OrtValueSparseTensor.fromCoo(data, [2, 2], indices);
        data[0] = 99;
        indices[0] = 1;
        final options = OrtSessionOptions()..setIntraOpNumThreads(1);
        final session = OrtSession.fromFile(
          File('test/fixtures/sparse_input.onnx'),
          options,
        );
        final run = OrtRunOptions();
        try {
          expect(sparse.value, isNull);
          final outputs = await session.runAsyncOrThrow(run, {'input': sparse});
          try {
            final snapshot =
                (outputs.single as OrtValueSparseTensor).toSparseData();
            expect(snapshot.values, [3, 4]);
            expect(snapshot.shape, [2, 2]);
            expect(
              snapshot.indices[csr ? 'inner' : 'coo'],
              csr ? [0, 1] : [0, 3],
            );
            expect(
              snapshot.format,
              csr ? OrtSparseFormat.csrc : OrtSparseFormat.coo,
            );
            sparse.release();
            expect(snapshot.values, [3, 4]);
            expect(() => sparse.toSparseData(), throwsStateError);
          } finally {
            for (final value in outputs) {
              value?.release();
            }
          }
        } finally {
          await session.closeAsync();
          options.release();
          run.release();
          sparse.release();
        }
      },
    );
  }
  test('block sparse extraction preserves values and index shape', () {
    final sparse = OrtValueSparseTensor.fromBlockSparse(
      Float32List.fromList([7]),
      [2, 2],
      [1, 1, 1],
      Int32List.fromList([0, 0]),
      [2, 1],
    );
    try {
      final snapshot = sparse.toSparseData();
      expect(snapshot.format, OrtSparseFormat.blockSparse);
      expect(snapshot.values, [7]);
      expect(snapshot.valuesShape, [1, 1, 1]);
      expect(snapshot.indices['block'], [0, 0]);
      expect(snapshot.blockIndicesShape, [2, 1]);
      expect(sparse.value, isNull);
    } finally {
      sparse.release();
    }
  });
  test('sparse empty tensors and invalid indices', () {
    final empty = OrtValueSparseTensor.fromCoo(Float32List(0), [
      2,
      2,
    ], Int64List(0));
    try {
      expect(empty.toSparseData().values.lengthInBytes, 0);
    } finally {
      empty.release();
    }
    expect(
      () => OrtValueSparseTensor.fromCoo(Float32List(1), [
        2,
        2,
      ], Int64List.fromList([4])),
      throwsArgumentError,
    );
    expect(
      () => OrtValueSparseTensor.fromCsr(
        Float32List(1),
        [2, 2],
        Int64List.fromList([0]),
        Int64List.fromList([0, 2, 1]),
      ),
      throwsArgumentError,
    );
  });
}
