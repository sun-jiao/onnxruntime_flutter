import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:onnxruntime/src/bindings/onnxruntime_bindings_generated.dart'
    as bg;

void main() {
  for (final strings in [false, true]) {
    test('map address restoration initializes keys, values and size ($strings)',
        () {
      final keys = OrtValueTensor.createTensorWithDataList(
          strings ? ['中文', 'emoji🧠'] : Int64List.fromList([10, 20]));
      final values =
          OrtValueTensor.createTensorWithDataList(Float32List.fromList([1, 2]));
      final pointer = _createValue([keys, values], ONNXType.map);
      try {
        final direct = OrtValueMap(pointer);
        final restored = OrtValueMap.fromAddress(pointer.address);
        expect(restored.size, 2);
        expect(restored.value, direct.value);
        expect(restored.value,
            strings ? {'中文': 1.0, 'emoji🧠': 2.0} : {10: 1.0, 20: 2.0});
      } finally {
        _release(pointer); // Both wrappers refer to this one native handle.
        keys.release();
        values.release();
      }
    });
  }

  test('sequence address restoration preserves child tensor values', () {
    final input =
        OrtValueTensor.createTensorWithDataList(Float32List.fromList([1, 2]));
    final pointer = _createValue([input, input], ONNXType.sequence);
    try {
      final direct = OrtValueSequence(pointer);
      final restored = OrtValueSequence.fromAddress(pointer.address);
      expect(_read(restored), _read(direct));
      expect(_read(restored), [
        [1.0, 2.0],
        [1.0, 2.0]
      ]);
    } finally {
      _release(pointer);
      input.release();
    }
  });

  for (final filled in [false, true]) {
    test(
        'sparse address restoration preserves existing value semantics ($filled)',
        () {
      final pointer = _createSparse(filled);
      try {
        final direct = OrtValueSparseTensor(pointer);
        final restored = OrtValueSparseTensor.fromAddress(pointer.address);
        if (filled) {
          // COO values are currently unimplemented; do not add a new result type.
          expect(direct.value, isNull);
          expect(restored.value, direct.value);
        } else {
          final undefinedFormat = throwsA(isA<Exception>().having(
              (error) => error.toString(),
              'message',
              contains('Undefined sparsity type')));
          expect(() => direct.value, undefinedFormat);
          expect(() => restored.value, undefinedFormat);
        }
      } finally {
        _release(pointer);
      }
    });
  }

  test('released complex values reject data and handle access', () {
    final keys = OrtValueTensor.createTensorWithDataList([10, 20]);
    final values =
        OrtValueTensor.createTensorWithDataList(Float32List.fromList([1, 2]));
    final wrappers = <OrtValue>[];
    try {
      wrappers.add(OrtValueMap(_createValue([keys, values], ONNXType.map)));
      wrappers.add(OrtValueSequence(_createValue([values], ONNXType.sequence)));
      wrappers.add(OrtValueSparseTensor(_createSparse(true)));
      for (final wrapper in wrappers) {
        expect(wrapper.address, wrapper.ptr.address);
        // Reading children before release must remain valid.
        _read(wrapper);
        wrapper.release();
        expect(() => wrapper.value, throwsStateError);
        expect(() => wrapper.ptr, throwsStateError);
        expect(() => wrapper.address, throwsStateError);
        wrapper.release();
      }
      // The child tensors retain their independent ownership.
      expect(keys.value, [10, 20]);
      expect(values.value, [1.0, 2.0]);
    } finally {
      for (final wrapper in wrappers) {
        wrapper.release();
      }
      keys.release();
      values.release();
    }
  });

  test('sparse input handles survive synchronous and asynchronous inference',
      () async {
    final options = OrtSessionOptions()..setIntraOpNumThreads(1);
    final session =
        OrtSession.fromFile(File('test/fixtures/sparse_input.onnx'), options);
    final runOptions = OrtRunOptions();
    final input = OrtValueSparseTensor(_createSparse(true));
    final outputs = <OrtValue?>[];
    try {
      outputs.addAll(session.run(runOptions, {'input': input}));
      outputs.addAll(await session.runAsync(runOptions, {'input': input})!);
      expect(outputs, hasLength(2));
      for (final output in outputs) {
        expect(output, isA<OrtValueSparseTensor>());
        // Sparse value extraction remains unimplemented and returns null.
        expect(output!.value, isNull);
        expect(output.address, isNot(input.address));
      }
      expect(input.value, isNull);
    } finally {
      final worker = session.isolateSession;
      session.release();
      await worker?.release();
      for (final output in outputs) {
        output?.release();
      }
      input.release();
      runOptions.release();
      options.release();
    }
  });

  for (final sequence in [true, false]) {
    test('${sequence ? 'sequence' : 'map'} inputs match sync inference',
        () async {
      final options = OrtSessionOptions()..setIntraOpNumThreads(1);
      final fixture = sequence ? 'sequence_input' : 'map_input';
      final session =
          OrtSession.fromFile(File('test/fixtures/$fixture.onnx'), options);
      final runOptions = OrtRunOptions();
      final ownedInputs = <OrtValue>[];
      final ownedOutputs = <OrtValue?>[];
      try {
        // Concurrent first calls and a second batch after worker initialization.
        for (var batch = 0; batch < 2; batch++) {
          final pending = <Future<List<OrtValue?>>>[];
          final expected = <List<Object?>>[];
          for (var i = 0; i < 3; i++) {
            final number = (batch * 10 + i + 1).toDouble();
            final tensor = OrtValueTensor.createTensorWithDataList(
                Float32List.fromList([number, number + 1]), [1, 2]);
            ownedInputs.add(tensor);
            final OrtValue complex;
            final Object complexValue;
            if (sequence) {
              complex = OrtValueSequence(
                  _createValue([tensor, tensor], ONNXType.sequence));
              complexValue = [
                [
                  [number, number + 1]
                ],
                [
                  [number, number + 1]
                ],
              ];
            } else {
              final keys = OrtValueTensor.createTensorWithDataList([10, 20]);
              final values = OrtValueTensor.createTensorWithDataList(
                  Float32List.fromList([number, number + 1]));
              ownedInputs.addAll([keys, values]);
              complex = OrtValueMap(_createValue([keys, values], ONNXType.map));
              complexValue = {10: number, 20: number + 1};
            }
            ownedInputs.add(complex);
            final inputs = {'input': complex, 'tensor_input': tensor};
            final names =
                i.isEven ? null : session.outputNames.reversed.toList();
            final ordered = <Object?>[
              complexValue,
              [
                [number, number + 1]
              ]
            ];
            final values = names == null ? ordered : ordered.reversed.toList();
            final sync = session.run(runOptions, inputs, names);
            ownedOutputs.addAll(sync);
            expect(sync.map((value) => _read(value!)).toList(), values);
            expected.add(values);
            pending.add(session.runAsync(runOptions, inputs, names)!);
          }
          // Failures must remain local to their request and preserve [].
          final invalid = session.runAsync(runOptions, {})!;
          final results =
              await Future.wait(pending).timeout(const Duration(seconds: 5));
          for (final result in results) {
            ownedOutputs.addAll(result);
          }
          expect(await invalid.timeout(const Duration(seconds: 5)), isEmpty);
          for (var i = 0; i < results.length; i++) {
            expect(
                results[i].map((value) => _read(value!)).toList(), expected[i]);
          }
          final addresses = results
              .expand((result) => result)
              .map((value) => value!.address)
              .toList();
          expect(addresses.toSet().length, addresses.length);
        }
        // Worker restoration must not release or alter caller-owned inputs.
        for (final input in ownedInputs) {
          expect(_read(input), isNotNull);
        }
      } finally {
        final worker = session.isolateSession;
        session.release();
        await worker?.release();
        for (final output in ownedOutputs) {
          output?.release();
        }
        for (final input in ownedInputs.reversed) {
          input.release();
        }
        runOptions.release();
        options.release();
      }
    });
  }

  for (final fixture in ['tensor_sequence', 'map_sequence']) {
    test('$fixture has equivalent synchronous and asynchronous outputs',
        () async {
      final options = OrtSessionOptions()..setIntraOpNumThreads(1);
      final session = OrtSession.fromBuffer(
          File('test/fixtures/$fixture.onnx').readAsBytesSync(), options);
      final runOptions = OrtRunOptions();
      final inputs = <OrtValueTensor>[];
      final outputs = <OrtValue?>[];
      try {
        // Both cold initialization and concurrent calls after initialization.
        for (final count in [1, 3]) {
          final pending = <Future<List<OrtValue?>>>[];
          final expected = <List<Object?>>[];
          final types = <List<Type>>[];
          for (var i = 0; i < count; i++) {
            final input = OrtValueTensor.createTensorWithDataList(
                Float32List.fromList([i + 1.0, i + 2.0]), [1, 2]);
            inputs.add(input);
            final names = i.isEven
                ? session.outputNames
                : session.outputNames.reversed.toList();
            final sync = session.run(runOptions, {'input': input}, names);
            outputs.addAll(sync);
            expected.add(sync.map((value) => _read(value!)).toList());
            types.add(sync.map((value) => value.runtimeType).toList());
            pending.add(session.runAsync(runOptions, {'input': input}, names)!);
          }
          final results =
              await Future.wait(pending).timeout(const Duration(seconds: 5));
          for (var i = 0; i < results.length; i++) {
            outputs.addAll(results[i]);
            expect(results[i].map((value) => value.runtimeType).toList(),
                types[i]);
            expect(
                results[i].map((value) => _read(value!)).toList(), expected[i]);
          }
          final addresses = results
              .expand((result) => result)
              .map((value) => value!.address)
              .toList();
          expect(addresses.toSet().length, addresses.length);
          if (fixture == 'map_sequence') {
            expect(_read(results.first.single!), [
              {10: 1.0, 20: 2.0}
            ]);
          }
        }
      } finally {
        final worker = session.isolateSession;
        session.release();
        await worker?.release();
        for (final output in outputs) {
          output?.release();
        }
        for (final input in inputs) {
          input.release();
        }
        runOptions.release();
        options.release();
      }
    });
  }
}

Object? _read(OrtValue value) {
  if (value is OrtValueSequence) {
    final children = value.value!;
    try {
      return children.map(_read).toList();
    } finally {
      for (final child in children) {
        child.release();
      }
    }
  }
  return value.value;
}

Pointer<bg.OrtValue> _createValue(List<OrtValue> children, ONNXType type) =>
    using((arena) {
      final pointers = arena<Pointer<bg.OrtValue>>(children.length);
      for (var i = 0; i < children.length; i++) {
        pointers[i] = children[i].ptr;
      }
      final output = arena<Pointer<bg.OrtValue>>();
      OrtStatus.checkOrtStatus(OrtEnv.instance.ortApiPtr.ref.CreateValue
              .asFunction<
                  bg.OrtStatusPtr Function(Pointer<Pointer<bg.OrtValue>>, int,
                      int, Pointer<Pointer<bg.OrtValue>>)>()(
          pointers, children.length, type.value, output));
      return output.value;
    });

Pointer<bg.OrtValue> _createSparse(bool fill) => using((arena) {
      final shape = arena<Int64>(2);
      shape[0] = 2;
      shape[1] = 2;
      final output = arena<Pointer<bg.OrtValue>>();
      final api = OrtEnv.instance.ortApiPtr.ref;
      OrtStatus.checkOrtStatus(api.CreateSparseTensorAsOrtValue.asFunction<
              bg.OrtStatusPtr Function(Pointer<bg.OrtAllocator>, Pointer<Int64>,
                  int, int, Pointer<Pointer<bg.OrtValue>>)>()(
          OrtAllocator.instance.ptr,
          shape,
          2,
          ONNXTensorElementDataType.float.value,
          output));
      try {
        if (fill) {
          final info = arena<Pointer<bg.OrtMemoryInfo>>();
          OrtStatus.checkOrtStatus(api.AllocatorGetInfo.asFunction<
                  bg.OrtStatusPtr Function(Pointer<bg.OrtAllocator>,
                      Pointer<Pointer<bg.OrtMemoryInfo>>)>()(
              OrtAllocator.instance.ptr, info));
          final valuesShape = arena<Int64>()..value = 1;
          final values = arena<Float>()..value = 3;
          final indices = arena<Int64>()..value = 0;
          OrtStatus.checkOrtStatus(api.FillSparseTensorCoo.asFunction<
                  bg.OrtStatusPtr Function(
                      Pointer<bg.OrtValue>,
                      Pointer<bg.OrtMemoryInfo>,
                      Pointer<Int64>,
                      int,
                      Pointer<Void>,
                      Pointer<Int64>,
                      int)>()(output.value, info.value, valuesShape, 1,
              values.cast(), indices, 1));
        }
        return output.value;
      } catch (_) {
        _release(output.value);
        rethrow;
      }
    });

void _release(Pointer<bg.OrtValue> pointer) =>
    OrtEnv.instance.ortApiPtr.ref.ReleaseValue
        .asFunction<void Function(Pointer<bg.OrtValue>)>()(pointer);
