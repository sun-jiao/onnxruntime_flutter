import 'dart:async';
import 'dart:collection';
import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:onnxruntime/src/bindings/onnxruntime_bindings_generated.dart'
    as bg;
import 'package:onnxruntime/src/util/native_memory.dart';

void main() {
  late Uint8List model;
  late OrtSessionOptions options;
  late OrtSession session;
  late OrtRunOptions runOptions;
  late OrtValueTensor input;
  late Map<String, OrtValue> inputs;

  setUp(() {
    model = File('example/assets/models/test_types_FLOAT.pb').readAsBytesSync();
    options = OrtSessionOptions()..setIntraOpNumThreads(1);
    session = OrtSession.fromBuffer(model, options);
    runOptions = OrtRunOptions();
    input = OrtValueTensor.createTensorWithDataList(
        Float32List.fromList([1, 2, 3, 4, 5]), [1, 5]);
    inputs = {session.inputNames.single: input};
  });

  tearDown(() {
    input.release();
    runOptions.release();
    session.release();
    options.release();
  });

  test('repeated successful runs release all temporary allocations', () {
    final allocator = _TrackedAllocator();
    for (var i = 0; i < 20; i++) {
      final outputs = allocator.run(() => session.run(runOptions, inputs));
      try {
        expect(allocator.live, isEmpty);
        expect(outputs.single!.value, [
          [1.0, 2.0, 3.0, 4.0, 5.0]
        ]);
      } finally {
        for (final output in outputs) {
          output?.release();
        }
      }
    }
    expect(allocator.allocations, greaterThan(0));
  });

  test('native inference errors preserve exceptions and free temporary names',
      () {
    final allocator = _TrackedAllocator();
    // The wrapper preserves native messages, which differ across pinned ORTs.
    final message = OrtEnv.version == '1.15.1'
        ? 'Invalid Feed Input Name'
        : 'Invalid input name: missing';
    for (var i = 0; i < 5; i++) {
      expect(
        () => allocator.run(() => session.run(runOptions, {'missing': input})),
        throwsA(isA<Exception>()
            .having((error) => error.toString(), 'message', contains(message))),
      );
      expect(allocator.live, isEmpty);
    }
    expect(allocator.allocations, greaterThan(0));
  });

  test(
      'Dart errors during output-name preparation also free earlier allocations',
      () {
    final allocator = _TrackedAllocator();
    final error = StateError('output name failed');
    expect(
      () => allocator
          .run(() => session.run(runOptions, inputs, _ThrowingNames(error))),
      throwsA(same(error)),
    );
    expect(allocator.allocations, greaterThan(0));
    expect(allocator.live, isEmpty);
  });

  test('invalid model loading releases the copied model buffer', () {
    final allocator = _TrackedAllocator();
    final invalidModel = Uint8List(1024 * 1024)..fillRange(0, 1024 * 1024, 255);
    expect(
        () => allocator.run(() => OrtSession.fromBuffer(invalidModel, options)),
        throwsException);
    expect(allocator.allocatedBytes, greaterThanOrEqualTo(invalidModel.length));
    expect(allocator.live, isEmpty);
  });

  test('a successful numeric tensor retains only its data until release', () {
    final allocator = _TrackedAllocator();
    final tensor = allocator.run(() => OrtValueTensor.createTensorWithDataList(
        Float32List.fromList([3, 4]), [1, 2]));
    try {
      expect(allocator.live.values.toList(), [2 * sizeOf<Float>()]);
      expect(tensor.value, [
        [3.0, 4.0]
      ]);
    } finally {
      // Release outside the allocation zone must use the original allocator.
      tensor.release();
    }
    expect(allocator.live, isEmpty);
  });

  test('shape mismatch releases numeric tensor data', () {
    final allocator = _TrackedAllocator();
    expect(
        () => allocator.run(() => OrtValueTensor.createTensorWithDataList(
            Float32List.fromList([1, 2]), [3])),
        throwsException);
    expect(allocator.allocations, greaterThan(0));
    expect(allocator.live, isEmpty);
  });

  test(
      'all supported numeric buffers preserve values and release their storage',
      () {
    for (final data in <List>[
      Uint8List.fromList([1, 2]),
      Int8List.fromList([-1, 2]),
      Uint16List.fromList([1, 2]),
      Int16List.fromList([-1, 2]),
      Uint32List.fromList([1, 2]),
      Int32List.fromList([-1, 2]),
      Uint64List.fromList([1, 2]),
      Int64List.fromList([-1, 2]),
      Float32List.fromList([1.5, 2.5]),
      Float64List.fromList([1.5, 2.5]),
      [true, false],
    ]) {
      final allocator = _TrackedAllocator();
      final tensor =
          allocator.run(() => OrtValueTensor.createTensorWithDataList(data));
      try {
        expect(tensor.value, data);
        expect(allocator.live.length, 1);
      } finally {
        tensor.release();
      }
      expect(allocator.live, isEmpty);
    }
  });

  test('sequence extraction unwinds partial child construction', () {
    final sequence = using((arena) {
      final children = arena<Pointer<bg.OrtValue>>(2);
      children[0] = input.ptr;
      children[1] = input.ptr;
      final output = arena<Pointer<bg.OrtValue>>();
      final status = OrtEnv.instance.ortApiPtr.ref.CreateValue.asFunction<
              bg.OrtStatusPtr Function(Pointer<Pointer<bg.OrtValue>>, int, int,
                  Pointer<Pointer<bg.OrtValue>>)>()(
          children, 2, ONNXType.sequence.value, output);
      OrtStatus.checkOrtStatus(status);
      return OrtValueSequence(output.value);
    });
    try {
      void extract() {
        final children = sequence.value!;
        try {
          expect(children.length, 2);
        } finally {
          for (final child in children) {
            child.release();
          }
        }
      }

      final baseline = _TrackedAllocator();
      baseline.run(extract);
      expect(baseline.live, isEmpty);
      for (var i = 1; i <= baseline.allocations; i++) {
        final allocator = _TrackedAllocator(failAt: i);
        expect(
            () => allocator.run(extract), throwsA(isA<_AllocationFailure>()));
        expect(allocator.live, isEmpty);
      }
    } finally {
      sequence.release();
    }
  });

  test('provider options free temporary names on success or native failure',
      () {
    final allocator = _TrackedAllocator();
    final available = OrtEnv.instance.availableProviders();
    if (available.contains(OrtProvider.xnnpack)) {
      expect(allocator.run(options.appendXnnpackProvider), isTrue);
    } else {
      expect(
          () => allocator.run(options.appendXnnpackProvider), throwsException);
    }
    expect(allocator.allocations, greaterThan(0));
    expect(allocator.live, isEmpty);
  });

  test('string tensors release temporary UTF-8 strings without losing values',
      () {
    final allocator = _TrackedAllocator();
    final tensor = allocator.run(
        () => OrtValueTensor.createTensorWithDataList(['hello', '中文🧠'], [2]));
    try {
      expect(allocator.allocations, greaterThan(0));
      expect(allocator.live, isEmpty);
      expect(tensor.value, ['hello', '中文🧠']);
    } finally {
      tensor.release();
    }
  });

  test('string fill failures clean up allocations and keep throwing', () {
    final allocator = _TrackedAllocator();
    expect(
        () => allocator.run(() => OrtValueTensor.createTensorWithDataList(
            ['first', 'out of bounds'], [1])),
        throwsException);
    expect(allocator.allocations, greaterThan(0));
    expect(allocator.live, isEmpty);
  });

  test('mixed nested string input throws a type error without leaking', () {
    final allocator = _TrackedAllocator();
    expect(
        () => allocator.run(() => OrtValueTensor.createTensorWithDataList([
              ['first', 1]
            ], [
              1,
              2
            ])),
        throwsA(isA<TypeError>()));
    expect(allocator.live, isEmpty);
  });

  test('run tags retain their value after temporary strings are freed', () {
    final allocator = _TrackedAllocator();
    allocator.run(() => runOptions.setRunTag('推理🧠'));
    expect(allocator.run(runOptions.getRunTag), '推理🧠');
    expect(allocator.allocations, greaterThan(0));
    expect(allocator.live, isEmpty);
  });

  // Fail at every allocation position, including positions after native objects
  // have already been created, to exercise unwinding rather than just success.
  for (final operation in ['model', 'run', 'numeric tensor', 'string tensor']) {
    test('$operation unwinds every injected allocation failure', () {
      void action() {
        switch (operation) {
          case 'model':
            OrtSession.fromBuffer(model, options).release();
            break;
          case 'run':
            for (final output in session.run(runOptions, inputs)) {
              output?.release();
            }
            break;
          case 'numeric tensor':
            OrtValueTensor.createTensorWithDataList(
                Float32List.fromList([1, 2]), [2]).release();
            break;
          case 'string tensor':
            OrtValueTensor.createTensorWithDataList(['a', 'b'], [2]).release();
            break;
        }
      }

      final baseline = _TrackedAllocator();
      baseline.run(action);
      expect(baseline.allocations, greaterThan(0));
      expect(baseline.live, isEmpty);
      for (var failure = 1; failure <= baseline.allocations; failure++) {
        final allocator = _TrackedAllocator(failAt: failure);
        expect(() => allocator.run(action), throwsA(isA<_AllocationFailure>()),
            reason: '$operation allocation $failure');
        expect(allocator.live, isEmpty,
            reason: '$operation allocation $failure leaked');
      }
    });
  }
}

class _AllocationFailure implements Exception {}

class _TrackedAllocator implements Allocator {
  _TrackedAllocator({this.failAt});

  final int? failAt;
  final live = <int, int>{};
  int allocations = 0;
  int allocatedBytes = 0;

  T run<T>(T Function() action) =>
      runZoned(action, zoneValues: {nativeAllocatorZoneKey: this});

  @override
  Pointer<T> allocate<T extends NativeType>(int byteCount, {int? alignment}) {
    if (++allocations == failAt) {
      throw _AllocationFailure();
    }
    final pointer = calloc.allocate<T>(byteCount, alignment: alignment);
    live[pointer.address] = byteCount;
    allocatedBytes += byteCount;
    return pointer;
  }

  @override
  void free(Pointer pointer) {
    if (live.remove(pointer.address) == null) {
      throw StateError('Freeing an unowned or already freed allocation');
    }
    calloc.free(pointer);
  }
}

class _ThrowingNames extends ListBase<String> {
  _ThrowingNames(this.error);
  final Object error;

  @override
  int get length => 1;
  @override
  String operator [](int index) => throw error;
  @override
  set length(int value) => throw UnsupportedError('Read only');
  @override
  void operator []=(int index, String value) =>
      throw UnsupportedError('Read only');
}
