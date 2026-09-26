import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:onnxruntime/onnxruntime.dart';

Future<void> checkStrictAsync(OrtSession session, {required bool web}) async {
  final options = OrtRunOptions();
  final input = OrtValueTensor.createTensorWithDataList(
      Float32List.fromList([1, 2]), [1, 2]);
  final inputs = {'input': input};
  final errors = <OrtInferenceException>[];
  Future<void> failure(Future<List<OrtValue?>> future) async {
    try {
      final result = await future;
      for (final value in result) { value?.release(); }
      fail('Expected strict inference failure');
    } on OrtInferenceException catch (error) {
      errors.add(error);
      expect(error.message, isNotEmpty);
      expect(error.requestId, isNotNull);
      expect(error.remoteStackTrace, isNotNull);
      expect(error.backend, web ? 'web' : 'native');
      if (!web) expect(error.code, 2);
    }
  }
  Future<void> success(Future<List<OrtValue?>> future) async {
    final result = await future;
    try { expect(result.single!.value, [[1.0, 2.0]]); }
    finally { for (final value in result) { value?.release(); } }
  }
  try {
    // Mix cold-start errors, strict successes and legacy requests concurrently.
    await Future.wait([
      failure(session.runAsyncOrThrow(options, {'wrong': input})),
      success(session.runAsyncOrThrow(options, inputs)),
      failure(session.runAsyncOrThrow(options, inputs, ['wrong'])),
      success(session.runAsync(options, inputs)!),
      expectLater(session.runAsync(options, {'wrong': input}), completion(isEmpty)),
    ]);
    expect(errors.map((e) => e.requestId).toSet(), hasLength(2));
    await success(session.runAsyncOrThrow(options, inputs, ['output']));
    // Accepted work must survive session release, including an earlier failure.
    final bad = failure(session.runAsyncOrThrow(options, inputs, ['wrong']));
    final good = success(session.runAsyncOrThrow(options, inputs));
    session.release();
    await Future.wait([bad, good]);
    await expectLater(session.runAsyncOrThrow(options, inputs), throwsStateError);
    expect(await session.runAsync(options, inputs), isEmpty);
  } finally {
    session.release(); options.release(); input.release();
  }
}
