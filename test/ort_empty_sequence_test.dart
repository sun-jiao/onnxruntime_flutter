import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:onnxruntime/onnxruntime.dart';

void main() {
  late OrtSessionOptions options;
  late OrtSession session;
  late OrtRunOptions runOptions;
  late OrtValueTensor input;
  late List<OrtValue?> outputs;

  setUp(() {
    options = OrtSessionOptions()..setIntraOpNumThreads(1);
    session =
        OrtSession.fromFile(File('test/fixtures/empty_sequence.onnx'), options);
    runOptions = OrtRunOptions();
    input = OrtValueTensor.createTensorWithDataList(
        Float32List.fromList([1, 2]), [1, 2]);
    outputs = [];
  });

  tearDown(() async {
    final worker = session.isolateSession;
    session.release();
    await worker?.release();
    for (final output in outputs) {
      output?.release();
    }
    input.release();
    runOptions.release();
    options.release();
  });

  test('empty sequence reads and address restoration return fresh empty lists',
      () {
    outputs.addAll(session.run(runOptions, {'input': input}));
    final sequence = outputs.first! as OrtValueSequence;
    final borrowed = OrtValueSequence.fromAddress(sequence.address);
    expect(sequence.value, isA<List<OrtValue>>());
    expect(sequence.value, isEmpty);
    expect(borrowed.value, isEmpty);
    final first = sequence.value!;
    first.add(input);
    expect(sequence.value, isEmpty);
    expect(borrowed.value, isEmpty);
    expect(outputs[1]!.value, [
      [1.0, 2.0]
    ]);
    sequence.release();
    expect(() => sequence.value, throwsStateError);
    sequence.release();
    // borrowed shares the native handle, so only sequence releases it.
  });

  test('concurrent async empty sequences preserve output counts and ordering',
      () async {
    for (var batch = 0; batch < 2; batch++) {
      final pending = List.generate(
          3,
          (index) => session.runAsync(runOptions, {'input': input},
              index.isEven ? null : ['tensor', 'sequence'])!);
      final results =
          await Future.wait(pending).timeout(const Duration(seconds: 5));
      for (final result in results) {
        outputs.addAll(result);
      }
      for (var i = 0; i < results.length; i++) {
        final result = results[i];
        expect(result, hasLength(2));
        final sequenceIndex = i.isEven ? 0 : 1;
        expect(result[sequenceIndex], isA<OrtValueSequence>());
        expect(result[sequenceIndex]!.value, isEmpty);
        expect(result[1 - sequenceIndex]!.value, [
          [1.0, 2.0]
        ]);
      }
      final addresses = results
          .map((result) => result.whereType<OrtValueSequence>().single.address)
          .toList();
      expect(addresses.toSet().length, 3);
    }
  });

  test('empty sequence can be reused as sync and async input', () async {
    outputs.addAll(session.run(runOptions, {'input': input}));
    final empty = outputs.first! as OrtValueSequence;
    final identity =
        OrtSession.fromFile(File('test/fixtures/sequence_input.onnx'), options);
    try {
      final inputs = {'input': empty, 'tensor_input': input};
      final sync = identity.run(runOptions, inputs);
      outputs.addAll(sync);
      final async = await identity
          .runAsync(runOptions, inputs)!
          .timeout(const Duration(seconds: 5));
      outputs.addAll(async);
      for (final result in [sync, async]) {
        expect(result, hasLength(2));
        expect(result.first, isA<OrtValueSequence>());
        expect(result.first!.value, isEmpty);
        expect(result[1]!.value, [
          [1.0, 2.0]
        ]);
      }
      expect(empty.value, isEmpty);
    } finally {
      final worker = identity.isolateSession;
      identity.release();
      await worker?.release();
    }
  });
}
