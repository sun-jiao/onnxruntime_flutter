import 'dart:collection';
import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:onnxruntime/src/ort_isolate_session.dart';

void main() {
  late OrtSession session;
  late OrtSessionOptions options;
  late OrtRunOptions runOptions;
  late List<OrtValue> values;

  setUp(() {
    options = OrtSessionOptions()..setIntraOpNumThreads(1);
    session = OrtSession.fromBuffer(
      File('example/assets/models/test_types_FLOAT.pb').readAsBytesSync(),
      options,
    );
    runOptions = OrtRunOptions();
    values = [];
  });

  tearDown(() {
    // Deduplicate to avoid double-freeing when testing the original bug.
    final released = <int>{};
    for (final value in values) {
      if (released.add(value.address)) {
        value.release();
      }
    }
    session.release();
    runOptions.release();
    options.release();
  });

  Map<String, OrtValue> inputFor(int index) {
    final input = OrtValueTensor.createTensorWithDataList(
      Float32List.fromList(List.generate(5, (i) => index * 10.0 + i)),
      [1, 5],
    );
    values.add(input);
    return {session.inputNames.single: input};
  }

  void remember(List<OrtValue?> outputs) {
    values.addAll(outputs.whereType<OrtValue>());
  }

  Future<void> checkConcurrentRequests(int offset) async {
    final inputs = List.generate(8, (i) => inputFor(offset + i));
    final expected = inputs.map((input) {
      final outputs = session.run(runOptions, input);
      try {
        return outputs.map((value) => value!.value).toList();
      } finally {
        for (final output in outputs) {
          output?.release();
        }
      }
    }).toList();

    final results = await Future.wait(
      List.generate(inputs.length, (i) async {
        final outputs = await session.runAsync(
          runOptions,
          inputs[i],
          i.isEven ? null : session.outputNames,
        )!;
        remember(outputs);
        return outputs;
      }),
    );

    // Keep all results alive until comparison: each caller must own its handles.
    final addresses = results
        .expand((outputs) => outputs)
        .map((output) => output!.address)
        .toList();
    expect(addresses.toSet().length, addresses.length);
    for (var i = 0; i < results.length; i++) {
      expect(results[i].map((output) => output!.value).toList(), expected[i]);
    }
    expect(session.isolateSession!.state, IsolateSessionState.idle);
  }

  test('concurrent first calls share initialization and receive their results',
      () async {
    await checkConcurrentRequests(0);
  });

  test('concurrent calls on an initialized session own distinct results',
      () async {
    remember(await session.runAsync(runOptions, inputFor(-1))!);
    await checkConcurrentRequests(10);
    await checkConcurrentRequests(20);
  });

  test('sequential calls preserve values and optional output names', () async {
    for (var i = 0; i < 3; i++) {
      final input = inputFor(i);
      final expected = session.run(runOptions, input);
      remember(expected);
      final pending = session.runAsync(
        runOptions,
        input,
        i.isEven ? null : session.outputNames,
      )!;
      if (i > 0) {
        expect(session.isolateSession!.state, IsolateSessionState.loading);
      }
      final actual = await pending;
      remember(actual);
      expect(actual.map((output) => output!.value).toList(),
          expected.map((output) => output!.value).toList());
      expect(session.isolateSession!.state, IsolateSessionState.idle);
    }
  });

  for (final failure in ['input name', 'input shape', 'output name']) {
    test(
        'invalid $failure returns an empty result and the worker remains usable',
        () async {
      var input = inputFor(1);
      List<String>? outputNames;
      switch (failure) {
        case 'input name':
          input = {'missing_input': input.values.single};
          break;
        case 'input shape':
          final tensor = OrtValueTensor.createTensorWithDataList(
            Float32List(3),
            [1, 3],
          );
          values.add(tensor);
          input = {session.inputNames.single: tensor};
          break;
        case 'output name':
          outputNames = ['missing_output'];
          break;
      }
      final failed = await session
          .runAsync(runOptions, input, outputNames)!
          .timeout(const Duration(seconds: 5));
      remember(failed);
      expect(failed, isEmpty);
      expect(session.isolateSession!.state, IsolateSessionState.idle);

      final validInput = inputFor(2);
      final expected = session.run(runOptions, validInput);
      remember(expected);
      final actual = await session
          .runAsync(runOptions, validInput)!
          .timeout(const Duration(seconds: 5));
      remember(actual);
      expect(actual.map((output) => output!.value).toList(),
          expected.map((output) => output!.value).toList());
    });
  }

  test('a failed concurrent request does not consume successful results',
      () async {
    final inputs = List.generate(3, inputFor);
    final expected = [
      session.run(runOptions, inputs[0]),
      session.run(runOptions, inputs[2]),
    ];
    expected.forEach(remember);
    final results = await Future.wait([
      session.runAsync(runOptions, inputs[0])!,
      session.runAsync(runOptions, {'missing_input': inputs[1].values.single})!,
      session.runAsync(runOptions, inputs[2])!,
    ]).timeout(const Duration(seconds: 5));
    results.forEach(remember);
    expect(results[1], isEmpty);
    for (final i in [0, 2]) {
      expect(results[i].map((output) => output!.value).toList(),
          expected[i ~/ 2].map((output) => output!.value).toList());
    }
    expect(results[0].single!.address, isNot(results[2].single!.address));
    expect(session.isolateSession!.state, IsolateSessionState.idle);
  });

  test('worker exit completes pending and subsequent requests without hanging',
      () async {
    final input = inputFor(0);
    remember(await session.runAsync(runOptions, input)!);
    final results = await Future.wait([
      session.runAsync(runOptions, input, _ExitOnReadOutputNames())!,
      session.runAsync(runOptions, input)!,
    ]).timeout(const Duration(seconds: 5));
    results.forEach(remember);
    expect(results, [isEmpty, isEmpty]);
    final subsequent = await session
        .runAsync(runOptions, input)!
        .timeout(const Duration(seconds: 5));
    remember(subsequent);
    expect(subsequent, isEmpty);
    expect(session.isolateSession!.state, IsolateSessionState.idle);
  });
}

// The worker first reads outputNames inside run(), before entering native Run.
// This simulates abrupt worker termination without touching private state or
// killing a worker while it holds native resources used by the test.
class _ExitOnReadOutputNames extends ListBase<String> {
  @override
  int get length => Isolate.exit();

  @override
  set length(int value) => throw UnsupportedError('Read only');

  @override
  String operator [](int index) => throw UnsupportedError('Read only');

  @override
  void operator []=(int index, String value) =>
      throw UnsupportedError('Read only');
}
