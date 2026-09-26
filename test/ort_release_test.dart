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
  late OrtValueTensor input;
  late Map<String, OrtValue> inputs;
  late List<OrtValue?> outputs;

  setUp(() {
    options = OrtSessionOptions()..setIntraOpNumThreads(1);
    session = OrtSession.fromBuffer(
      File('example/assets/models/test_types_FLOAT.pb').readAsBytesSync(),
      options,
    );
    runOptions = OrtRunOptions();
    input = OrtValueTensor.createTensorWithDataList(
      Float32List.fromList([1, 2, 3, 4, 5]),
      [1, 5],
    );
    inputs = {session.inputNames.single: input};
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

  test('an unused isolate session can be released repeatedly', () async {
    final worker = OrtIsolateSession(session);
    await Future.wait([worker.release(), worker.release()])
        .timeout(const Duration(seconds: 5));
    expect(await worker.run(runOptions, inputs), isEmpty);
  });

  test('release during initialization drains all accepted requests', () async {
    final expected = session.run(runOptions, inputs);
    outputs.addAll(expected);
    final pending =
        List.generate(3, (_) => session.runAsync(runOptions, inputs)!);
    final worker = session.isolateSession!;
    session.release();
    session.release();
    expect(session.isolateSession, isNull);
    expect(await session.runAsync(runOptions, inputs), isEmpty);
    final results =
        await Future.wait(pending).timeout(const Duration(seconds: 5));
    for (final result in results) {
      outputs.addAll(result);
      expect(result.map((value) => value!.value).toList(),
          expected.map((value) => value!.value).toList());
    }
    await worker.release().timeout(const Duration(seconds: 5));
  });

  test('release waits for a busy worker and preserves accepted results',
      () async {
    final expected = session.run(runOptions, inputs);
    outputs.addAll(expected);
    outputs.addAll(await session.runAsync(runOptions, inputs)!);
    final worker = session.isolateSession!;
    final directory = Directory.systemTemp.createTempSync('ort-release-');
    final gate = File('${directory.path}/continue');
    final entered = ReceivePort();
    final pending = session.runAsync(
        runOptions,
        inputs,
        _GatedOutputNames(
            session.outputNames.single, entered.sendPort, gate.path))!;
    try {
      await entered.first.timeout(const Duration(seconds: 5));
      // The worker is paused just before native Run and still needs the session.
      session.release();
      session.release();
      var stopped = false;
      final stopping = worker.release().then((_) => stopped = true);
      await Future<void>.delayed(Duration.zero);
      expect(stopped, isFalse);
      expect(await session.runAsync(runOptions, inputs), isEmpty);
      expect(() => session.run(runOptions, inputs), throwsStateError);
      gate.writeAsStringSync('continue');
      final result = await pending.timeout(const Duration(seconds: 5));
      outputs.addAll(result);
      await stopping.timeout(const Duration(seconds: 5));
      expect(result.map((value) => value!.value).toList(),
          expected.map((value) => value!.value).toList());
    } finally {
      // Always unblock the worker, including when an assertion fails.
      gate.writeAsStringSync('continue');
      await worker.release().timeout(const Duration(seconds: 5));
      entered.close();
      directory.deleteSync(recursive: true);
    }
  });

  test('release after a failed request is repeatable', () async {
    expect(
      await session.runAsync(runOptions, {'missing_input': input}),
      isEmpty,
    );
    final worker = session.isolateSession!;
    session.release();
    await Future.wait([worker.release(), worker.release()])
        .timeout(const Duration(seconds: 5));
    session.release();
  });

  test('released tensor data and handles are rejected', () {
    final tensors = [
      OrtValueTensor.createTensorWithData(1),
      OrtValueTensor.createTensorWithData(1.5),
      OrtValueTensor.createTensorWithData(true),
      OrtValueTensor.createTensorWithData('test'),
      OrtValueTensor.createTensorWithDataList([1, 2]),
      OrtValueTensor.createTensorWithDataList([1.5, 2.5]),
      OrtValueTensor.createTensorWithDataList([true, false]),
      OrtValueTensor.createTensorWithDataList(['first', 'second']),
      ...session.run(runOptions, inputs).cast<OrtValueTensor>(),
    ];
    try {
      for (final tensor in tensors) {
        expect(tensor.address, isNonZero);
        expect(tensor.ptr.address, tensor.address);
        expect(tensor.value, isNotNull);
        tensor.release();
        expect(() => tensor.value, throwsStateError);
        expect(() => tensor.ptr, throwsStateError);
        expect(() => tensor.address, throwsStateError);
        tensor.release();
      }
    } finally {
      for (final tensor in tensors) {
        tensor.release();
      }
    }
  });

  test('released inputs are rejected before synchronous inference', () {
    input.release();
    expect(() => session.run(runOptions, inputs), throwsStateError);
  });

  test('released run options are rejected by inference and all accessors', () {
    runOptions.setRunLogVerbosityLevel(2);
    runOptions.setRunLogSeverityLevel(3);
    runOptions.setRunTag('live');
    runOptions.setTerminate();
    runOptions.unsetTerminate();
    expect(runOptions.getRunLogVerbosityLevel(), 2);
    expect(runOptions.getRunLogSeverityLevel(), 3);
    expect(runOptions.getRunTag(), 'live');
    expect(runOptions.address, isNonZero);
    outputs.addAll(session.run(runOptions, inputs));

    runOptions.release();
    for (final operation in <Function>[
      () => runOptions.address,
      () => runOptions.setRunLogVerbosityLevel(1),
      runOptions.getRunLogVerbosityLevel,
      () => runOptions.setRunLogSeverityLevel(1),
      runOptions.getRunLogSeverityLevel,
      () => runOptions.setRunTag('released'),
      runOptions.getRunTag,
      runOptions.setTerminate,
      runOptions.unsetTerminate,
      () => session.run(runOptions, inputs),
    ]) {
      expect(operation, throwsStateError);
    }
  });

  for (final releaseInput in [true, false]) {
    test(
        'async released ${releaseInput ? 'input' : 'options'} still returns []',
        () async {
      if (releaseInput) {
        input.release();
      } else {
        runOptions.release();
      }
      expect(await session.runAsync(runOptions, inputs), isEmpty);
      final liveInput = OrtValueTensor.createTensorWithDataList(
          Float32List.fromList([1, 2, 3, 4, 5]), [1, 5]);
      final liveOptions = OrtRunOptions();
      try {
        final result = await session
            .runAsync(liveOptions, {session.inputNames.single: liveInput})!;
        outputs.addAll(result);
        expect(result, isNotEmpty);
      } finally {
        liveInput.release();
        liveOptions.release();
      }
    });
  }

  test('released session options are rejected by setters and constructors', () {
    options.release();
    final model = File('example/assets/models/test_types_FLOAT.pb');
    for (final operation in <Function>[
      () => options.setIntraOpNumThreads(1),
      () => options.setInterOpNumThreads(1),
      () => options.setSessionGraphOptimizationLevel(
          GraphOptimizationLevel.ortEnableAll),
      () => options.appendCPUProvider(CPUFlags.useNone),
      // These must fail before symbol lookup, even on unsupported platforms.
      () => options.appendCoreMLProvider(CoreMLFlags.useNone),
      () => options.appendNnapiProvider(NnapiFlags.useNone),
      options.appendQnnProvider,
      options.appendXnnpackProvider,
      () => OrtSession.fromFile(model, options),
      () => OrtSession.fromBuffer(model.readAsBytesSync(), options),
    ]) {
      expect(operation, throwsStateError);
    }
    // Session creation copied the options; releasing them does not end it.
    outputs.addAll(session.run(runOptions, inputs));
    expect(outputs, isNotEmpty);
  });

  test('released threading options are rejected by setters and env init', () {
    final threading = OrtThreadingOptions();
    threading.setGlobalIntraOpNumThreads(1);
    threading.setGlobalInterOpNumThreads(1);
    threading.setGlobalSpinControl(false);
    threading.setGlobalDenormalAsZero();
    threading.release();
    for (final operation in <Function>[
      () => threading.setGlobalIntraOpNumThreads(1),
      () => threading.setGlobalInterOpNumThreads(1),
      () => threading.setGlobalSpinControl(true),
      threading.setGlobalDenormalAsZero,
      () => threading.setGlobalIntraOpThreadAffinity('1'),
      () => OrtEnv.instance.init(options: threading),
    ]) {
      expect(operation, throwsStateError);
    }
    threading.release();
    outputs.addAll(session.run(runOptions, inputs));
    expect(outputs, isNotEmpty);
  });

  test('released session rejects handles but preserves cached metadata',
      () async {
    final names = session.inputNames;
    final count = session.inputCount;
    session.release();
    expect(() => session.address, throwsStateError);
    expect(() => session.getMetadatas('author'), throwsStateError);
    expect(() => session.run(runOptions, inputs), throwsStateError);
    expect(await session.runAsync(runOptions, inputs), isEmpty);
    expect(session.inputNames, names);
    expect(session.inputCount, count);
  });

  test('native wrappers tolerate repeated release', () {
    final threadingOptions = OrtThreadingOptions();
    threadingOptions.release();
    threadingOptions.release();
    final stringTensor = OrtValueTensor.createTensorWithData('test');
    stringTensor.release();
    stringTensor.release();
    input.release();
    input.release();
    runOptions.release();
    runOptions.release();
    options.release();
    options.release();
    session.release();
    session.release();
    // tearDown repeats these releases once more.
  });
}

// A file gate lets the main isolate request release while the worker is inside
// run(), without relying on a timing window or accessing private library state.
class _GatedOutputNames extends ListBase<String> {
  _GatedOutputNames(this.name, this.entered, this.gatePath);

  final String name;
  final SendPort entered;
  final String gatePath;
  bool _waited = false;

  @override
  int get length {
    if (!_waited) {
      _waited = true;
      entered.send(null);
      final deadline = DateTime.now().add(const Duration(seconds: 10));
      while (!File(gatePath).existsSync()) {
        if (DateTime.now().isAfter(deadline)) {
          throw StateError('Release test did not open the gate.');
        }
        sleep(const Duration(milliseconds: 1));
      }
    }
    return 1;
  }

  @override
  String operator [](int index) => name;

  @override
  set length(int value) => throw UnsupportedError('Read only');

  @override
  void operator []=(int index, String value) =>
      throw UnsupportedError('Read only');
}
