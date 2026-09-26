@TestOn('browser')
library;

import '../model_info_checks.dart';
import '../strict_async_checks.dart';
import '../typed_data_checks.dart';
import '../lifecycle_checks.dart';
import 'dart:async';
import 'dart:convert';
import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:onnxruntime/onnxruntime.dart';

@JS('document')
external JSObject get _document;
@JS('ort')
external JSObject get _ort;
@JS('fetch')
external JSPromise<JSObject> _fetch(JSString url);

const _base = String.fromEnvironment(
  'ORT_WEB_URL',
  defaultValue: 'http://127.0.0.1:8765',
);

Future<Uint8List> _model(String path) async {
  final response = await _fetch('$_base/$path'.toJS).toDart;
  final buffer =
      await response
          .callMethod<JSPromise<JSArrayBuffer>>('arrayBuffer'.toJS)
          .toDart;
  return buffer.toDart.asUint8List();
}

void main() {
  setUpAll(() async {
    final loaded = Completer<void>();
    final script = _document.callMethod<JSObject>(
      'createElement'.toJS,
      'script'.toJS,
    );
    script.setProperty('src'.toJS, '$_base/ort.min.js'.toJS);
    script.setProperty('onload'.toJS, (() => loaded.complete()).toJS);
    script.setProperty(
      'onerror'.toJS,
      (() => loaded.completeError(
            StateError('Cannot load test runtime at $_base'),
          ))
          .toJS,
    );
    _document
        .getProperty<JSObject>('head'.toJS)
        .callMethod<JSAny?>('appendChild'.toJS, script);
    await loaded.future;
    final wasm = _ort
        .getProperty<JSObject>('env'.toJS)
        .getProperty<JSObject>('wasm'.toJS);
    wasm.setProperty('wasmPaths'.toJS, '$_base/'.toJS);
    wasm.setProperty('numThreads'.toJS, 1.toJS);
    OrtEnv.instance.init();
  });

  test('model descriptions and opt-in validation', () async {
    final options = OrtSessionOptions();
    final session = OrtSession.fromBuffer(await _model('dynamic_identity.onnx'), options);
    options.release();
    try { checkModelInfo(session); } finally { session.release(); }
  });

  test('strict asynchronous failures, concurrent requests and recovery', () async {
    final options = OrtSessionOptions();
    final session = OrtSession.fromBuffer(await _model('metadata.onnx'), options);
    options.release();
    await checkStrictAsync(session, web: true);
  });

  test('flat typed copies preserve dtype and ownership', () {
    checkTypedData(web: true);
  });

  test('real WASM output typed copy survives release', () async {
    final options = OrtSessionOptions();
    final session = OrtSession.fromBuffer(await _model('metadata.onnx'), options);
    options.release();
    await checkTypedInference(session);
  });

  test('awaitable close drains accepted browser work', () async {
    final options = OrtSessionOptions();
    final session = OrtSession.fromBuffer(await _model('metadata.onnx'), options);
    options.release();
    await checkClose(session);
  });

  test('runtime, providers, options and unsupported native operations', () {
    expect(OrtEnv.version, '1.23.2');
    expect(OrtEnv.instance.availableProviders(), [OrtProvider.cpu]);
    final options = OrtSessionOptions();
    expect(options.appendCPUProvider(CPUFlags.useArena), isTrue);
    expect(options.appendQnnProvider(), isFalse);
    expect(options.appendXnnpackProvider(), isFalse);
    expect(options.appendCoreMLProvider(CoreMLFlags.useNone), isFalse);
    expect(options.appendNnapiProvider(NnapiFlags.useNone), isFalse);
    options.setIntraOpNumThreads(1);
    options.setInterOpNumThreads(1);
    options.setSessionGraphOptimizationLevel(
      GraphOptimizationLevel.ortEnableAll,
    );
    expect(() => options.setInterOpNumThreads(2), throwsUnsupportedError);
    options.release();
    options.release();
    expect(() => options.appendCPUProvider(CPUFlags.useNone), throwsStateError);
    expect(() => OrtSession.fromAddress(1), throwsUnsupportedError);
    expect(() => OrtValueTensor.fromAddress(1), throwsUnsupportedError);
    final run = OrtRunOptions();
    run.setRunLogSeverityLevel(3);
    run.setRunLogVerbosityLevel(1);
    run.setRunTag('web');
    expect(run.getRunLogSeverityLevel(), 3);
    expect(run.getRunLogVerbosityLevel(), 1);
    expect(run.getRunTag(), 'web');
    run.release();
    expect(run.getRunTag, throwsStateError);
  });

  test('tensor types, scalars, Unicode, shapes, copies and release', () {
    for (final data in <List>[
      Uint8List.fromList([1, 255]),
      Int8List.fromList([-128, 127]),
      Uint16List.fromList([1, 65535]),
      Int16List.fromList([-32768, 32767]),
      Uint32List.fromList([1, 4294967295]),
      Int32List.fromList([-2147483648, 2147483647]),
      Float32List.fromList([1.5, 2.5]),
      Float64List.fromList([1.5, 2.5]),
      [1, -2],
      [true, false],
      ['中文🧠', ''],
    ]) {
      final tensor = OrtValueTensor.createTensorWithDataList(data, [1, 2]);
      expect(tensor.value, [data]);
      final returned = tensor.value as List;
      returned[0][0] = data.last;
      expect(tensor.value, [data]);
      tensor.release();
      tensor.release();
      expect(() => tensor.value, throwsStateError);
    }
    for (final data in <Object>[1, 1.5, true, '中文🧠']) {
      final tensor = OrtValueTensor.createTensorWithData(data);
      expect(tensor.value, data);
      tensor.release();
    }
    final empty = OrtValueTensor.createTensorWithDataList(Float32List(0), [
      2,
      0,
      3,
    ]);
    expect(empty.value, [[], []]);
    empty.release();
    final source = Float32List.fromList([1, 2]);
    final copy = OrtValueTensor.createTensorWithDataList(source);
    source[0] = 9;
    expect(copy.value, [1, 2]);
    copy.release();
    expect(
      () => OrtValueTensor.createTensorWithDataList([
        [1],
        [2, 3],
      ]),
      throwsArgumentError,
    );
    expect(
      () => OrtValueTensor.createTensorWithDataList([1, 2], [1]),
      throwsArgumentError,
    );
    expect(
      () => OrtValueTensor.createTensorWithDataList([1], [-1]),
      throwsArgumentError,
    );
    expect(
      () => OrtValueTensor.createTensorWithDataList(['a'], [2]),
      throwsArgumentError,
    );
    expect(
      () => OrtValueTensor.createTensorWithDataList([9007199254740992]),
      throwsUnsupportedError,
    );
  });

  test('synchronous model metadata and real WASM inference', () async {
    final bytes = await _model('metadata.onnx');
    final options = OrtSessionOptions();
    final session = OrtSession.fromBuffer(bytes, options);
    bytes.fillRange(0, bytes.length, 0); // The constructor owns a snapshot.
    options.release();
    expect(session.inputNames, ['input']);
    expect(session.outputNames, ['output']);
    expect(session.inputCount, 1);
    expect(session.outputCount, 1);
    expect(session.getMetadatas('author'), 'onnxruntime_flutter');
    expect(session.getMetadatas('说明🧠'), '中文元数据🧠');
    expect(session.getMetadatas('empty'), '');
    expect(session.getMetadatas('nul'), 'prefix');
    expect(() => session.getMetadatas('missing'), throwsUnsupportedError);
    final run = OrtRunOptions();
    final input = OrtValueTensor.createTensorWithDataList(
      Float32List.fromList([2, 3]),
      [1, 2],
    );
    expect(() => session.run(run, {'input': input}), throwsUnsupportedError);
    final pending = List.generate(
      3,
      (_) => session.runAsync(run, {'input': input})!,
    );
    session.release();
    session.release(); // Accepted runs must finish.
    for (final result in await Future.wait(pending)) {
      expect(result, hasLength(1));
      expect(result.single!.value, [
        [2, 3],
      ]);
      result.single!.release();
    }
    expect(await session.runAsync(run, {'input': input}), isEmpty);
    expect(session.inputNames, ['input']);
    expect(() => session.getMetadatas('author'), throwsStateError);
    input.release();
    run.release();
  });

  test('inference errors recover and termination can be reset', () async {
    final options = OrtSessionOptions();
    final session = OrtSession.fromBuffer(
      await _model('metadata.onnx'),
      options,
    );
    options.release();
    final run = OrtRunOptions();
    final input = OrtValueTensor.createTensorWithDataList(
      Float32List.fromList([4, 5]),
      [1, 2],
    );
    expect(await session.runAsync(run, {'wrong': input}), isEmpty);
    expect(await session.runAsync(run, {'input': input}, ['wrong']), isEmpty);
    run.setTerminate();
    expect(await session.runAsync(run, {'input': input}), isEmpty);
    run.unsetTerminate();
    final result = await session.runAsync(run, {'input': input});
    expect(result!.single!.value, [
      [4, 5],
    ]);
    result.single!.release();
    input.release();
    expect(await session.runAsync(run, {'input': input}), isEmpty);
    run.release();
    session.release();
  });

  test(
    'numeric, bool, int64 and string models execute with original API',
    () async {
      final cases = <String, List>{
        'FLOAT': Float32List.fromList([1, 2, 3, 4, 5]),
        'DOUBLE': Float64List.fromList([1, 2, 3, 4, 5]),
        'INT32': Int32List.fromList([1, 2, 3, 4, 5]),
        'INT64': [1, 2, 3, 4, 5],
        'BOOL': [true, false, true, false, true],
        'STRING': ['你好', '', '🧠', 'a', 'b'],
      };
      for (final entry in cases.entries) {
        final options = OrtSessionOptions();
        final session = OrtSession.fromBuffer(
          await _model('test_types_${entry.key}.pb'),
          options,
        );
        final input = OrtValueTensor.createTensorWithDataList(entry.value, [
          1,
          5,
        ]);
        final run = OrtRunOptions();
        final result = await session.runAsync(run, {
          session.inputNames.single: input,
        });
        expect(result, hasLength(1), reason: entry.key);
        // Existing type fixtures slice the complete input.
        expect(result!.single!.value, [entry.value], reason: entry.key);
        result.single!.release();
        input.release();
        run.release();
        session.release();
        options.release();
      }
    },
  );

  test(
    'invalid models fail synchronously without starting background futures',
    () {
      final options = OrtSessionOptions();
      for (final bytes in [
        Uint8List(0),
        Uint8List.fromList([58, 10, 1]),
        utf8.encode('invalid'),
      ]) {
        expect(
          () => OrtSession.fromBuffer(Uint8List.fromList(bytes), options),
          throwsFormatException,
        );
      }
      options.release();
    },
  );
}
