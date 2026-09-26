part of 'ort_web.dart';

class OrtSession {
  bool _released = false;
  Uint8List? _model;
  late final ModelInfo _info;
  late final JSObject _options;
  int? _threads;
  _Session? _session;
  Future<void> _tail = Future<void>.value();

  OrtSession.fromBuffer(Uint8List modelBuffer, OrtSessionOptions options) {
    options._check();
    _model = Uint8List.fromList(modelBuffer);
    _info = ModelInfo.read(_model!);
    final env = OrtEnv.instance;
    if (!env._initialized) env.init();
    _options =
        <String, Object>{
              ...options._values,
              'executionProviders': ['wasm'],
              'logSeverityLevel': env._level.value,
              'logId': env._logId,
            }.jsify()
            as JSObject;
    _threads = options._threads;
  }

  OrtSession.fromFile(File modelFile, OrtSessionOptions options) {
    _nativeOnly('OrtSession.fromFile; use fromBuffer with asset or HTTP bytes');
  }
  OrtSession.fromAddress(int address) {
    _nativeOnly('OrtSession.fromAddress');
  }

  void _check() {
    if (_released) throw StateError('The session has been released.');
  }

  int get address {
    _check();
    return _nativeOnly('OrtSession.address');
  }

  int get inputCount => _info.inputs.length;
  List<String> get inputNames => List.of(_info.inputs);
  int get outputCount => _info.outputs.length;
  List<String> get outputNames => List.of(_info.outputs);
  OrtIsolateSession? get isolateSession => null;

  List<OrtValue?> run(
    OrtRunOptions runOptions,
    Map<String, OrtValue> inputs, [
    List<String>? outputNames,
  ]) {
    _check();
    throw UnsupportedError(
      'ONNX Runtime Web only supports asynchronous '
      'inference. Use the existing runAsync() method.',
    );
  }

  Future<List<OrtValue?>>? runAsync(
    OrtRunOptions runOptions,
    Map<String, OrtValue> inputs, [
    List<String>? outputNames,
  ]) {
    if (_released) return Future.value(<OrtValue?>[]);
    // Snapshot collections now; callers retain tensor ownership until completion.
    final selected = List<String>.of(outputNames ?? _info.outputs);
    final feeds = Map<String, OrtValue>.of(inputs);
    final result = _tail.then((_) => _run(runOptions, feeds, selected));
    _tail = result.then<void>((_) {});
    return result;
  }

  Future<List<OrtValue?>> _run(
    OrtRunOptions runOptions,
    Map<String, OrtValue> inputs,
    List<String> names,
  ) async {
    JSObject? result;
    final outputs = <OrtValue?>[];
    try {
      runOptions._check();
      if (runOptions._terminated) throw StateError('Run has been terminated.');
      final feeds = JSObject();
      for (final entry in inputs.entries) {
        entry.value._check();
        if (entry.value is! OrtValueTensor) {
          throw UnsupportedError('Web inference supports tensor inputs only.');
        }
        feeds.setProperty(
          entry.key.toJS,
          (entry.value as OrtValueTensor)._tensor,
        );
      }
      if (_session == null) {
        final env = OrtEnv.instance;
        final wasm = _runtime
            .getProperty<JSObject>('env'.toJS)
            .getProperty<JSObject>('wasm'.toJS);
        final configured =
            wasm.getProperty<JSNumber?>('numThreads'.toJS)?.toDartInt;
        final count =
            _threads ??
            env._threads ??
            ((configured == null || configured == 0) ? 1 : configured);
        env._configureThreads(count);
        env._threads = count;
        _session = await _createSession(_model!.toJS, _options).toDart;
        _model = null;
      }
      if (runOptions._terminated) throw StateError('Run has been terminated.');
      result =
          await _session!
              .run(
                feeds,
                names.map((n) => n.toJS).toList().toJS,
                runOptions._snapshot(),
              )
              .toDart;
      for (final name in names) {
        final tensor = result.getProperty<_Tensor?>(name.toJS);
        if (tensor == null) throw StateError('Missing output: $name');
        // Copy each output, including repeated names, into independently owned
        // tensors before disposing the runtime's output buffers.
        outputs.add(OrtValueTensor._copy(tensor));
      }
      return outputs;
    } catch (error) {
      for (final output in outputs) {
        output?.release();
      }
      // Match the native isolate's existing asynchronous failure contract.
      debugPrint('ONNX Runtime Web inference failed: $error');
      return <OrtValue?>[];
    } finally {
      if (result != null) {
        for (final name in _objectKeys(result).toDart) {
          result.getProperty<_Tensor?>(name)?.dispose();
        }
      }
    }
  }

  String getMetadatas(String key) {
    _check();
    final value = _info.metadata[key.split('\u0000').first];
    if (value == null) {
      throw UnsupportedError(
        "Operation 'toDartString' not allowed on a 'nullptr'.",
      );
    }
    return value.split('\u0000').first;
  }

  void release() {
    if (_released) return;
    _released = true;
    _tail
        .then((_) async {
          _model = null;
          final session = _session;
          _session = null;
          if (session != null) await session.release().toDart;
        })
        .catchError((Object error) {
          debugPrint('ONNX Runtime Web release failed: $error');
        });
  }
}

/// No isolates are used by the browser backend; sessions use JavaScript futures.
class OrtIsolateSession {
  OrtIsolateSession(OrtSession session) {
    _nativeOnly('OrtIsolateSession');
  }
  Future<List<OrtValue?>> run(
    OrtRunOptions options,
    Map<String, OrtValue> inputs, [
    List<String>? outputNames,
  ]) async => _nativeOnly('OrtIsolateSession.run');
  Future<void> release() async {}
}

class OrtSessionOptions {
  bool _released = false;
  int? _threads;
  final Map<String, Object> _values = {};
  void _check() {
    if (_released) throw StateError('The session options have been released.');
  }

  void release() => _released = true;
  void setIntraOpNumThreads(int numThreads) {
    _check();
    if (numThreads < 0) throw ArgumentError.value(numThreads, 'numThreads');
    _threads = numThreads;
  }

  void setInterOpNumThreads(int numThreads) {
    _check();
    if (numThreads != 0 && numThreads != 1) {
      throw UnsupportedError('Web supports sequential graph execution only.');
    }
  }

  void setSessionGraphOptimizationLevel(GraphOptimizationLevel level) {
    _check();
    _values['graphOptimizationLevel'] =
        ['disabled', 'basic', 'extended', 'all'][level.index];
  }

  bool appendCPUProvider(CPUFlags flags) {
    _check();
    _values['enableCpuMemArena'] = flags == CPUFlags.useArena;
    return true;
  }

  bool appendCoreMLProvider(CoreMLFlags flags) {
    _check();
    return false;
  }

  bool appendNnapiProvider(NnapiFlags flags) {
    _check();
    return false;
  }

  bool appendQnnProvider() {
    _check();
    return false;
  }

  bool appendXnnpackProvider() {
    _check();
    return false;
  }
}

class OrtRunOptions {
  bool _released = false;
  bool _terminated = false;
  int _verbosity = 0;
  int _severity = 2;
  String _tag = '';
  OrtRunOptions();
  OrtRunOptions.fromAddress(int address) {
    _nativeOnly('OrtRunOptions.fromAddress');
  }
  void _check() {
    if (_released) throw StateError('The run options have been released.');
  }

  int get address {
    _check();
    return _nativeOnly('OrtRunOptions.address');
  }

  void release() => _released = true;
  void setRunLogVerbosityLevel(int level) {
    _check();
    _verbosity = level;
  }

  int getRunLogVerbosityLevel() {
    _check();
    return _verbosity;
  }

  void setRunLogSeverityLevel(int level) {
    _check();
    if (level < 0 || level > 4) throw ArgumentError.value(level, 'level');
    _severity = level;
  }

  int getRunLogSeverityLevel() {
    _check();
    return _severity;
  }

  void setRunTag(String tag) {
    _check();
    _tag = tag;
  }

  String getRunTag() {
    _check();
    return _tag;
  }

  void setTerminate() {
    _check();
    _terminated = true;
  }

  void unsetTerminate() {
    _check();
    _terminated = false;
  }

  JSObject _snapshot() =>
      <String, Object>{
            'logSeverityLevel': _severity,
            'logVerbosityLevel': _verbosity,
            'tag': _tag,
          }.jsify()
          as JSObject;
}
