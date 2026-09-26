part of 'ort_web.dart';

class OrtSession {
  bool _released = false;
  Future<void>? _closeFuture;
  Uint8List? _model;
  late final ModelInfo _info;
  late final JSObject _options;
  late final OrtWebOptions _webOptions;
  OrtWebInitializationInfo? _webInitialization;
  OrtWebInitializationInfo? get webInitialization => _webInitialization;
  int? _threads;
  _Session? _session;
  int _nextRequestId = 0;
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
    _webOptions = options._webOptions;
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

  List<OrtValueInfo> get inputInfo {
    _check();
    return _info.inputInfo;
  }

  List<OrtValueInfo> get outputInfo {
    _check();
    return _info.outputInfo;
  }

  void validateInputs(Map<String, OrtValue> inputs) {
    _check();
    validateOrtInputs(
      inputInfo,
      inputs.map((name, value) {
        value._check();
        if (value is! OrtValueTensor) {
          throw UnsupportedError('Web validation supports tensor inputs only.');
        }
        return MapEntry(
          name,
          OrtValueInfo(
            name,
            ONNXType.tensor,
            elementType: value.elementType,
            shape: value.shape,
          ),
        );
      }),
    );
  }

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

  /// Strict counterpart of runAsync. A rejected request never poisons the queue.
  Future<List<OrtValue?>> runAsyncOrThrow(
    OrtRunOptions runOptions, Map<String, OrtValue> inputs,
    [List<String>? outputNames]) async {
    _check();
    final selected = List<String>.of(outputNames ?? _info.outputs);
    final feeds = Map<String, OrtValue>.of(inputs);
    final requestId = _nextRequestId++;
    final result = _tail.then((_) => _run(runOptions, feeds, selected,
        strict: true, requestId: requestId));
    _tail = result.then<void>((_) {}, onError: (Object error, StackTrace stack) {});
    return result;
  }

  /// Initializes the model without inference, sharing the serialized queue.
  Future<void> initialize() async {
    _check();
    final result = _tail.then((_) => _ensureSession());
    _tail = result.then<void>((_) {}, onError: (Object error, StackTrace stack) {});
    return result;
  }
  Future<void> get ready => initialize();

  Future<void> _ensureSession() async {
    if (_session != null) return;
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
    final result = await createWebBackend<_Session>(_webOptions,
        OrtEnv.instance.probeWebGpu, (backend) async {
      if (backend == OrtWebBackend.webgpu &&
          (wasm.getProperty<JSBoolean?>('proxy'.toJS)?.toDart ?? false)) {
        throw UnsupportedError('WebGPU cannot use the WASM proxy worker.');
      }
      _options.setProperty('executionProviders'.toJS, [backend.name].jsify());
      return _createSession(_model!.toJS, _options).toDart;
    });
    _session = result.session;
    _webInitialization = result.info;
    _model = null;
  }

  Future<List<OrtValue?>> _run(
    OrtRunOptions runOptions,
    Map<String, OrtValue> inputs,
    List<String> names, {
    bool strict = false,
    int? requestId,
  }) async {
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
      await _ensureSession();
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
    } catch (error, stack) {
      for (final output in outputs) {
        output?.release();
      }
      if (strict) {
        throw OrtInferenceException(error.toString(), requestId: requestId,
            backend: 'web', remoteStackTrace: stack.toString());
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

  /// Web writes profiling information through ORT's browser console, not a file.
  String? endProfiling() {
    _check();
    if (_session == null) throw StateError('The Web session has not initialized.');
    _session!.endProfiling();
    return null;
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

  /// Waits for accepted runs and runtime session disposal.
  Future<void> closeAsync() {
    release();
    return _closeFuture!;
  }

  void release() {
    if (_released) return;
    _released = true;
    _closeFuture = _tail.then((_) async {
          _model = null;
          final session = _session;
          _session = null;
          if (session != null) await session.release().toDart;
        });
    _closeFuture!.catchError((Object error) {
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
  OrtWebOptions _webOptions = const OrtWebOptions();
  void setWebOptions(OrtWebOptions options) { _check(); _webOptions = options; }
  bool _released = false;
  int? _threads;
  final Map<String, Object> _values = {};
  void _check() {
    if (_released) throw StateError('The session options have been released.');
  }

  void release() => _released = true;
  /// Enables ORT Web profiling. Prefix is validated but no browser file is made.
  void enableProfiling([String prefix = 'onnxruntime_profile']) {
    _check();
    if (prefix.isEmpty || prefix.contains('\u0000')) throw ArgumentError.value(prefix, 'prefix');
    _values['enableProfiling'] = true;
  }
  void disableProfiling() { _check(); _values['enableProfiling'] = false; }

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

  OrtProviderReport configureProviders(List<OrtProviderConfig> providers,
      {OrtProviderFallback fallback = OrtProviderFallback.error}) {
    _check();
    return registerOrtProviders(providers, fallback,
        OrtEnv.instance.availableProviderNames(), (config) => config.provider == OrtProvider.cpu);
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
