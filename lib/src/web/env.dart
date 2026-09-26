part of 'ort_web.dart';

class OrtEnv {
  OrtEnv._();
  static final OrtEnv instance = OrtEnv._();
  OrtLoggingLevel _level = OrtLoggingLevel.warning;
  String _logId = 'DartOnnxRuntime';
  bool _initialized = false;
  int? _threads;

  static void setApiVersion(OrtApiVersion apiVersion) {
    // C API selection is irrelevant to the JavaScript runtime.
  }

  void init({
    OrtLoggingLevel level = OrtLoggingLevel.warning,
    String logId = 'DartOnnxRuntime',
    OrtThreadingOptions? options,
  }) {
    options?._check();
    final env = _runtime.getProperty<JSObject>('env'.toJS);
    env.setProperty('logLevel'.toJS, level.name.toJS);
    _level = level;
    _logId = logId;
    if (options?._threads != null) _configureThreads(options!._threads!);
    _initialized = true;
  }

  void _configureThreads(int count) {
    if (_threads != null && _threads != count) {
      throw StateError(
        'WebAssembly thread count is global and cannot change '
        'after the first session. Configure ort.env.wasm.numThreads in HTML.',
      );
    }
    final wasm = _runtime
        .getProperty<JSObject>('env'.toJS)
        .getProperty<JSObject>('wasm'.toJS);
    wasm.setProperty('numThreads'.toJS, count.toJS);
  }

  void release() {
    // The JavaScript runtime owns its global WASM instance, shared by sessions.
    _initialized = false;
  }

  static String get version =>
      _runtime
          .getProperty<JSObject>('env'.toJS)
          .getProperty<JSObject>('versions'.toJS)
          .getProperty<JSString>('web'.toJS)
          .toDart;
  Object get ptr => _nativeOnly('OrtEnv.ptr');
  Object get ortApiPtr => _nativeOnly('OrtEnv.ortApiPtr');
  List<OrtProvider> availableProviders() => [OrtProvider.cpu];
}

class OrtThreadingOptions {
  bool _released = false;
  int? _threads;
  void _check() {
    if (_released) {
      throw StateError('The threading options have been released.');
    }
  }

  void release() => _released = true;
  void setGlobalIntraOpNumThreads(int numThreads) {
    _check();
    if (numThreads < 0) throw ArgumentError.value(numThreads, 'numThreads');
    _threads = numThreads;
  }

  void setGlobalInterOpNumThreads(int numThreads) {
    _check();
    if (numThreads != 0 && numThreads != 1) {
      throw UnsupportedError('Web supports sequential graph execution only.');
    }
  }

  void setGlobalSpinControl(bool allowSpinning) {
    _check();
    _nativeOnly('setGlobalSpinControl');
  }

  void setGlobalDenormalAsZero() {
    _check();
    _nativeOnly('setGlobalDenormalAsZero');
  }

  void setGlobalIntraOpThreadAffinity(String affinity) {
    _check();
    _nativeOnly('setGlobalIntraOpThreadAffinity');
  }
}

class OrtAllocator {
  OrtAllocator._();
  static final OrtAllocator instance = OrtAllocator._();
  Object get ptr => _nativeOnly('OrtAllocator.ptr');
}
