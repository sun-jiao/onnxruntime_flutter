import 'ort_inference_exception.dart';
import 'ort_status.dart' show inferenceErrorCode, inferenceErrorMessage;
import 'dart:async';
import 'dart:isolate';

import 'package:onnxruntime/src/ort_session.dart';
import 'package:onnxruntime/src/ort_value.dart';

class OrtIsolateSession {
  int address;
  final String debugName;
  Isolate? _newIsolate;
  late SendPort _newIsolateSendPort;
  StreamSubscription? _streamSubscription;
  final _outputController = StreamController<_IsolateSessionResult>.broadcast();

  IsolateSessionState get state => _state;
  var _state = IsolateSessionState.idle;
  var _initialized = false;
  var _workerStopped = false;
  Future<void>? _initialization;
  var _nextRequestId = 0;
  final _completer = Completer();
  final _workerExited = Completer<void>();
  var _released = false;
  var _activeRuns = 0;
  Completer<void>? _drained;
  Future<void>? _releaseFuture;

  OrtIsolateSession(
    OrtSession session, {
    this.debugName = 'OnnxRuntimeSessionIsolate',
  }) : address = session.address;

  Future<void> _init() async {
    final rootIsolateReceivePort = ReceivePort();
    final rootIsolateSendPort = rootIsolateReceivePort.sendPort;
    _newIsolate = await Isolate.spawn(
        createNewIsolateContext, rootIsolateSendPort,
        debugName: debugName,
        onError: rootIsolateSendPort,
        onExit: rootIsolateSendPort);
    _streamSubscription = rootIsolateReceivePort.listen((message) {
      if (message is SendPort) {
        _newIsolateSendPort = message;
        _completer.complete();
      }
      if (message is _IsolateSessionResult) {
        _outputController.add(message);
      }
      if (message == null) {
        _workerExited.complete();
      }
      if (message == null || message is List) {
        _handleWorkerStopped();
      }
    });
    await _completer.future;
  }

  void _handleWorkerStopped() {
    if (_workerStopped) {
      return;
    }
    _workerStopped = true;
    final error = StateError('The inference isolate has stopped.');
    if (!_completer.isCompleted) {
      _completer.completeError(error);
    }
    // Complete every pending request through run()'s existing error handling.
    _outputController.addError(error);
  }

  static Future<void> createNewIsolateContext(
      SendPort rootIsolateSendPort) async {
    final newIsolateReceivePort = ReceivePort();
    final newIsolateSendPort = newIsolateReceivePort.sendPort;
    rootIsolateSendPort.send(newIsolateSendPort);
    await for (final message in newIsolateReceivePort) {
      if (message == null) {
        newIsolateReceivePort.close();
        break;
      }
      final data = message as _IsolateSessionData;
      try {
        final session = OrtSession.fromAddress(data.session);
        final runOptions = OrtRunOptions.fromAddress(data.runOptions);
        final inputs =
            data.inputs.map((key, value) => MapEntry(key, value.restore()));
        final outputNames = data.outputNames;
        final outputs = session.run(runOptions, inputs, outputNames).map((e) {
          ONNXType onnxType;
          if (e is OrtValueTensor) {
            onnxType = ONNXType.tensor;
          } else if (e is OrtValueSequence) {
            onnxType = ONNXType.sequence;
          } else if (e is OrtValueMap) {
            onnxType = ONNXType.map;
          } else if (e is OrtValueSparseTensor) {
            onnxType = ONNXType.sparseTensor;
          } else {
            onnxType = ONNXType.tensor;
          }
          return MapEntry(onnxType.value, e?.address);
        }).toList();
        rootIsolateSendPort
            .send(_IsolateSessionResult(data.requestId, outputs));
      } catch (error, stack) {
        rootIsolateSendPort.send(_IsolateSessionResult(data.requestId, [],
          OrtInferenceException(inferenceErrorMessage(error),
            code: inferenceErrorCode(error), requestId: data.requestId,
            backend: 'native', remoteStackTrace: stack.toString())));
      }
    }
  }

  Future<List<OrtValue?>> run(
      OrtRunOptions runOptions, Map<String, OrtValue> inputs,
      [List<String>? outputNames]) =>
      _run(runOptions, inputs, outputNames, false);

  Future<List<OrtValue?>> runOrThrow(
      OrtRunOptions runOptions, Map<String, OrtValue> inputs,
      [List<String>? outputNames]) =>
      _run(runOptions, inputs, outputNames, true);

  Future<List<OrtValue?>> _run(OrtRunOptions runOptions,
      Map<String, OrtValue> inputs, List<String>? outputNames, bool strict) async {
    if (_released) {
      if (strict) throw StateError('The inference isolate has been released.');
      return [];
    }
    final requestId = _nextRequestId++;
    ++_activeRuns;
    try {
      // Concurrent first calls must share the same worker and handshake.
      if (!_initialized && !_workerStopped) {
        await (_initialization ??= _init());
        _initialized = true;
      }
      if (_workerStopped) {
        throw StateError('The inference isolate has stopped.');
      }
      final transformedInputs =
          inputs.map((key, value) => MapEntry(key, _IsolateInputValue(value)));
      _state = IsolateSessionState.loading;
      final data = _IsolateSessionData(
          requestId: requestId,
          session: address,
          runOptions: runOptions.address,
          inputs: transformedInputs,
          outputNames: outputNames);
      // Register before sending, and consume only this request's output handles.
      final response = _outputController.stream
          .firstWhere((result) => result.requestId == requestId);
      _newIsolateSendPort.send(data);
      final result = await response;
      if (result.error != null) throw result.error!;
      final outputs = result.outputs.map((e) {
        final onnxType = ONNXType.valueOf(e.key);
        switch (onnxType) {
          case ONNXType.tensor:
            return OrtValueTensor.fromAddress(e.value);
          case ONNXType.sequence:
            return OrtValueSequence.fromAddress(e.value);
          case ONNXType.map:
            return OrtValueMap.fromAddress(e.value);
          case ONNXType.sparseTensor:
            return OrtValueSparseTensor.fromAddress(e.value);
          default:
            return null;
        }
      }).toList();
      _state = IsolateSessionState.idle;
      return outputs;
    } catch (e, stack) {
      if (!_initialized) {
        _initialization = null;
      }
      _state = IsolateSessionState.idle;
      if (strict) {
        if (e is OrtInferenceException) rethrow;
        throw OrtInferenceException(inferenceErrorMessage(e),
            code: inferenceErrorCode(e), requestId: requestId,
            backend: 'native', remoteStackTrace: stack.toString());
      }
      return [];
    } finally {
      if (--_activeRuns == 0) {
        _drained?.complete();
      }
    }
  }

  Future<void> release() {
    _released = true;
    return _releaseFuture ??= _release();
  }

  Future<void> _release() async {
    // Accepted calls include those still waiting for the initial handshake.
    if (_activeRuns != 0) {
      _drained = Completer<void>();
      await _drained!.future;
    }
    if (_newIsolate != null) {
      if (!_workerStopped) {
        _newIsolateSendPort.send(null);
      }
      // A stop request (or an error notification) is not proof of exit.
      await _workerExited.future;
    }
    await _streamSubscription?.cancel();
    await _outputController.close();
  }
}

enum IsolateSessionState {
  idle,
  loading,
}

class _IsolateSessionData {
  _IsolateSessionData(
      {required this.requestId,
      required this.session,
      required this.runOptions,
      required this.inputs,
      this.outputNames});

  final int requestId;
  final int session;
  final int runOptions;
  final Map<String, _IsolateInputValue> inputs;
  final List<String>? outputNames;
}

class _IsolateSessionResult {
  _IsolateSessionResult(this.requestId, this.outputs, [this.error]);

  final OrtInferenceException? error;

  final int requestId;
  final List<MapEntry> outputs;
}

// Input handles remain owned by the caller; worker wrappers only borrow them.
class _IsolateInputValue {
  _IsolateInputValue(OrtValue value)
      : address = value.address,
        type = value is OrtValueSequence
            ? ONNXType.sequence
            : value is OrtValueMap
                ? ONNXType.map
                : value is OrtValueSparseTensor
                    ? ONNXType.sparseTensor
                    : ONNXType.tensor;

  final int address;
  final ONNXType type;

  OrtValue restore() {
    switch (type) {
      case ONNXType.sequence:
        return OrtValueSequence.fromAddress(address);
      case ONNXType.map:
        return OrtValueMap.fromAddress(address);
      case ONNXType.sparseTensor:
        return OrtValueSparseTensor.fromAddress(address);
      default:
        return OrtValueTensor.fromAddress(address);
    }
  }
}
