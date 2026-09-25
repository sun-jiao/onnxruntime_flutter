import 'dart:async';
import 'dart:isolate';

import 'package:onnxruntime/src/ort_session.dart';
import 'package:onnxruntime/src/ort_value.dart';

class OrtIsolateSession {
  int address;
  final String debugName;
  late Isolate _newIsolate;
  late SendPort _newIsolateSendPort;
  late StreamSubscription _streamSubscription;
  final _outputController = StreamController<_IsolateSessionResult>.broadcast();

  IsolateSessionState get state => _state;
  var _state = IsolateSessionState.idle;
  var _initialized = false;
  Future<void>? _initialization;
  var _nextRequestId = 0;
  final _completer = Completer();

  OrtIsolateSession(
    OrtSession session, {
    this.debugName = 'OnnxRuntimeSessionIsolate',
  }) : address = session.address;

  Future<void> _init() async {
    final rootIsolateReceivePort = ReceivePort();
    final rootIsolateSendPort = rootIsolateReceivePort.sendPort;
    _newIsolate = await Isolate.spawn(
        createNewIsolateContext, rootIsolateSendPort,
        debugName: debugName);
    _streamSubscription = rootIsolateReceivePort.listen((message) {
      if (message is SendPort) {
        _newIsolateSendPort = message;
        _completer.complete();
      }
      if (message is _IsolateSessionResult) {
        _outputController.add(message);
      }
    });
    await _completer.future;
  }

  static Future<void> createNewIsolateContext(
      SendPort rootIsolateSendPort) async {
    final newIsolateReceivePort = ReceivePort();
    final newIsolateSendPort = newIsolateReceivePort.sendPort;
    rootIsolateSendPort.send(newIsolateSendPort);
    await for (final _IsolateSessionData data in newIsolateReceivePort) {
      final session = OrtSession.fromAddress(data.session);
      final runOptions = OrtRunOptions.fromAddress(data.runOptions);
      final inputs = data.inputs.map(
          (key, value) => MapEntry(key, OrtValueTensor.fromAddress(value)));
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
      rootIsolateSendPort.send(_IsolateSessionResult(data.requestId, outputs));
    }
  }

  Future<List<OrtValue?>> run(
      OrtRunOptions runOptions, Map<String, OrtValue> inputs,
      [List<String>? outputNames]) async {
    try {
      // Concurrent first calls must share the same worker and handshake.
      if (!_initialized) {
        await (_initialization ??= _init());
        _initialized = true;
      }
      final transformedInputs =
          inputs.map((key, value) => MapEntry(key, value.address));
      _state = IsolateSessionState.loading;
      final requestId = _nextRequestId++;
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
      final outputs = result.outputs.map((e) {
        final onnxType = ONNXType.valueOf(e.key);
        switch (onnxType) {
          case ONNXType.tensor:
            return OrtValueTensor.fromAddress(e.value);
          case ONNXType.sequence:
            return OrtValueSparseTensor.fromAddress(e.value);
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
    } catch (e) {
      if (!_initialized) {
        _initialization = null;
      }
      _state = IsolateSessionState.idle;
      return [];
    }
  }

  Future<void> release() async {
    await _streamSubscription.cancel();
    await _outputController.close();
    _newIsolate.kill();
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
  final Map<String, int> inputs;
  final List<String>? outputNames;
}

class _IsolateSessionResult {
  _IsolateSessionResult(this.requestId, this.outputs);

  final int requestId;
  final List<MapEntry> outputs;
}
