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
  final _outputController = StreamController<List<MapEntry>>.broadcast();

  IsolateSessionState get state => _state;
  var _state = IsolateSessionState.idle;
  var _initialized = false;
  final _completer = Completer();

  OrtIsolateSession(
    OrtSession session, {
    this.debugName = 'OnnxRuntimeSessionIsolate',
  }) : address = session.address;

  Future<void> _init() async {
    // Create a receive port for communication with the new isolate
    final rootIsolateReceivePort = ReceivePort();
    final rootIsolateSendPort = rootIsolateReceivePort.sendPort;

    // Spawn the new isolate for running inference
    _newIsolate = await Isolate.spawn(
        createNewIsolateContext, rootIsolateSendPort,
        debugName: debugName);

    // Listen for messages from the new isolate
    _streamSubscription = rootIsolateReceivePort.listen((message) {
      // Handle initial SendPort message for bidirectional communication
      if (message is SendPort) {
        _newIsolateSendPort = message;
        // Only complete the completer if it hasn't been completed yet
        // This prevents the "Future already completed" error
        if (!_completer.isCompleted) {
          _completer.complete();
        }
      }
      // Handle inference output results
      if (message is List<MapEntry>) {
        _outputController.add(message);
      }
    });
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
      rootIsolateSendPort.send(outputs);
    }
  }

  Future<List<OrtValue?>> run(
      OrtRunOptions runOptions, Map<String, OrtValue> inputs,
      [List<String>? outputNames]) async {
    if (!_initialized) {
      await _init();
      await _completer.future;
      _initialized = true;
    }
    final transformedInputs =
        inputs.map((key, value) => MapEntry(key, value.address));
    _state = IsolateSessionState.loading;
    final data = _IsolateSessionData(
        session: address,
        runOptions: runOptions.address,
        inputs: transformedInputs,
        outputNames: outputNames);
    _newIsolateSendPort.send(data);
    late List<OrtValue?> outputs;
    await for (final result in _outputController.stream) {
      outputs = result.map((e) {
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
      break;
    }
    _state = IsolateSessionState.idle;
    return outputs;
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
      {required this.session,
      required this.runOptions,
      required this.inputs,
      this.outputNames});

  final int session;
  final int runOptions;
  final Map<String, int> inputs;
  final List<String>? outputNames;
}
