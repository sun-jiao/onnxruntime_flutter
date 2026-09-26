import 'ort_model_info.dart';
import 'util/session_type_info.dart';
import 'dart:ffi' as ffi;
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/services.dart';

import 'package:ffi/ffi.dart';
import 'package:onnxruntime/src/bindings/bindings.dart';
import 'package:onnxruntime/src/bindings/onnxruntime_bindings_generated.dart'
    as bg;
import 'package:onnxruntime/src/ort_env.dart';
import 'package:onnxruntime/src/ort_isolate_session.dart';
import 'package:onnxruntime/src/ort_status.dart';
import 'package:onnxruntime/src/ort_value.dart';
import 'package:onnxruntime/src/ort_provider.dart';
import 'package:onnxruntime/src/providers/ort_flags.dart';
import 'package:onnxruntime/src/util/native_path.dart';
import 'package:onnxruntime/src/util/native_memory.dart';
import 'package:onnxruntime/src/util/model_metadata.dart';
import 'package:onnxruntime/src/util/execution_provider.dart';

class OrtSession {
  bool _released = false;
  late ffi.Pointer<bg.OrtSession> _ptr;

  late int _inputCount;
  late List<String> _inputNames;
  late int _outputCount;
  late List<String> _outputNames;
  OrtIsolateSession? _isolateSession;

  void _checkNotReleased() {
    if (_released) {
      throw StateError('The session has been released.');
    }
  }

  int get address {
    _checkNotReleased();
    return _ptr.address;
  }

  int get inputCount => _inputCount;
  List<String> get inputNames => _inputNames;
  int get outputCount => _outputCount;
  List<String> get outputNames => _outputNames;
  OrtIsolateSession? get isolateSession => _isolateSession;

  /// Reads model input descriptions without changing inference behavior.
  List<OrtValueInfo> get inputInfo {
    _checkNotReleased();
    return List.unmodifiable(
      List.generate(
        _inputCount,
        (i) => readSessionValueInfo(_ptr, i, _inputNames[i], true),
      ),
    );
  }

  List<OrtValueInfo> get outputInfo {
    _checkNotReleased();
    return List.unmodifiable(
      List.generate(
        _outputCount,
        (i) => readSessionValueInfo(_ptr, i, _outputNames[i], false),
      ),
    );
  }

  /// Opt-in validation; existing run methods do not call this method.
  void validateInputs(Map<String, OrtValue> inputs) {
    _checkNotReleased();
    validateOrtInputs(
      inputInfo,
      inputs.map((name, value) {
        value.address; // Reject released wrappers.
        return MapEntry(
          name,
          value is OrtValueTensor
              ? OrtValueInfo(
                name,
                ONNXType.tensor,
                elementType: value.elementType,
                shape: value.shape,
              )
              : OrtValueInfo(
                name,
                value is OrtValueSequence
                    ? ONNXType.sequence
                    : value is OrtValueMap
                    ? ONNXType.map
                    : value is OrtValueSparseTensor
                    ? ONNXType.sparseTensor
                    : ONNXType.unknown,
              ),
        );
      }),
    );
  }

  /// Creates a session from a file.
  OrtSession.fromFile(File modelFile, OrtSessionOptions options) {
    options._checkNotReleased();
    usingNative((arena) {
      final pp = arena<ffi.Pointer<bg.OrtSession>>();
      final path = allocateOrtPath(modelFile.path,
          isWindows: Platform.isWindows, allocator: arena);
      final statusPtr = OrtEnv.instance.ortApiPtr.ref.CreateSession.asFunction<
              bg.OrtStatusPtr Function(
                  ffi.Pointer<bg.OrtEnv>,
                  ffi.Pointer<ffi.Char>,
                  ffi.Pointer<bg.OrtSessionOptions>,
                  ffi.Pointer<ffi.Pointer<bg.OrtSession>>)>()(
          OrtEnv.instance.ptr, path, options._ptr, pp);
      OrtStatus.checkOrtStatus(statusPtr);
      _ptr = pp.value;
    });
    _initOwned();
  }

  /// Creates a session from buffer.
  OrtSession.fromBuffer(Uint8List modelBuffer, OrtSessionOptions options) {
    options._checkNotReleased();
    usingNative((arena) {
      final pp = arena<ffi.Pointer<bg.OrtSession>>();
      final size = modelBuffer.length;
      final bufferPtr = arena<ffi.Uint8>(size);
      bufferPtr.asTypedList(size).setRange(0, size, modelBuffer);
      final statusPtr = OrtEnv.instance.ortApiPtr.ref.CreateSessionFromArray
              .asFunction<
                  bg.OrtStatusPtr Function(
                      ffi.Pointer<bg.OrtEnv>,
                      ffi.Pointer<ffi.Void>,
                      int,
                      ffi.Pointer<bg.OrtSessionOptions>,
                      ffi.Pointer<ffi.Pointer<bg.OrtSession>>)>()(
          OrtEnv.instance.ptr, bufferPtr.cast(), size, options._ptr, pp);
      OrtStatus.checkOrtStatus(statusPtr);
      _ptr = pp.value;
    });
    _initOwned();
  }

  /// Creates a session from a pointer's address.
  OrtSession.fromAddress(int address) {
    _ptr = ffi.Pointer.fromAddress(address);
    _init();
  }

  void _initOwned() {
    try {
      _init();
    } catch (_) {
      _releaseNative();
      rethrow;
    }
  }

  void _init() {
    _inputCount = _getInputCount();
    _inputNames = _getInputNames();
    _outputCount = _getOutputCount();
    _outputNames = _getOutputNames();
  }

  int _getInputCount() {
    return usingNative((arena) {
      final countPtr = arena<ffi.Size>();
      final statusPtr = OrtEnv.instance.ortApiPtr.ref.SessionGetInputCount
          .asFunction<
              bg.OrtStatusPtr Function(ffi.Pointer<bg.OrtSession>,
                  ffi.Pointer<ffi.Size>)>()(_ptr, countPtr);
      OrtStatus.checkOrtStatus(statusPtr);
      final count = countPtr.value;
      return count;
    });
  }

  int _getOutputCount() {
    return usingNative((arena) {
      final countPtr = arena<ffi.Size>();
      final statusPtr = OrtEnv.instance.ortApiPtr.ref.SessionGetOutputCount
          .asFunction<
              bg.OrtStatusPtr Function(ffi.Pointer<bg.OrtSession>,
                  ffi.Pointer<ffi.Size>)>()(_ptr, countPtr);
      OrtStatus.checkOrtStatus(statusPtr);
      final count = countPtr.value;
      return count;
    });
  }

  List<String> _getInputNames() {
    return usingNative((arena) {
      final list = <String>[];
      for (var i = 0; i < _inputCount; ++i) {
        final namePtrPtr = arena<ffi.Pointer<ffi.Char>>();
        var statusPtr = OrtEnv.instance.ortApiPtr.ref.SessionGetInputName
                .asFunction<
                    bg.OrtStatusPtr Function(
                        ffi.Pointer<bg.OrtSession>,
                        int,
                        ffi.Pointer<bg.OrtAllocator>,
                        ffi.Pointer<ffi.Pointer<ffi.Char>>)>()(
            _ptr, i, OrtAllocator.instance.ptr, namePtrPtr);
        OrtStatus.checkOrtStatus(statusPtr);
        arena.onReleaseAll(() {
          statusPtr = OrtEnv.instance.ortApiPtr.ref.AllocatorFree.asFunction<
                  bg.OrtStatusPtr Function(
                      ffi.Pointer<bg.OrtAllocator>, ffi.Pointer<ffi.Void>)>()(
              OrtAllocator.instance.ptr, namePtrPtr.value.cast());
          OrtStatus.checkOrtStatus(statusPtr);
        });
        final name = namePtrPtr.value.cast<Utf8>().toDartString();
        list.add(name);
      }
      return list;
    });
  }

  List<String> _getOutputNames() {
    return usingNative((arena) {
      final list = <String>[];
      for (var i = 0; i < _outputCount; ++i) {
        final namePtrPtr = arena<ffi.Pointer<ffi.Char>>();
        var statusPtr = OrtEnv.instance.ortApiPtr.ref.SessionGetOutputName
                .asFunction<
                    bg.OrtStatusPtr Function(
                        ffi.Pointer<bg.OrtSession>,
                        int,
                        ffi.Pointer<bg.OrtAllocator>,
                        ffi.Pointer<ffi.Pointer<ffi.Char>>)>()(
            _ptr, i, OrtAllocator.instance.ptr, namePtrPtr);
        OrtStatus.checkOrtStatus(statusPtr);
        arena.onReleaseAll(() {
          statusPtr = OrtEnv.instance.ortApiPtr.ref.AllocatorFree.asFunction<
                  bg.OrtStatusPtr Function(
                      ffi.Pointer<bg.OrtAllocator>, ffi.Pointer<ffi.Void>)>()(
              OrtAllocator.instance.ptr, namePtrPtr.value.cast());
          OrtStatus.checkOrtStatus(statusPtr);
        });
        final name = namePtrPtr.value.cast<Utf8>().toDartString();
        list.add(name);
      }
      return list;
    });
  }

  /// Performs inference synchronously.
  List<OrtValue?> run(OrtRunOptions runOptions, Map<String, OrtValue> inputs,
      [List<String>? outputNames]) {
    return usingNative((arena) {
      _checkNotReleased();
      runOptions._checkNotReleased();
      final inputLength = inputs.length;
      final inputNamePtrs = arena<ffi.Pointer<ffi.Char>>(inputLength);
      final inputPtrs = arena<ffi.Pointer<bg.OrtValue>>(inputLength);
      var i = 0;
      for (final entry in inputs.entries) {
        inputNamePtrs[i] =
            entry.key.toNativeUtf8(allocator: arena).cast<ffi.Char>();
        inputPtrs[i] = entry.value.ptr;
        ++i;
      }
      final selectedOutputNames = outputNames ?? _outputNames;
      final outputLength = selectedOutputNames.length;
      final outputNamePtrs = arena<ffi.Pointer<ffi.Char>>(outputLength);
      final outputPtrs = arena<ffi.Pointer<bg.OrtValue>>(outputLength);
      arena.onReleaseAll(() {
        for (var i = 0; i < outputLength; ++i) {
          if (outputPtrs[i] != ffi.nullptr) {
            OrtEnv.instance.ortApiPtr.ref.ReleaseValue
                    .asFunction<void Function(ffi.Pointer<bg.OrtValue>)>()(
                outputPtrs[i]);
          }
        }
      });
      for (int i = 0; i < outputLength; ++i) {
        outputNamePtrs[i] = selectedOutputNames[i]
            .toNativeUtf8(allocator: arena)
            .cast<ffi.Char>();
        outputPtrs[i] = ffi.nullptr;
      }
      var statusPtr = OrtEnv.instance.ortApiPtr.ref.Run.asFunction<
              bg.OrtStatusPtr Function(
                  ffi.Pointer<bg.OrtSession>,
                  ffi.Pointer<bg.OrtRunOptions>,
                  ffi.Pointer<ffi.Pointer<ffi.Char>>,
                  ffi.Pointer<ffi.Pointer<bg.OrtValue>>,
                  int,
                  ffi.Pointer<ffi.Pointer<ffi.Char>>,
                  int,
                  ffi.Pointer<ffi.Pointer<bg.OrtValue>>)>()(
          _ptr,
          runOptions._ptr,
          inputNamePtrs,
          inputPtrs,
          inputLength,
          outputNamePtrs,
          outputLength,
          outputPtrs);
      OrtStatus.checkOrtStatus(statusPtr);
      final outputs = List<OrtValue?>.generate(outputLength, (index) {
        final ortValuePtr = outputPtrs[index];
        final onnxTypePtr = arena<ffi.Int32>();
        statusPtr = OrtEnv.instance.ortApiPtr.ref.GetValueType.asFunction<
            bg.OrtStatusPtr Function(ffi.Pointer<bg.OrtValue>,
                ffi.Pointer<ffi.Int32>)>()(ortValuePtr, onnxTypePtr);
        OrtStatus.checkOrtStatus(statusPtr);
        final onnxType = ONNXType.valueOf(onnxTypePtr.value);
        switch (onnxType) {
          case ONNXType.tensor:
            return OrtValueTensor(ortValuePtr);
          case ONNXType.sequence:
            return OrtValueSequence(ortValuePtr);
          case ONNXType.map:
            return OrtValueMap(ortValuePtr);
          case ONNXType.sparseTensor:
            return OrtValueSparseTensor(ortValuePtr);
          case ONNXType.unknown:
          case ONNXType.opaque:
          case ONNXType.optional:
            return null;
        }
      });
      for (var i = 0; i < outputLength; ++i) {
        if (outputs[i] != null) {
          outputPtrs[i] = ffi.nullptr;
        }
      }
      return outputs;
    });
  }

  /// Performs inference asynchronously.
  Future<List<OrtValue?>>? runAsync(
      OrtRunOptions runOptions, Map<String, OrtValue> inputs,
      [List<String>? outputNames]) {
    if (_released) {
      return Future.value(<OrtValue?>[]);
    }
    _isolateSession ??= OrtIsolateSession(this);
    return _isolateSession?.run(runOptions, inputs, outputNames);
  }

  String getMetadatas(String key) {
    _checkNotReleased();
    return readModelMetadata(
        OrtEnv.instance.ortApiPtr, _ptr, OrtAllocator.instance.ptr, key);
  }

  /// Stops accepting runs and releases this session once any worker has exited.
  /// Already accepted asynchronous runs complete before native destruction.
  void release() {
    if (_released) {
      return;
    }
    _released = true;
    final isolateSession = _isolateSession;
    _isolateSession = null;
    if (isolateSession == null) {
      _releaseNative();
    } else {
      // Keep the void API while deferring destruction until the worker exits.
      isolateSession.release().then((_) => _releaseNative());
    }
  }

  void _releaseNative() {
    OrtEnv.instance.ortApiPtr.ref.ReleaseSession
        .asFunction<void Function(ffi.Pointer<bg.OrtSession>)>()(_ptr);
  }
}

class OrtSessionOptions {
  bool _released = false;
  late ffi.Pointer<bg.OrtSessionOptions> _ptr;

  int _intraOpNumThreads = 0;

  void _checkNotReleased() {
    if (_released) {
      throw StateError('The session options have been released.');
    }
  }

  OrtSessionOptions() {
    _create();
  }

  void _create() {
    usingNative((arena) {
      final pp = arena<ffi.Pointer<bg.OrtSessionOptions>>();
      final statusPtr = OrtEnv.instance.ortApiPtr.ref.CreateSessionOptions
          .asFunction<
              bg.OrtStatusPtr Function(
                  ffi.Pointer<ffi.Pointer<bg.OrtSessionOptions>>)>()(pp);
      OrtStatus.checkOrtStatus(statusPtr);
      _ptr = pp.value;
    });
  }

  void release() {
    if (_released) {
      return;
    }
    _released = true;
    OrtEnv.instance.ortApiPtr.ref.ReleaseSessionOptions
        .asFunction<void Function(ffi.Pointer<bg.OrtSessionOptions>)>()(_ptr);
  }

  /// Sets the number of intra op threads.
  void setIntraOpNumThreads(int numThreads) {
    _checkNotReleased();
    _intraOpNumThreads = numThreads;
    final statusPtr = OrtEnv.instance.ortApiPtr.ref.SetIntraOpNumThreads
        .asFunction<
            bg.OrtStatusPtr Function(
                ffi.Pointer<bg.OrtSessionOptions>, int)>()(_ptr, numThreads);
    OrtStatus.checkOrtStatus(statusPtr);
  }

  /// Sets the number of inter op threads.
  void setInterOpNumThreads(int numThreads) {
    _checkNotReleased();
    final statusPtr = OrtEnv.instance.ortApiPtr.ref.SetInterOpNumThreads
        .asFunction<
            bg.OrtStatusPtr Function(
                ffi.Pointer<bg.OrtSessionOptions>, int)>()(_ptr, numThreads);
    OrtStatus.checkOrtStatus(statusPtr);
  }

  /// Sets the level of session graph optimization.
  void setSessionGraphOptimizationLevel(GraphOptimizationLevel level) {
    _checkNotReleased();
    final statusPtr = OrtEnv
        .instance.ortApiPtr.ref.SetSessionGraphOptimizationLevel
        .asFunction<
            bg.OrtStatusPtr Function(
                ffi.Pointer<bg.OrtSessionOptions>, int)>()(_ptr, level.value);
    OrtStatus.checkOrtStatus(statusPtr);
  }

  bool _appendExecutionProvider(OrtProvider provider, OrtFlags flags) {
    _checkNotReleased();
    var result = false;
    bg.OrtStatusPtr? statusPtr;
    switch (provider) {
      case OrtProvider.cpu:
        statusPtr =
            onnxRuntimeBinding.OrtSessionOptionsAppendExecutionProvider_CPU(
                _ptr, flags.value);
        result = true;
        break;
      case OrtProvider.coreml:
        statusPtr =
            onnxRuntimeBinding.OrtSessionOptionsAppendExecutionProvider_CoreML(
                _ptr, flags.value);
        result = true;
        break;
      case OrtProvider.nnapi:
        statusPtr =
            onnxRuntimeBinding.OrtSessionOptionsAppendExecutionProvider_Nnapi(
                _ptr, flags.value);
        result = true;
        break;
      default:
        break;
    }
    OrtStatus.checkOrtStatus(statusPtr);
    return result;
  }

  bool _appendExecutionProvider2(
      OrtProvider provider, Map<String, String> providerOptions) {
    _checkNotReleased();
    return appendExecutionProvider(
      OrtEnv.instance.ortApiPtr,
      _ptr,
      provider,
      providerOptions,
      availableProviders: OrtEnv.instance.availableProviders,
    );
  }

  /// Appends cpu provider.
  bool appendCPUProvider(CPUFlags flags) {
    return _appendExecutionProvider(OrtProvider.cpu, flags);
  }

  /// Appends CoreML provider.
  bool appendCoreMLProvider(CoreMLFlags flags) {
    return _appendExecutionProvider(OrtProvider.coreml, flags);
  }

  /// Appends Nnapi provider.
  bool appendNnapiProvider(NnapiFlags flags) {
    return _appendExecutionProvider(OrtProvider.nnapi, flags);
  }

  /// Appends QNN provider when available, otherwise returns false.
  ///
  /// A QNN-enabled runtime and its QnnHtp backend library must be installed.
  /// Native registration errors are reported like other supported providers.
  bool appendQnnProvider() {
    return _appendExecutionProvider2(OrtProvider.qnn, {});
  }

  /// Appends Xnnpack provider.
  bool appendXnnpackProvider() {
    return _appendExecutionProvider2(OrtProvider.xnnpack,
        {'intra_op_num_threads': _intraOpNumThreads.toString()});
  }
}

class OrtRunOptions {
  bool _released = false;
  late ffi.Pointer<bg.OrtRunOptions> _ptr;

  void _checkNotReleased() {
    if (_released) {
      throw StateError('The run options have been released.');
    }
  }

  int get address {
    _checkNotReleased();
    return _ptr.address;
  }

  OrtRunOptions() {
    _create();
  }

  OrtRunOptions.fromAddress(int address) {
    _ptr = ffi.Pointer.fromAddress(address);
  }

  void _create() {
    usingNative((arena) {
      final pp = arena<ffi.Pointer<bg.OrtRunOptions>>();
      final statusPtr = OrtEnv.instance.ortApiPtr.ref.CreateRunOptions
          .asFunction<
              bg.OrtStatusPtr Function(
                  ffi.Pointer<ffi.Pointer<bg.OrtRunOptions>>)>()(pp);
      OrtStatus.checkOrtStatus(statusPtr);
      _ptr = pp.value;
    });
  }

  void release() {
    if (_released) {
      return;
    }
    _released = true;
    OrtEnv.instance.ortApiPtr.ref.ReleaseRunOptions
        .asFunction<void Function(ffi.Pointer<bg.OrtRunOptions> input)>()(_ptr);
  }

  void setRunLogVerbosityLevel(int level) {
    _checkNotReleased();
    final statusPtr = OrtEnv
        .instance.ortApiPtr.ref.RunOptionsSetRunLogVerbosityLevel
        .asFunction<
            bg.OrtStatusPtr Function(
                ffi.Pointer<bg.OrtRunOptions>, int)>()(_ptr, level);
    OrtStatus.checkOrtStatus(statusPtr);
  }

  int getRunLogVerbosityLevel() {
    _checkNotReleased();
    return usingNative((arena) {
      final levelPtr = arena<ffi.Int>();
      final statusPtr = OrtEnv
          .instance.ortApiPtr.ref.RunOptionsGetRunLogVerbosityLevel
          .asFunction<
              bg.OrtStatusPtr Function(ffi.Pointer<bg.OrtRunOptions>,
                  ffi.Pointer<ffi.Int>)>()(_ptr, levelPtr);
      OrtStatus.checkOrtStatus(statusPtr);
      final level = levelPtr.value;
      return level;
    });
  }

  void setRunLogSeverityLevel(int level) {
    _checkNotReleased();
    final statusPtr = OrtEnv
        .instance.ortApiPtr.ref.RunOptionsSetRunLogSeverityLevel
        .asFunction<
            bg.OrtStatusPtr Function(
                ffi.Pointer<bg.OrtRunOptions>, int)>()(_ptr, level);
    OrtStatus.checkOrtStatus(statusPtr);
  }

  int getRunLogSeverityLevel() {
    _checkNotReleased();
    return usingNative((arena) {
      final levelPtr = arena<ffi.Int>();
      final statusPtr = OrtEnv
          .instance.ortApiPtr.ref.RunOptionsGetRunLogSeverityLevel
          .asFunction<
              bg.OrtStatusPtr Function(ffi.Pointer<bg.OrtRunOptions>,
                  ffi.Pointer<ffi.Int>)>()(_ptr, levelPtr);
      OrtStatus.checkOrtStatus(statusPtr);
      final level = levelPtr.value;
      return level;
    });
  }

  void setRunTag(String tag) {
    _checkNotReleased();
    usingNative((arena) {
      final statusPtr = OrtEnv.instance.ortApiPtr.ref.RunOptionsSetRunTag
              .asFunction<
                  bg.OrtStatusPtr Function(
                      ffi.Pointer<bg.OrtRunOptions>, ffi.Pointer<ffi.Char>)>()(
          _ptr, tag.toNativeUtf8(allocator: arena).cast<ffi.Char>());
      OrtStatus.checkOrtStatus(statusPtr);
    });
  }

  String getRunTag() {
    _checkNotReleased();
    return usingNative((arena) {
      final tagPtr = arena<ffi.Pointer<ffi.Char>>();
      final statusPtr = OrtEnv.instance.ortApiPtr.ref.RunOptionsGetRunTag
          .asFunction<
              bg.OrtStatusPtr Function(ffi.Pointer<bg.OrtRunOptions>,
                  ffi.Pointer<ffi.Pointer<ffi.Char>>)>()(_ptr, tagPtr);
      OrtStatus.checkOrtStatus(statusPtr);
      final tag = tagPtr.value.cast<Utf8>().toDartString();
      return tag;
    });
  }

  void setTerminate() {
    _checkNotReleased();
    final statusPtr = OrtEnv.instance.ortApiPtr.ref.RunOptionsSetTerminate
        .asFunction<
            bg.OrtStatusPtr Function(ffi.Pointer<bg.OrtRunOptions>)>()(_ptr);
    OrtStatus.checkOrtStatus(statusPtr);
  }

  void unsetTerminate() {
    _checkNotReleased();
    final statusPtr = OrtEnv.instance.ortApiPtr.ref.RunOptionsUnsetTerminate
        .asFunction<
            bg.OrtStatusPtr Function(ffi.Pointer<bg.OrtRunOptions>)>()(_ptr);
    OrtStatus.checkOrtStatus(statusPtr);
  }
}

enum GraphOptimizationLevel {
  ortDisableAll(bg.GraphOptimizationLevel.ORT_DISABLE_ALL),
  ortEnableBasic(bg.GraphOptimizationLevel.ORT_ENABLE_BASIC),
  ortEnableExtended(bg.GraphOptimizationLevel.ORT_ENABLE_EXTENDED),
  ortEnableAll(bg.GraphOptimizationLevel.ORT_ENABLE_ALL);

  final int value;

  const GraphOptimizationLevel(this.value);
}
