import 'ort_types.dart';
export 'ort_types.dart';
import 'dart:ffi' as ffi;
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import 'package:onnxruntime/src/bindings/onnxruntime_bindings_generated.dart'
    as bg;
import 'package:onnxruntime/src/ort_env.dart';
import 'package:onnxruntime/src/ort_status.dart';
import 'package:onnxruntime/src/util/list_shape_extension.dart';
import 'package:onnxruntime/src/util/native_memory.dart';

abstract class OrtValue {
  bool _released = false;
  late ffi.Pointer<bg.OrtValue> _ptr;

  void _checkNotReleased() {
    if (_released) {
      throw StateError('The value has been released.');
    }
  }

  ffi.Pointer<bg.OrtValue> get ptr {
    _checkNotReleased();
    return _ptr;
  }

  int get address {
    _checkNotReleased();
    return _ptr.address;
  }

  Object? get value;

  Map<OrtTensorTypeAndShapeInfo, OrtTensorTypeAndShapeInfo> _createMapInfo(
      ffi.Pointer<bg.OrtValue> ortValuePtr) {
    return usingNative((arena) {
      final keyPtrPtr = arena<ffi.Pointer<bg.OrtValue>>();
      final keyPtr = _getOrtValue(ortValuePtr, 0, keyPtrPtr);
      arena.using(keyPtr, _releaseOrtValue);
      final keyInfo = OrtTensorTypeAndShapeInfo(keyPtr);

      final valuePtrPtr = arena<ffi.Pointer<bg.OrtValue>>();
      final valuePtr = _getOrtValue(ortValuePtr, 1, valuePtrPtr);
      arena.using(valuePtr, _releaseOrtValue);
      final valueInfo = OrtTensorTypeAndShapeInfo(valuePtr);

      return {keyInfo: valueInfo};
    });
  }

  ffi.Pointer<bg.OrtValue> _getOrtValue(ffi.Pointer<bg.OrtValue> ortValuePtr,
      int index, ffi.Pointer<ffi.Pointer<bg.OrtValue>> indexOrtValuePtrPtr) {
    final statusPtr = OrtEnv.instance.ortApiPtr.ref.GetValue.asFunction<
            bg.OrtStatusPtr Function(
                ffi.Pointer<bg.OrtValue>,
                int,
                ffi.Pointer<bg.OrtAllocator>,
                ffi.Pointer<ffi.Pointer<bg.OrtValue>>)>()(
        ortValuePtr, index, OrtAllocator.instance.ptr, indexOrtValuePtrPtr);
    OrtStatus.checkOrtStatus(statusPtr);
    return indexOrtValuePtrPtr.value;
  }

  ffi.Pointer<T> _getTensorMutableData<T extends ffi.NativeType>(
      ffi.Pointer<bg.OrtValue> ortValuePtr,
      ffi.Pointer<ffi.Pointer<T>> dataPtrPtr) {
    final statusPtr = OrtEnv.instance.ortApiPtr.ref.GetTensorMutableData
            .asFunction<
                bg.OrtStatusPtr Function(ffi.Pointer<bg.OrtValue>,
                    ffi.Pointer<ffi.Pointer<ffi.Void>>)>()(
        ortValuePtr, dataPtrPtr.cast());
    OrtStatus.checkOrtStatus(statusPtr);
    return dataPtrPtr.value;
  }

  List<String> _getStringList(ffi.Pointer<bg.OrtValue> ortValuePtr) {
    return usingNative((arena) {
      final info = OrtTensorTypeAndShapeInfo(ortValuePtr);
      final tensorShapeElementCount = info._tensorShapeElementCount;
      final dataLengthPtr = arena<ffi.Size>();
      var statusPtr = OrtEnv.instance.ortApiPtr.ref.GetStringTensorDataLength
          .asFunction<
              bg.OrtStatusPtr Function(ffi.Pointer<bg.OrtValue>,
                  ffi.Pointer<ffi.Size>)>()(ortValuePtr, dataLengthPtr);
      OrtStatus.checkOrtStatus(statusPtr);
      final dataLength = dataLengthPtr.value;

      // last index is '\0'
      final dataPtr = arena<ffi.Char>(dataLength + 1);
      // last index is dataLength
      final offsetPtr = arena<ffi.Size>(tensorShapeElementCount + 1);
      statusPtr = OrtEnv.instance.ortApiPtr.ref.GetStringTensorContent
              .asFunction<
                  bg.OrtStatusPtr Function(
                      ffi.Pointer<bg.OrtValue>,
                      ffi.Pointer<ffi.Void>,
                      int,
                      ffi.Pointer<ffi.Size>,
                      int)>()(ortValuePtr, dataPtr.cast(), dataLength,
          offsetPtr, tensorShapeElementCount);
      OrtStatus.checkOrtStatus(statusPtr);
      statusPtr = OrtEnv.instance.ortApiPtr.ref.GetStringTensorDataLength
              .asFunction<
                  bg.OrtStatusPtr Function(
                      ffi.Pointer<bg.OrtValue>, ffi.Pointer<ffi.Size>)>()(
          ortValuePtr,
          ffi.Pointer.fromAddress(offsetPtr.address +
              tensorShapeElementCount * ffi.sizeOf<ffi.Size>()));
      OrtStatus.checkOrtStatus(statusPtr);
      final list = <String>[];
      for (int i = 0; i < tensorShapeElementCount; ++i) {
        final size = offsetPtr[i + 1] - offsetPtr[i];
        final strPtr = arena<ffi.Char>(size + 1);
        for (int j = 0; j < size; ++j) {
          strPtr[j] = dataPtr[offsetPtr[i] + j];
        }
        final str = strPtr.cast<Utf8>().toDartString();
        list.add(str);
      }

      return list;
    });
  }

  List<num> _getNumList(ffi.Pointer<bg.OrtValue> ortValuePtr) {
    return usingNative((arena) {
      final info = OrtTensorTypeAndShapeInfo(ortValuePtr);
      final tensorElementType = info._tensorElementType;
      final tensorShapeElementCount = info._tensorShapeElementCount;
      final data = <num>[];
      if (tensorElementType == ONNXTensorElementDataType.uint8) {
        final dataPtrPtr = arena<ffi.Pointer<ffi.Uint8>>();
        final dataPtr = _getTensorMutableData(ortValuePtr, dataPtrPtr);
        for (int i = 0; i < tensorShapeElementCount; ++i) {
          data.add(dataPtr[i]);
        }
      } else if (tensorElementType == ONNXTensorElementDataType.int8) {
        final dataPtrPtr = arena<ffi.Pointer<ffi.Int8>>();
        final dataPtr = _getTensorMutableData(ortValuePtr, dataPtrPtr);
        for (int i = 0; i < tensorShapeElementCount; ++i) {
          data.add(dataPtr[i]);
        }
      } else if (tensorElementType == ONNXTensorElementDataType.uint16) {
        final dataPtrPtr = arena<ffi.Pointer<ffi.Uint16>>();
        final dataPtr = _getTensorMutableData(ortValuePtr, dataPtrPtr);
        for (int i = 0; i < tensorShapeElementCount; ++i) {
          data.add(dataPtr[i]);
        }
      } else if (tensorElementType == ONNXTensorElementDataType.int16) {
        final dataPtrPtr = arena<ffi.Pointer<ffi.Int16>>();
        final dataPtr = _getTensorMutableData(ortValuePtr, dataPtrPtr);
        for (int i = 0; i < tensorShapeElementCount; ++i) {
          data.add(dataPtr[i]);
        }
      } else if (tensorElementType == ONNXTensorElementDataType.uint32) {
        final dataPtrPtr = arena<ffi.Pointer<ffi.Uint32>>();
        final dataPtr = _getTensorMutableData(ortValuePtr, dataPtrPtr);
        for (int i = 0; i < tensorShapeElementCount; ++i) {
          data.add(dataPtr[i]);
        }
      } else if (tensorElementType == ONNXTensorElementDataType.int32) {
        final dataPtrPtr = arena<ffi.Pointer<ffi.Int32>>();
        final dataPtr = _getTensorMutableData(ortValuePtr, dataPtrPtr);
        for (int i = 0; i < tensorShapeElementCount; ++i) {
          data.add(dataPtr[i]);
        }
      } else if (tensorElementType == ONNXTensorElementDataType.uint64) {
        final dataPtrPtr = arena<ffi.Pointer<ffi.Uint64>>();
        final dataPtr = _getTensorMutableData(ortValuePtr, dataPtrPtr);
        for (int i = 0; i < tensorShapeElementCount; ++i) {
          data.add(dataPtr[i]);
        }
      } else if (tensorElementType == ONNXTensorElementDataType.int64) {
        final dataPtrPtr = arena<ffi.Pointer<ffi.Int64>>();
        final dataPtr = _getTensorMutableData(ortValuePtr, dataPtrPtr);
        for (int i = 0; i < tensorShapeElementCount; ++i) {
          data.add(dataPtr[i]);
        }
      } else if (tensorElementType == ONNXTensorElementDataType.float) {
        final dataPtrPtr = arena<ffi.Pointer<ffi.Float>>();
        final dataPtr = _getTensorMutableData(ortValuePtr, dataPtrPtr);
        for (int i = 0; i < tensorShapeElementCount; ++i) {
          data.add(dataPtr[i]);
        }
      } else if (tensorElementType == ONNXTensorElementDataType.double) {
        final dataPtrPtr = arena<ffi.Pointer<ffi.Double>>();
        final dataPtr = _getTensorMutableData(ortValuePtr, dataPtrPtr);
        for (int i = 0; i < tensorShapeElementCount; ++i) {
          data.add(dataPtr[i]);
        }
      }
      return data;
    });
  }

  List<bool> _getBoolList(ffi.Pointer<bg.OrtValue> ortValuePtr) {
    return usingNative((arena) {
      final info = OrtTensorTypeAndShapeInfo(ortValuePtr);
      final tensorElementType = info._tensorElementType;
      final tensorShapeElementCount = info._tensorShapeElementCount;
      final data = <bool>[];
      if (tensorElementType == ONNXTensorElementDataType.bool) {
        final dataPtrPtr = arena<ffi.Pointer<ffi.Bool>>();
        final dataPtr = _getTensorMutableData(ortValuePtr, dataPtrPtr);
        for (int i = 0; i < tensorShapeElementCount; ++i) {
          data.add(dataPtr[i]);
        }
      }
      return data;
    });
  }

  void _releaseOrtValue(ffi.Pointer<bg.OrtValue> ortValuePtr) {
    OrtEnv.instance.ortApiPtr.ref.ReleaseValue
        .asFunction<void Function(ffi.Pointer<bg.OrtValue>)>()(ortValuePtr);
  }

  void release() {
    if (_released) {
      return;
    }
    _released = true;
    _releaseOrtValue(_ptr);
  }
}

class OrtValueTensor extends OrtValue {
  /// Returns a flat, independently owned copy of numeric tensor data.
  /// The copy survives release() and modifications never affect this tensor.
  /// Bool, string, half-precision and complex values are unsupported.
  TypedData toTypedData() {
    _checkNotReleased();
    final count = _info._tensorShapeElementCount;
    return usingNative((arena) {
      switch (_info._tensorElementType) {
        case ONNXTensorElementDataType.uint8:
          if (count == 0) return Uint8List(0);
          final out = arena<ffi.Pointer<ffi.Uint8>>();
          final data = _getTensorMutableData(_ptr, out);
          return Uint8List.fromList(data.asTypedList(count));
        case ONNXTensorElementDataType.int8:
          if (count == 0) return Int8List(0);
          final out = arena<ffi.Pointer<ffi.Int8>>();
          final data = _getTensorMutableData(_ptr, out);
          return Int8List.fromList(data.asTypedList(count));
        case ONNXTensorElementDataType.uint16:
          if (count == 0) return Uint16List(0);
          final out = arena<ffi.Pointer<ffi.Uint16>>();
          final data = _getTensorMutableData(_ptr, out);
          return Uint16List.fromList(data.asTypedList(count));
        case ONNXTensorElementDataType.int16:
          if (count == 0) return Int16List(0);
          final out = arena<ffi.Pointer<ffi.Int16>>();
          final data = _getTensorMutableData(_ptr, out);
          return Int16List.fromList(data.asTypedList(count));
        case ONNXTensorElementDataType.uint32:
          if (count == 0) return Uint32List(0);
          final out = arena<ffi.Pointer<ffi.Uint32>>();
          final data = _getTensorMutableData(_ptr, out);
          return Uint32List.fromList(data.asTypedList(count));
        case ONNXTensorElementDataType.int32:
          if (count == 0) return Int32List(0);
          final out = arena<ffi.Pointer<ffi.Int32>>();
          final data = _getTensorMutableData(_ptr, out);
          return Int32List.fromList(data.asTypedList(count));
        case ONNXTensorElementDataType.uint64:
          if (count == 0) return Uint64List(0);
          final out = arena<ffi.Pointer<ffi.Uint64>>();
          final data = _getTensorMutableData(_ptr, out);
          return Uint64List.fromList(data.asTypedList(count));
        case ONNXTensorElementDataType.int64:
          if (count == 0) return Int64List(0);
          final out = arena<ffi.Pointer<ffi.Int64>>();
          final data = _getTensorMutableData(_ptr, out);
          return Int64List.fromList(data.asTypedList(count));
        case ONNXTensorElementDataType.float:
          if (count == 0) return Float32List(0);
          final out = arena<ffi.Pointer<ffi.Float>>();
          final data = _getTensorMutableData(_ptr, out);
          return Float32List.fromList(data.asTypedList(count));
        case ONNXTensorElementDataType.double:
          if (count == 0) return Float64List(0);
          final out = arena<ffi.Pointer<ffi.Double>>();
          final data = _getTensorMutableData(_ptr, out);
          return Float64List.fromList(data.asTypedList(count));
        default:
          throw UnsupportedError('TypedData extraction is not supported for '
              '${_info._tensorElementType}.');
      }
    });
  }

  ONNXTensorElementDataType get elementType {
    _checkNotReleased();
    return _info._tensorElementType;
  }

  List<int> get shape {
    _checkNotReleased();
    return List.unmodifiable(_info._tensorShape);
  }


  late OrtTensorTypeAndShapeInfo _info;
  ffi.Pointer<ffi.Void> _dataPtr = ffi.nullptr;
  ffi.Allocator _dataAllocator = calloc;

  OrtValueTensor(ffi.Pointer<bg.OrtValue> ptr,
      [ffi.Pointer<ffi.Void>? dataPtr]) {
    _ptr = ptr;
    _info = OrtTensorTypeAndShapeInfo(ptr);
    if (dataPtr != null) {
      _dataPtr = dataPtr;
    }
  }

  factory OrtValueTensor.fromAddress(int address) {
    return OrtValueTensor(ffi.Pointer.fromAddress(address));
  }

  // Only inferred shapes require rectangular nesting. An explicit shape may
  // intentionally reshape any nested input after flattening it.
  static void _validateInferredShape(List data, List<int> shape,
      [int depth = 0]) {
    if (depth >= shape.length || data.length != shape[depth]) {
      throw ArgumentError.value(
          data, 'data', 'Cannot infer a rectangular shape.');
    }
    for (final item in data) {
      if (item is List) {
        _validateInferredShape(item, shape, depth + 1);
      } else if (depth != shape.length - 1) {
        throw ArgumentError.value(
            data, 'data', 'Cannot infer a rectangular shape.');
      }
    }
  }

  // Keep native errors for negative dimensions, undersized numeric buffers and
  // out-of-bounds string writes. Reject the opposite mismatches that ORT accepts
  // by ignoring extra numeric data or leaving string elements unfilled.
  static void _validatePreviouslyAcceptedShape(
      List<int> shape, int elementCount,
      {required bool strings}) {
    var expected = BigInt.one;
    for (final dimension in shape) {
      if (dimension < 0) {
        return;
      }
      expected *= BigInt.from(dimension);
    }
    final actual = BigInt.from(elementCount);
    if (strings ? expected > actual : expected < actual) {
      throw ArgumentError.value(shape, 'shape',
          'Expected $expected elements, but received $elementCount.');
    }
  }

  static OrtValueTensor _createTensorWithString(String data) {
    return _createTensorWithStringList(<String>[data], []);
  }

  static OrtValueTensor _createTensorWithStringList(List<String> data,
      [List<int>? shape]) {
    final selectedShape = shape ?? data.shape;
    _validatePreviouslyAcceptedShape(selectedShape, data.length, strings: true);
    return usingNative((arena) {
      final ortValuePtrPtr = arena<ffi.Pointer<bg.OrtValue>>();
      var transferred = false;
      arena.onReleaseAll(() {
        if (!transferred && ortValuePtrPtr.value != ffi.nullptr) {
          OrtEnv.instance.ortApiPtr.ref.ReleaseValue
                  .asFunction<void Function(ffi.Pointer<bg.OrtValue>)>()(
              ortValuePtrPtr.value);
        }
      });
      final shapeSize = selectedShape.length;
      final shapePtr = arena<ffi.Int64>(shapeSize);
      shapePtr.asTypedList(shapeSize).setRange(0, shapeSize, selectedShape);

      var statusPtr = OrtEnv.instance.ortApiPtr.ref.CreateTensorAsOrtValue
              .asFunction<
                  bg.OrtStatusPtr Function(
                      ffi.Pointer<bg.OrtAllocator>,
                      ffi.Pointer<ffi.Int64> shape,
                      int,
                      int,
                      ffi.Pointer<ffi.Pointer<bg.OrtValue>>)>()(
          OrtAllocator.instance.ptr,
          shapePtr,
          shapeSize,
          ONNXTensorElementDataType.string.value,
          ortValuePtrPtr);
      OrtStatus.checkOrtStatus(statusPtr);
      final ortValuePtr = ortValuePtrPtr.value;
      for (int i = 0; i < data.length; ++i) {
        final str = data[i].toNativeUtf8(allocator: arena).cast<ffi.Char>();
        statusPtr = OrtEnv.instance.ortApiPtr.ref.FillStringTensorElement
            .asFunction<
                bg.OrtStatusPtr Function(ffi.Pointer<bg.OrtValue>,
                    ffi.Pointer<ffi.Char>, int)>()(ortValuePtr, str, i);
        OrtStatus.checkOrtStatus(statusPtr);
      }

      final tensor = OrtValueTensor(ortValuePtr);
      transferred = true;
      return tensor;
    });
  }

  static OrtValueTensor createTensorWithData(dynamic data) {
    if (data is int) {
      return createTensorWithDataList(<int>[data], []);
    }
    if (data is double) {
      return createTensorWithDataList(<double>[data], []);
    }
    if (data is bool) {
      return createTensorWithDataList(<bool>[data], []);
    }
    if (data is String) {
      return _createTensorWithString(data);
    }
    throw Exception('Invalid element type');
  }

  static OrtValueTensor createTensorWithDataList(List data,
      [List<int>? shape]) {
    return usingNative((arena) {
      final selectedShape = shape ?? data.shape;
      if (shape == null && data.isNotEmpty) {
        _validateInferredShape(data, selectedShape);
      }
      final element = data.element();
      var dataType = ONNXTensorElementDataType.undefined;
      ffi.Pointer<ffi.Void> dataPtr = ffi.nullptr;
      final dataAllocator = nativeAllocator;
      var transferred = false;
      arena.onReleaseAll(() {
        if (!transferred && dataPtr != ffi.nullptr) {
          dataAllocator.free(dataPtr);
        }
      });
      int dataSize = 0;
      int dataByteCount = 0;
      if (element is Uint8List) {
        final flattenData = data.flatten<int>();
        dataSize = flattenData.length;
        dataType = ONNXTensorElementDataType.uint8;
        dataPtr = (dataAllocator<ffi.Uint8>(dataSize)
              ..asTypedList(dataSize).setRange(0, dataSize, flattenData))
            .cast();
        dataByteCount = dataSize;
      } else if (element is Int8List) {
        final flattenData = data.flatten<int>();
        dataSize = flattenData.length;
        dataType = ONNXTensorElementDataType.int8;
        dataPtr = (dataAllocator<ffi.Int8>(dataSize)
              ..asTypedList(dataSize).setRange(0, dataSize, flattenData))
            .cast();
        dataByteCount = dataSize;
      } else if (element is Uint16List) {
        final flattenData = data.flatten<int>();
        dataSize = flattenData.length;
        dataType = ONNXTensorElementDataType.uint16;
        dataPtr = (dataAllocator<ffi.Uint16>(dataSize)
              ..asTypedList(dataSize).setRange(0, dataSize, flattenData))
            .cast();
        dataByteCount = dataSize * 2;
      } else if (element is Int16List) {
        final flattenData = data.flatten<int>();
        dataSize = flattenData.length;
        dataType = ONNXTensorElementDataType.int16;
        dataPtr = (dataAllocator<ffi.Int16>(dataSize)
              ..asTypedList(dataSize).setRange(0, dataSize, flattenData))
            .cast();
        dataByteCount = dataSize * 2;
      } else if (element is Uint32List) {
        final flattenData = data.flatten<int>();
        dataSize = flattenData.length;
        dataType = ONNXTensorElementDataType.uint32;
        dataPtr = (dataAllocator<ffi.Uint32>(dataSize)
              ..asTypedList(dataSize).setRange(0, dataSize, flattenData))
            .cast();
        dataByteCount = dataSize * 4;
      } else if (element is Int32List) {
        final flattenData = data.flatten<int>();
        dataSize = flattenData.length;
        dataType = ONNXTensorElementDataType.int32;
        dataPtr = (dataAllocator<ffi.Int32>(dataSize)
              ..asTypedList(dataSize).setRange(0, dataSize, flattenData))
            .cast();
        dataByteCount = dataSize * 4;
      } else if (element is Uint64List) {
        final flattenData = data.flatten<int>();
        dataSize = flattenData.length;
        dataType = ONNXTensorElementDataType.uint64;
        dataPtr = (dataAllocator<ffi.Uint64>(dataSize)
              ..asTypedList(dataSize).setRange(0, dataSize, flattenData))
            .cast();
        dataByteCount = dataSize * 8;
      } else if (element is Int64List || element is int) {
        final flattenData = data.flatten<int>();
        dataSize = flattenData.length;
        dataType = ONNXTensorElementDataType.int64;
        dataPtr = (dataAllocator<ffi.Int64>(dataSize)
              ..asTypedList(dataSize).setRange(0, dataSize, flattenData))
            .cast();
        dataByteCount = dataSize * 8;
      } else if (element is Float32List) {
        final flattenData = data.flatten<double>();
        dataSize = flattenData.length;
        dataType = ONNXTensorElementDataType.float;
        dataPtr = (dataAllocator<ffi.Float>(dataSize)
              ..asTypedList(dataSize).setRange(0, dataSize, flattenData))
            .cast();
        dataByteCount = dataSize * 4;
      } else if (element is Float64List || element is double) {
        final flattenData = data.flatten<double>();
        dataSize = flattenData.length;
        dataType = ONNXTensorElementDataType.double;
        dataPtr = (dataAllocator<ffi.Double>(dataSize)
              ..asTypedList(dataSize).setRange(0, dataSize, flattenData))
            .cast();
        dataByteCount = dataSize * 8;
      } else if (element is bool) {
        final flattenData = data.flatten<bool>();
        dataSize = flattenData.length;
        dataType = ONNXTensorElementDataType.bool;
        final ptr = dataAllocator<ffi.Bool>(dataSize);
        for (int i = 0; i < dataSize; ++i) {
          ptr[i] = flattenData[i];
        }
        dataPtr = ptr.cast();
        dataByteCount = dataSize;
      } else if (element is String) {
        // Keep non-string leaves so the cast still rejects mixed element types.
        return _createTensorWithStringList(
            data.flatten<dynamic>().cast<String>(), selectedShape);
      } else {
        throw Exception('Invalid inputTensor element type.');
      }

      _validatePreviouslyAcceptedShape(selectedShape, dataSize, strings: false);
      final shapeSize = selectedShape.length;
      final shapePtr = arena<ffi.Int64>(shapeSize);
      shapePtr.asTypedList(shapeSize).setRange(0, shapeSize, selectedShape);

      final ortMemoryInfoPtrPtr = arena<ffi.Pointer<bg.OrtMemoryInfo>>();
      var statusPtr = OrtEnv.instance.ortApiPtr.ref.AllocatorGetInfo.asFunction<
              bg.OrtStatusPtr Function(ffi.Pointer<bg.OrtAllocator>,
                  ffi.Pointer<ffi.Pointer<bg.OrtMemoryInfo>>)>()(
          OrtAllocator.instance.ptr, ortMemoryInfoPtrPtr);
      OrtStatus.checkOrtStatus(statusPtr);
      // or
      // OrtEnv.instance.ortApiPtr.ref.CreateCpuMemoryInfo.asFunction<
      //         bg.OrtStatusPtr Function(
      //             int, int, ffi.Pointer<ffi.Pointer<bg.OrtMemoryInfo>>)>()(
      //     bg.OrtAllocatorType.OrtDeviceAllocator,
      //     bg.OrtMemType.OrtMemTypeCPU,
      //     ortMemoryInfoPtrPtr);
      final ortMemoryInfoPtr = ortMemoryInfoPtrPtr.value;
      final ortValuePtrPtr = arena<ffi.Pointer<bg.OrtValue>>();
      arena.onReleaseAll(() {
        if (!transferred && ortValuePtrPtr.value != ffi.nullptr) {
          OrtEnv.instance.ortApiPtr.ref.ReleaseValue
                  .asFunction<void Function(ffi.Pointer<bg.OrtValue>)>()(
              ortValuePtrPtr.value);
        }
      });
      statusPtr = OrtEnv.instance.ortApiPtr.ref.CreateTensorWithDataAsOrtValue
              .asFunction<
                  bg.OrtStatusPtr Function(
                      ffi.Pointer<bg.OrtMemoryInfo>,
                      ffi.Pointer<ffi.Void>,
                      int,
                      ffi.Pointer<ffi.Int64>,
                      int,
                      int,
                      ffi.Pointer<ffi.Pointer<bg.OrtValue>>)>()(
          ortMemoryInfoPtr,
          dataPtr,
          dataByteCount,
          shapePtr,
          shapeSize,
          dataType.value,
          ortValuePtrPtr);
      OrtStatus.checkOrtStatus(statusPtr);
      final ortValuePtr = ortValuePtrPtr.value;
      final tensor = OrtValueTensor(ortValuePtr, dataPtr);
      tensor._dataAllocator = dataAllocator;
      transferred = true;
      return tensor;
    });
  }

  @override
  dynamic get value {
    _checkNotReleased();
    if (_info._dimensionsCount == 0) {
      // scalar tensor
      switch (_info._tensorElementType) {
        case ONNXTensorElementDataType.uint8:
        case ONNXTensorElementDataType.int8:
        case ONNXTensorElementDataType.uint16:
        case ONNXTensorElementDataType.int16:
        case ONNXTensorElementDataType.uint32:
        case ONNXTensorElementDataType.int32:
        case ONNXTensorElementDataType.uint64:
        case ONNXTensorElementDataType.int64:
        case ONNXTensorElementDataType.float:
        case ONNXTensorElementDataType.double:
          return _getNumList(_ptr)[0];
        case ONNXTensorElementDataType.bool:
          return _getBoolList(_ptr)[0];
        case ONNXTensorElementDataType.string:
          return _getStringList(_ptr)[0];
        default:
          throw Exception('Extracting the value of an invalid Tensor.');
      }
    } else {
      // vector tensor
      switch (_info._tensorElementType) {
        case ONNXTensorElementDataType.uint8:
        case ONNXTensorElementDataType.int8:
        case ONNXTensorElementDataType.uint16:
        case ONNXTensorElementDataType.int16:
        case ONNXTensorElementDataType.uint32:
        case ONNXTensorElementDataType.int32:
        case ONNXTensorElementDataType.uint64:
        case ONNXTensorElementDataType.int64:
          return _getNumList(_ptr).reshape<int>(_info._tensorShape);
        case ONNXTensorElementDataType.float:
        case ONNXTensorElementDataType.double:
          return _getNumList(_ptr).reshape<double>(_info._tensorShape);
        case ONNXTensorElementDataType.bool:
          return _getBoolList(_ptr).reshape<bool>(_info._tensorShape);
        case ONNXTensorElementDataType.string:
          return _getStringList(_ptr).reshape<String>(_info._tensorShape);
        default:
          throw Exception('Extracting the value of an invalid Tensor.');
      }
    }
  }

  @override
  void release() {
    super.release();
    if (_dataPtr == ffi.nullptr) {
      return;
    }
    _dataAllocator.free(_dataPtr);
    _dataPtr = ffi.nullptr;
  }
}

class OrtValueSequence extends OrtValue {
  int _valueCount = 0;
  var _onnxType = ONNXType.unknown;
  OrtTensorTypeAndShapeInfo? _tensorInfo;

  // OrtTensorTypeAndShapeInfo? _firstMapKeyInfo;
  // OrtTensorTypeAndShapeInfo? _firstMapValueInfo;

  OrtValueSequence(ffi.Pointer<bg.OrtValue> ptr) {
    usingNative((arena) {
      _ptr = ptr;
      final valueCountPtr = arena<ffi.Size>();
      var statusPtr = OrtEnv.instance.ortApiPtr.ref.GetValueCount.asFunction<
          bg.OrtStatusPtr Function(ffi.Pointer<bg.OrtValue>,
              ffi.Pointer<ffi.Size>)>()(_ptr, valueCountPtr);
      OrtStatus.checkOrtStatus(statusPtr);
      _valueCount = valueCountPtr.value;

      if (_valueCount <= 0) {
        return;
      }
      final firstElementPtrPtr = arena<ffi.Pointer<bg.OrtValue>>();
      final firstElementPtr = _getOrtValue(_ptr, 0, firstElementPtrPtr);
      arena.using(firstElementPtr, _releaseOrtValue);
      final onnxTypePtr = arena<ffi.Int32>();
      statusPtr = OrtEnv.instance.ortApiPtr.ref.GetValueType.asFunction<
          bg.OrtStatusPtr Function(ffi.Pointer<bg.OrtValue>,
              ffi.Pointer<ffi.Int32>)>()(firstElementPtr, onnxTypePtr);
      OrtStatus.checkOrtStatus(statusPtr);
      _onnxType = ONNXType.valueOf(onnxTypePtr.value);
      if (_onnxType == ONNXType.tensor) {
        _tensorInfo = OrtTensorTypeAndShapeInfo(firstElementPtr);
      } else if (_onnxType == ONNXType.map) {
        // final infoMap = _createMapInfo(firstElementPtr);
        // _firstMapKeyInfo = infoMap.entries.first.key;
        // _firstMapValueInfo = infoMap.entries.first.value;
      }
    });
  }

  OrtValueSequence.fromAddress(int address)
      : this(ffi.Pointer.fromAddress(address));

  @override
  List<OrtValue>? get value {
    _checkNotReleased();
    // Empty sequences have no first element from which to infer a child type.
    if (_valueCount == 0) {
      return <OrtValue>[];
    }
    return usingNative((arena) {
      var transferred = false;
      if (_onnxType == ONNXType.map) {
        final maps = <OrtValueMap>[];
        for (int i = 0; i < _valueCount; ++i) {
          final ortValuePtrPtr = arena<ffi.Pointer<bg.OrtValue>>();
          final ortValuePtr = _getOrtValue(_ptr, i, ortValuePtrPtr);
          arena.onReleaseAll(() {
            if (!transferred) {
              _releaseOrtValue(ortValuePtr);
            }
          });
          maps.add(OrtValueMap(ortValuePtr));
        }
        transferred = true;
        return maps;
      } else if (_onnxType == ONNXType.tensor) {
        switch (_tensorInfo?._tensorElementType) {
          case ONNXTensorElementDataType.string:
          case ONNXTensorElementDataType.int64:
          case ONNXTensorElementDataType.float:
          case ONNXTensorElementDataType.double:
            final tensors = <OrtValueTensor>[];
            for (int i = 0; i < _valueCount; ++i) {
              final ortValuePtrPtr = arena<ffi.Pointer<bg.OrtValue>>();
              final ortValuePtr = _getOrtValue(_ptr, i, ortValuePtrPtr);
              arena.onReleaseAll(() {
                if (!transferred) {
                  _releaseOrtValue(ortValuePtr);
                }
              });
              tensors.add(OrtValueTensor(ortValuePtr));
            }
            transferred = true;
            return tensors;
          default:
            throw Exception(
                'Unsupported type in a sequence, found ${_tensorInfo?._tensorElementType}');
        }
      } else {
        throw Exception("Invalid element type found in sequence");
      }
    });
  }
}

class OrtValueMap extends OrtValue {
  late OrtTensorTypeAndShapeInfo _keyInfo;
  late OrtTensorTypeAndShapeInfo _valueInfo;

  OrtValueMap(ffi.Pointer<bg.OrtValue> ptr) {
    _ptr = ptr;
    final infoMap = _createMapInfo(ptr);
    _keyInfo = infoMap.entries.first.key;
    _valueInfo = infoMap.entries.first.value;
  }

  OrtValueMap.fromAddress(int address) : this(ffi.Pointer.fromAddress(address));

  @override
  Map get value {
    _checkNotReleased();
    final keys = _getMapKeys();
    final values = _getMapValues();
    final map = {};
    for (int i = 0; i < keys.length; ++i) {
      map[keys[i]] = values[i];
    }
    return map;
  }

  List<dynamic> _getMapKeys() {
    switch (_keyInfo._tensorElementType) {
      case ONNXTensorElementDataType.string:
        return _getStringListWithIndex(0);
      case ONNXTensorElementDataType.int64:
        return _getNumListWithIndex(0);
      default:
        throw Exception(
            'Invalid or unknown valueType: ${_keyInfo._tensorElementType}');
    }
  }

  List<String> _getStringListWithIndex(int index) {
    return usingNative((arena) {
      final ortValuePtrPtr = arena<ffi.Pointer<bg.OrtValue>>();
      final ortValuePtr = _getOrtValue(_ptr, index, ortValuePtrPtr);
      arena.using(ortValuePtr, _releaseOrtValue);
      final list = _getStringList(ortValuePtr);

      return list;
    });
  }

  List<num> _getNumListWithIndex(int index) {
    return usingNative((arena) {
      final ortValuePtrPtr = arena<ffi.Pointer<bg.OrtValue>>();
      final ortValuePtr = _getOrtValue(_ptr, index, ortValuePtrPtr);
      arena.using(ortValuePtr, _releaseOrtValue);
      final list = _getNumList(ortValuePtr);

      return list;
    });
  }

  List<Object> _getMapValues() {
    switch (_valueInfo._tensorElementType) {
      case ONNXTensorElementDataType.string:
        return _getStringListWithIndex(1);
      case ONNXTensorElementDataType.int64:
      case ONNXTensorElementDataType.float:
      case ONNXTensorElementDataType.double:
        return _getNumListWithIndex(1);
      default:
        throw Exception(
            'Invalid or unknown valueType: ${_keyInfo._tensorElementType}');
    }
  }

  int get size => _keyInfo._tensorShapeElementCount;
}

class OrtValueSparseTensor extends OrtValue {
  // ignore: unused_field
  late OrtTensorTypeAndShapeInfo _info;
  late OrtSparseFormat _ortSparseFormat;

  OrtValueSparseTensor(ffi.Pointer<bg.OrtValue> ptr) {
    usingNative((arena) {
      _ptr = ptr;
      _info = OrtTensorTypeAndShapeInfo(ptr);
      final ortSparseFormatPtr = arena<ffi.Int32>();
      final statusPtr = OrtEnv.instance.ortApiPtr.ref.GetSparseTensorFormat
          .asFunction<
              bg.OrtStatusPtr Function(ffi.Pointer<bg.OrtValue>,
                  ffi.Pointer<ffi.Int32>)>()(ptr, ortSparseFormatPtr);
      OrtStatus.checkOrtStatus(statusPtr);
      _ortSparseFormat = OrtSparseFormat.valueOf(ortSparseFormatPtr.value);
    });
  }

  OrtValueSparseTensor.fromAddress(int address)
      : this(ffi.Pointer.fromAddress(address));

  @override
  // ignore: body_might_complete_normally_nullable
  Object? get value {
    _checkNotReleased();
    switch (_ortSparseFormat) {
      case OrtSparseFormat.coo:
        // TODO: Handle this case.
        break;
      case OrtSparseFormat.csrc:
        // TODO: Handle this case.
        break;
      case OrtSparseFormat.blockSparse:
        // TODO: Handle this case.
        break;
      case OrtSparseFormat.undefined:
        throw Exception('Undefined sparsity type in this sparse tensor.');
    }
  }
}

class OrtTensorTypeAndShapeInfo {
  int _dimensionsCount = 0;
  int _tensorShapeElementCount = 0;
  ONNXTensorElementDataType _tensorElementType =
      ONNXTensorElementDataType.undefined;
  List<int> _tensorShape = [];

  OrtTensorTypeAndShapeInfo(ffi.Pointer<bg.OrtValue> ortValuePtr) {
    usingNative((arena) {
      final infoPtrPtr = arena<ffi.Pointer<bg.OrtTensorTypeAndShapeInfo>>();
      final statusPtr = OrtEnv.instance.ortApiPtr.ref.GetTensorTypeAndShape
              .asFunction<
                  bg.OrtStatusPtr Function(
                      ffi.Pointer<bg.OrtValue>,
                      ffi.Pointer<
                          ffi.Pointer<bg.OrtTensorTypeAndShapeInfo>>)>()(
          ortValuePtr, infoPtrPtr);
      OrtStatus.checkOrtStatus(statusPtr);
      final infoPtr = infoPtrPtr.value;
      arena.using(infoPtr, _releaseTensorTypeAndShapeInfo);
      _tensorElementType = _getTensorElementType(infoPtr);
      // shape
      _dimensionsCount = _getDimensionsCount(infoPtr);
      _tensorShape = _getDimensions(infoPtr, _dimensionsCount);
      _tensorShapeElementCount = _getTensorShapeElementCount(infoPtr);
    });
  }

  static ONNXTensorElementDataType _getTensorElementType(
      ffi.Pointer<bg.OrtTensorTypeAndShapeInfo> infoPtr) {
    return usingNative((arena) {
      final onnxTensorElementDataTypePtr = arena<ffi.Int32>();
      final statusPtr = OrtEnv.instance.ortApiPtr.ref.GetTensorElementType
              .asFunction<
                  bg.OrtStatusPtr Function(
                      ffi.Pointer<bg.OrtTensorTypeAndShapeInfo>,
                      ffi.Pointer<ffi.Int32>)>()(
          infoPtr, onnxTensorElementDataTypePtr);
      OrtStatus.checkOrtStatus(statusPtr);
      final onnxTensorElementDataType = onnxTensorElementDataTypePtr.value;

      return ONNXTensorElementDataType.valueOf(onnxTensorElementDataType);
    });
  }

  static void _releaseTensorTypeAndShapeInfo(
      ffi.Pointer<bg.OrtTensorTypeAndShapeInfo> infoPtr) {
    OrtEnv.instance.ortApiPtr.ref.ReleaseTensorTypeAndShapeInfo.asFunction<
        void Function(ffi.Pointer<bg.OrtTensorTypeAndShapeInfo>)>()(infoPtr);
  }

  static int _getDimensionsCount(
      ffi.Pointer<bg.OrtTensorTypeAndShapeInfo> infoPtr) {
    return usingNative((arena) {
      final countPtr = arena<ffi.Size>();
      final statusPtr = OrtEnv.instance.ortApiPtr.ref.GetDimensionsCount
          .asFunction<
              bg.OrtStatusPtr Function(
                  ffi.Pointer<bg.OrtTensorTypeAndShapeInfo>,
                  ffi.Pointer<ffi.Size>)>()(infoPtr, countPtr);
      OrtStatus.checkOrtStatus(statusPtr);
      final count = countPtr.value;

      return count;
    });
  }

  static List<int> _getDimensions(
      ffi.Pointer<bg.OrtTensorTypeAndShapeInfo> infoPtr, int length) {
    return usingNative((arena) {
      final dimensionsPtr = arena<ffi.Int64>(length);
      final statusPtr = OrtEnv.instance.ortApiPtr.ref.GetDimensions.asFunction<
          bg.OrtStatusPtr Function(ffi.Pointer<bg.OrtTensorTypeAndShapeInfo>,
              ffi.Pointer<ffi.Int64>, int)>()(infoPtr, dimensionsPtr, length);
      OrtStatus.checkOrtStatus(statusPtr);
      final dimensions =
          List<int>.generate(length, (index) => dimensionsPtr[index]);

      return dimensions;
    });
  }

  static int _getTensorShapeElementCount(
      ffi.Pointer<bg.OrtTensorTypeAndShapeInfo> infoPtr) {
    return usingNative((arena) {
      final countPtr = arena<ffi.Size>();
      final statusPtr = OrtEnv.instance.ortApiPtr.ref.GetTensorShapeElementCount
          .asFunction<
              bg.OrtStatusPtr Function(
                  ffi.Pointer<bg.OrtTensorTypeAndShapeInfo>,
                  ffi.Pointer<ffi.Size>)>()(infoPtr, countPtr);
      OrtStatus.checkOrtStatus(statusPtr);
      final count = countPtr.value;

      return count;
    });
  }
}

enum OrtSparseFormat {
  undefined(bg.OrtSparseFormat.ORT_SPARSE_UNDEFINED),
  coo(bg.OrtSparseFormat.ORT_SPARSE_COO),
  csrc(bg.OrtSparseFormat.ORT_SPARSE_CSRC),
  blockSparse(bg.OrtSparseFormat.ORT_SPARSE_BLOCK_SPARSE);

  final int value;

  const OrtSparseFormat(this.value);

  static OrtSparseFormat valueOf(int type) {
    switch (type) {
      case bg.OrtSparseFormat.ORT_SPARSE_COO:
        return OrtSparseFormat.coo;
      case bg.OrtSparseFormat.ORT_SPARSE_CSRC:
        return OrtSparseFormat.csrc;
      case bg.OrtSparseFormat.ORT_SPARSE_BLOCK_SPARSE:
        return OrtSparseFormat.blockSparse;
      default:
        return OrtSparseFormat.undefined;
    }
  }
}
