part of 'ort_value.dart';

void _releaseExtraValue(ffi.Pointer<bg.OrtValue> value) => OrtEnv
    .instance
    .ortApiPtr
    .ref
    .ReleaseValue
    .asFunction<void Function(ffi.Pointer<bg.OrtValue>)>()(value);

OrtValue _wrapOwnedValue(ffi.Pointer<bg.OrtValue> pointer) {
  try {
    return usingNative((arena) {
      final kind = arena<ffi.Int32>();
      OrtStatus.checkOrtStatus(
        OrtEnv.instance.ortApiPtr.ref.GetValueType
            .asFunction<
              bg.OrtStatusPtr Function(
                ffi.Pointer<bg.OrtValue>,
                ffi.Pointer<ffi.Int32>,
              )
            >()(pointer, kind),
      );
      switch (ONNXType.valueOf(kind.value)) {
        case ONNXType.tensor:
          return OrtValueTensor(pointer);
        case ONNXType.sequence:
          return OrtValueSequence(pointer);
        case ONNXType.map:
          return OrtValueMap(pointer);
        case ONNXType.sparseTensor:
          return OrtValueSparseTensor(pointer);
        default:
          throw UnsupportedError(
            'Unsupported composite child type ${kind.value}.',
          );
      }
    });
  } catch (_) {
    _releaseExtraValue(pointer);
    rethrow;
  }
}

int _elementSize(ONNXTensorElementDataType type) {
  switch (type) {
    case ONNXTensorElementDataType.uint8:
    case ONNXTensorElementDataType.int8:
    case ONNXTensorElementDataType.bool:
      return 1;
    case ONNXTensorElementDataType.uint16:
    case ONNXTensorElementDataType.int16:
    case ONNXTensorElementDataType.float16:
    case ONNXTensorElementDataType.bFloat16:
      return 2;
    case ONNXTensorElementDataType.uint32:
    case ONNXTensorElementDataType.int32:
    case ONNXTensorElementDataType.float:
      return 4;
    case ONNXTensorElementDataType.uint64:
    case ONNXTensorElementDataType.int64:
    case ONNXTensorElementDataType.double:
      return 8;
    default:
      throw UnsupportedError('Unsupported numeric type $type.');
  }
}

OrtValueTensor _ownedNumericTensor(
  ONNXTensorElementDataType type,
  List<int> shape,
  Uint8List bytes,
) {
  final width = _elementSize(type);
  if (bytes.length % width != 0) {
    throw ArgumentError('Invalid data byte count.');
  }
  validateTensorShape(shape, bytes.length ~/ width);
  return usingNative((arena) {
    final out = arena<ffi.Pointer<bg.OrtValue>>();
    final dims = arena<ffi.Int64>(shape.length)
      ..asTypedList(shape.length).setAll(0, shape);
    final api = OrtEnv.instance.ortApiPtr.ref;
    OrtStatus.checkOrtStatus(
      api.CreateTensorAsOrtValue.asFunction<
        bg.OrtStatusPtr Function(
          ffi.Pointer<bg.OrtAllocator>,
          ffi.Pointer<ffi.Int64>,
          int,
          int,
          ffi.Pointer<ffi.Pointer<bg.OrtValue>>,
        )
      >()(OrtAllocator.instance.ptr, dims, shape.length, type.value, out),
    );
    try {
      if (bytes.isNotEmpty) {
        final data = arena<ffi.Pointer<ffi.Void>>();
        OrtStatus.checkOrtStatus(
          api.GetTensorMutableData.asFunction<
            bg.OrtStatusPtr Function(
              ffi.Pointer<bg.OrtValue>,
              ffi.Pointer<ffi.Pointer<ffi.Void>>,
            )
          >()(out.value, data),
        );
        data.value.cast<ffi.Uint8>().asTypedList(bytes.length).setAll(0, bytes);
      }
      return OrtValueTensor(out.value);
    } catch (_) {
      _releaseExtraValue(out.value);
      rethrow;
    }
  });
}

OrtValueTensor _halfTensor(
  Uint16List bits,
  List<int> shape,
  ONNXTensorElementDataType type,
) => _ownedNumericTensor(type, shape, Uint8List.sublistView(bits));

OrtValueTensor _copyTensor(OrtValueTensor source) {
  source._checkNotReleased();
  if (source.elementType == ONNXTensorElementDataType.string) {
    return OrtValueTensor.createTensorWithDataList(
      source._getStringList(source._ptr),
      source.shape,
    );
  }
  return usingNative((arena) {
    final length =
        source._info._tensorShapeElementCount *
        _elementSize(source.elementType);
    final out = arena<ffi.Pointer<ffi.Uint8>>();
    final bytes =
        length == 0
            ? Uint8List(0)
            : source
                ._getTensorMutableData(source._ptr, out)
                .asTypedList(length);
    return _ownedNumericTensor(source.elementType, source.shape, bytes);
  });
}

OrtValue _ownedComposite(List<OrtValueTensor> sources, ONNXType type) {
  final copies = <OrtValueTensor>[];
  try {
    for (final source in sources) {
      copies.add(_copyTensor(source));
    }
    return usingNative((arena) {
      final values = arena<ffi.Pointer<bg.OrtValue>>(copies.length);
      for (var i = 0; i < copies.length; i++) {
        values[i] = copies[i]._ptr;
      }
      final out = arena<ffi.Pointer<bg.OrtValue>>();
      OrtStatus.checkOrtStatus(
        OrtEnv.instance.ortApiPtr.ref.CreateValue
            .asFunction<
              bg.OrtStatusPtr Function(
                ffi.Pointer<ffi.Pointer<bg.OrtValue>>,
                int,
                int,
                ffi.Pointer<ffi.Pointer<bg.OrtValue>>,
              )
            >()(values, copies.length, type.value, out),
      );
      return _wrapOwnedValue(out.value);
    });
  } finally {
    for (final copy in copies) {
      copy.release();
    }
  }
}

OrtValueSparseTensor _createSparse(
  TypedData data,
  List<int> shape,
  Int64List indices, {
  Int64List? outer,
  List<int>? blockValuesShape,
  Int32List? blockIndices,
  List<int>? blockIndicesShape,
}) {
  if (data is! List || data is ByteData) {
    throw ArgumentError('Use a numeric typed list.');
  }
  if (shape.isEmpty || shape.any((d) => d < 0)) {
    throw ArgumentError('Invalid sparse dense shape.');
  }
  final values = OrtValueTensor.createTensorWithDataList(data as List, [
    (data as List).length,
  ]);
  try {
    final count = values._info._tensorShapeElementCount;
    final total = shape.fold<BigInt>(BigInt.one, (p, d) => p * BigInt.from(d));
    if (BigInt.from(count) > total) {
      throw ArgumentError('Too many sparse values.');
    }
    if (blockIndices != null) {
      validateTensorShape(blockValuesShape!, count);
      validateTensorShape(blockIndicesShape!, blockIndices.length);
      if (blockIndices.any((i) => i < 0)) {
        throw ArgumentError('Negative block index.');
      }
    } else if (outer == null) {
      if (indices.length != count && indices.length != count * shape.length) {
        throw ArgumentError(
          'COO indices must be linear or coordinate indices.',
        );
      }
      for (var i = 0; i < indices.length; i++) {
        final limit =
            indices.length == count
                ? total
                : BigInt.from(shape[i % shape.length]);
        if (indices[i] < 0 || BigInt.from(indices[i]) >= limit) {
          throw ArgumentError('COO index out of bounds.');
        }
      }
    } else {
      if (shape.length != 2 ||
          indices.length != count ||
          outer.length != shape[0] + 1 ||
          outer.first != 0 ||
          outer.last != count) {
        throw ArgumentError('Invalid CSR layout.');
      }
      for (var i = 1; i < outer.length; i++) {
        if (outer[i] < outer[i - 1] || outer[i] > count) {
          throw ArgumentError('CSR offsets must be monotonic.');
        }
      }
      if (indices.any((i) => i < 0 || i >= shape[1])) {
        throw ArgumentError('CSR column out of bounds.');
      }
    }
    return usingNative((arena) {
      final api = OrtEnv.instance.ortApiPtr.ref;
      final dense = arena<ffi.Int64>(shape.length)
        ..asTypedList(shape.length).setAll(0, shape);
      final out = arena<ffi.Pointer<bg.OrtValue>>();
      OrtStatus.checkOrtStatus(
        api.CreateSparseTensorAsOrtValue.asFunction<
          bg.OrtStatusPtr Function(
            ffi.Pointer<bg.OrtAllocator>,
            ffi.Pointer<ffi.Int64>,
            int,
            int,
            ffi.Pointer<ffi.Pointer<bg.OrtValue>>,
          )
        >()(
          OrtAllocator.instance.ptr,
          dense,
          shape.length,
          values.elementType.value,
          out,
        ),
      );
      try {
        final memory = arena<ffi.Pointer<bg.OrtMemoryInfo>>();
        OrtStatus.checkOrtStatus(
          api.AllocatorGetInfo.asFunction<
            bg.OrtStatusPtr Function(
              ffi.Pointer<bg.OrtAllocator>,
              ffi.Pointer<ffi.Pointer<bg.OrtMemoryInfo>>,
            )
          >()(OrtAllocator.instance.ptr, memory),
        );
        final valueDimensions = blockValuesShape ?? [count];
        final valuesShape = arena<ffi.Int64>(valueDimensions.length)
          ..asTypedList(valueDimensions.length).setAll(0, valueDimensions);
        final valuePtr = arena<ffi.Pointer<ffi.Void>>();
        values._getTensorMutableData(values._ptr, valuePtr);
        final inner = arena<ffi.Int64>(indices.length)
          ..asTypedList(indices.length).setAll(0, indices);
        if (blockIndices != null) {
          final block = arena<ffi.Int32>(blockIndices.length)
            ..asTypedList(blockIndices.length).setAll(0, blockIndices);
          final indexShape = arena<ffi.Int64>(
            blockIndicesShape!.length,
          )..asTypedList(blockIndicesShape.length).setAll(0, blockIndicesShape);
          OrtStatus.checkOrtStatus(
            api.FillSparseTensorBlockSparse.asFunction<
              bg.OrtStatusPtr Function(
                ffi.Pointer<bg.OrtValue>,
                ffi.Pointer<bg.OrtMemoryInfo>,
                ffi.Pointer<ffi.Int64>,
                int,
                ffi.Pointer<ffi.Void>,
                ffi.Pointer<ffi.Int64>,
                int,
                ffi.Pointer<ffi.Int32>,
              )
            >()(
              out.value,
              memory.value,
              valuesShape,
              valueDimensions.length,
              valuePtr.value,
              indexShape,
              blockIndicesShape.length,
              block,
            ),
          );
        } else if (outer == null) {
          OrtStatus.checkOrtStatus(
            api.FillSparseTensorCoo.asFunction<
              bg.OrtStatusPtr Function(
                ffi.Pointer<bg.OrtValue>,
                ffi.Pointer<bg.OrtMemoryInfo>,
                ffi.Pointer<ffi.Int64>,
                int,
                ffi.Pointer<ffi.Void>,
                ffi.Pointer<ffi.Int64>,
                int,
              )
            >()(
              out.value,
              memory.value,
              valuesShape,
              1,
              valuePtr.value,
              inner,
              indices.length,
            ),
          );
        } else {
          final rows = arena<ffi.Int64>(outer.length)
            ..asTypedList(outer.length).setAll(0, outer);
          OrtStatus.checkOrtStatus(
            api.FillSparseTensorCsr.asFunction<
              bg.OrtStatusPtr Function(
                ffi.Pointer<bg.OrtValue>,
                ffi.Pointer<bg.OrtMemoryInfo>,
                ffi.Pointer<ffi.Int64>,
                int,
                ffi.Pointer<ffi.Void>,
                ffi.Pointer<ffi.Int64>,
                int,
                ffi.Pointer<ffi.Int64>,
                int,
              )
            >()(
              out.value,
              memory.value,
              valuesShape,
              1,
              valuePtr.value,
              inner,
              indices.length,
              rows,
              outer.length,
            ),
          );
        }
        return OrtValueSparseTensor(out.value);
      } catch (_) {
        _releaseExtraValue(out.value);
        rethrow;
      }
    });
  } finally {
    values.release();
  }
}

({ONNXTensorElementDataType type, List<int> shape, int count}) _takeInfo(
  ffi.Pointer<bg.OrtTensorTypeAndShapeInfo> info,
) {
  try {
    return (
      type: OrtTensorTypeAndShapeInfo._getTensorElementType(info),
      shape: OrtTensorTypeAndShapeInfo._getDimensions(
        info,
        OrtTensorTypeAndShapeInfo._getDimensionsCount(info),
      ),
      count: OrtTensorTypeAndShapeInfo._getTensorShapeElementCount(info),
    );
  } finally {
    OrtTensorTypeAndShapeInfo._releaseTensorTypeAndShapeInfo(info);
  }
}

TypedData _copyNumericPointer(
  ffi.Pointer<ffi.Void> pointer,
  ONNXTensorElementDataType type,
  int count,
) {
  // An empty tensor may have a null data pointer. Avoid dereferencing it.
  switch (type) {
    case ONNXTensorElementDataType.float:
      return count == 0
          ? Float32List(0)
          : Float32List.fromList(pointer.cast<ffi.Float>().asTypedList(count));
    case ONNXTensorElementDataType.double:
      return count == 0
          ? Float64List(0)
          : Float64List.fromList(pointer.cast<ffi.Double>().asTypedList(count));
    case ONNXTensorElementDataType.int8:
      return count == 0
          ? Int8List(0)
          : Int8List.fromList(pointer.cast<ffi.Int8>().asTypedList(count));
    case ONNXTensorElementDataType.uint8:
    case ONNXTensorElementDataType.bool:
      return count == 0
          ? Uint8List(0)
          : Uint8List.fromList(pointer.cast<ffi.Uint8>().asTypedList(count));
    case ONNXTensorElementDataType.int16:
      return count == 0
          ? Int16List(0)
          : Int16List.fromList(pointer.cast<ffi.Int16>().asTypedList(count));
    case ONNXTensorElementDataType.uint16:
    case ONNXTensorElementDataType.float16:
    case ONNXTensorElementDataType.bFloat16:
      return count == 0
          ? Uint16List(0)
          : Uint16List.fromList(pointer.cast<ffi.Uint16>().asTypedList(count));
    case ONNXTensorElementDataType.int32:
      return count == 0
          ? Int32List(0)
          : Int32List.fromList(pointer.cast<ffi.Int32>().asTypedList(count));
    case ONNXTensorElementDataType.uint32:
      return count == 0
          ? Uint32List(0)
          : Uint32List.fromList(pointer.cast<ffi.Uint32>().asTypedList(count));
    case ONNXTensorElementDataType.int64:
      return count == 0
          ? Int64List(0)
          : Int64List.fromList(pointer.cast<ffi.Int64>().asTypedList(count));
    case ONNXTensorElementDataType.uint64:
      return count == 0
          ? Uint64List(0)
          : Uint64List.fromList(pointer.cast<ffi.Uint64>().asTypedList(count));
    default:
      throw UnsupportedError('Sparse extraction does not support $type.');
  }
}

OrtSparseTensorData _readSparse(OrtValueSparseTensor tensor) =>
    usingNative((arena) {
      if (tensor._ortSparseFormat == OrtSparseFormat.undefined) {
        throw StateError('Sparse tensor is not initialized.');
      }
      final api = OrtEnv.instance.ortApiPtr.ref;
      final info = arena<ffi.Pointer<bg.OrtTensorTypeAndShapeInfo>>();
      OrtStatus.checkOrtStatus(
        api.GetSparseTensorValuesTypeAndShape.asFunction<
          bg.OrtStatusPtr Function(
            ffi.Pointer<bg.OrtValue>,
            ffi.Pointer<ffi.Pointer<bg.OrtTensorTypeAndShapeInfo>>,
          )
        >()(tensor._ptr, info),
      );
      final valuesInfo = _takeInfo(info.value);
      final data = arena<ffi.Pointer<ffi.Void>>();
      OrtStatus.checkOrtStatus(
        api.GetSparseTensorValues.asFunction<
          bg.OrtStatusPtr Function(
            ffi.Pointer<bg.OrtValue>,
            ffi.Pointer<ffi.Pointer<ffi.Void>>,
          )
        >()(tensor._ptr, data),
      );
      final values = _copyNumericPointer(
        data.value,
        valuesInfo.type,
        valuesInfo.count,
      );
      final indices = <String, TypedData>{};
      List<int>? blockShape;
      final formats =
          tensor._ortSparseFormat == OrtSparseFormat.coo
              ? {'coo': 0}
              : tensor._ortSparseFormat == OrtSparseFormat.csrc
              ? {'inner': 1, 'outer': 2}
              : {'block': 3};
      for (final entry in formats.entries) {
        OrtStatus.checkOrtStatus(
          api.GetSparseTensorIndicesTypeShape.asFunction<
            bg.OrtStatusPtr Function(
              ffi.Pointer<bg.OrtValue>,
              int,
              ffi.Pointer<ffi.Pointer<bg.OrtTensorTypeAndShapeInfo>>,
            )
          >()(tensor._ptr, entry.value, info),
        );
        final indexInfo = _takeInfo(info.value);
        final count = arena<ffi.Size>();
        OrtStatus.checkOrtStatus(
          api.GetSparseTensorIndices.asFunction<
            bg.OrtStatusPtr Function(
              ffi.Pointer<bg.OrtValue>,
              int,
              ffi.Pointer<ffi.Size>,
              ffi.Pointer<ffi.Pointer<ffi.Void>>,
            )
          >()(tensor._ptr, entry.value, count, data),
        );
        indices[entry.key] = _copyNumericPointer(
          data.value,
          indexInfo.type,
          count.value,
        );
        if (entry.key == 'block') blockShape = indexInfo.shape;
      }
      return OrtSparseTensorData(
        tensor._ortSparseFormat,
        valuesInfo.type,
        tensor._info._tensorShape,
        valuesInfo.shape,
        values,
        indices,
        blockIndicesShape: blockShape,
      );
    });
