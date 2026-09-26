part of 'ort_web.dart';

abstract class OrtValue {
  bool _released = false;
  void _check() {
    if (_released) throw StateError('The value has been released.');
  }

  Object get ptr {
    _check();
    return _nativeOnly('OrtValue.ptr');
  }

  int get address {
    _check();
    return _nativeOnly('OrtValue.address');
  }

  Object? get value;
  void release() => _released = true;
}

class OrtValueTensor extends OrtValue {
  List<int> get shape {
    _check();
    return List.unmodifiable(_tensor.dims.toDart.map((d) => d.toDartInt));
  }

  ONNXTensorElementDataType get elementType {
    _check();
    const names = {
      'float32': 1,
      'uint8': 2,
      'int8': 3,
      'uint16': 4,
      'int16': 5,
      'int32': 6,
      'int64': 7,
      'string': 8,
      'bool': 9,
      'float16': 10,
      'float64': 11,
      'uint32': 12,
      'uint64': 13,
      'bfloat16': 16,
    };
    return ONNXTensorElementDataType.valueOf(names[_tensor.type.toDart] ?? 0);
  }

  late final _Tensor _tensor;
  OrtValueTensor(Object ptr, [Object? dataPtr]) {
    _nativeOnly('OrtValueTensor pointer constructor');
  }
  OrtValueTensor.fromAddress(int address) {
    _nativeOnly('OrtValueTensor.fromAddress');
  }
  OrtValueTensor._(String type, JSAny data, List<int> shape) {
    // Check explicitly so a missing script produces an actionable Dart error.
    _runtime;
    _tensor = _Tensor(type.toJS, data, shape.map((d) => d.toJS).toList().toJS);
  }
  OrtValueTensor._copy(_Tensor tensor) {
    final data = tensor.data as JSObject;
    final type = tensor.type.toDart;
    // TypedArray.slice and Array.slice both make independently owned copies.
    final copy = data.callMethod<JSAny>('slice'.toJS);
    _tensor = _Tensor(type.toJS, copy, tensor.dims);
  }

  static OrtValueTensor createTensorWithData(dynamic data) {
    if (data is! num && data is! bool && data is! String) {
      throw ArgumentError.value(data, 'data', 'Invalid element type.');
    }
    return createTensorWithDataList([data], []);
  }

  static OrtValueTensor createTensorWithDataList(
    List data, [
    List<int>? shape,
  ]) {
    final selected = shape == null ? _inferShape(data) : List<int>.of(shape);
    final flat = <Object?>[];
    void flatten(List list) {
      for (final item in list) {
        if (item is List) {
          flatten(item);
        } else {
          flat.add(item);
        }
      }
    }

    flatten(data);
    var count = BigInt.one;
    for (final dim in selected) {
      if (dim < 0) {
        throw ArgumentError.value(shape, 'shape', 'Negative dimension.');
      }
      count *= BigInt.from(dim);
    }
    if (count != BigInt.from(flat.length)) {
      throw ArgumentError(
        'Shape $selected requires $count elements, got ${flat.length}.',
      );
    }
    dynamic element = data;
    while (element is List && element is! TypedData && element.isNotEmpty) {
      element = element.first;
    }
    String type;
    JSAny values;
    if (element is Uint8List) {
      type = 'uint8';
      values = Uint8List.fromList(flat.cast<int>()).toJS;
    } else if (element is Int8List) {
      type = 'int8';
      values = Int8List.fromList(flat.cast<int>()).toJS;
    } else if (element is Uint16List) {
      type = 'uint16';
      values = Uint16List.fromList(flat.cast<int>()).toJS;
    } else if (element is Int16List) {
      type = 'int16';
      values = Int16List.fromList(flat.cast<int>()).toJS;
    } else if (element is Uint32List) {
      type = 'uint32';
      values = Uint32List.fromList(flat.cast<int>()).toJS;
    } else if (element is Int32List) {
      type = 'int32';
      values = Int32List.fromList(flat.cast<int>()).toJS;
    } else if (element is Uint64List ||
        element is Int64List ||
        element is int) {
      type = element is Uint64List ? 'uint64' : 'int64';
      final integers =
          flat
              .cast<int>()
              .map((n) {
                if (n.abs() > 9007199254740991) {
                  throw UnsupportedError(
                    'Web Dart integers must be within the exact '
                    'JavaScript range (±9007199254740991).',
                  );
                }
                return _bigInt(n.toString().toJS);
              })
              .toList()
              .toJS;
      values =
          type == 'uint64'
              ? _BigUint64Array(integers)
              : _BigInt64Array(integers);
    } else if (element is Float32List) {
      type = 'float32';
      values =
          Float32List.fromList(
            flat.cast<num>().map((v) => v.toDouble()).toList(),
          ).toJS;
    } else if (element is Float64List || element is double) {
      type = 'float64';
      values =
          Float64List.fromList(
            flat.cast<num>().map((v) => v.toDouble()).toList(),
          ).toJS;
    } else if (element is bool) {
      type = 'bool';
      values =
          Uint8List.fromList(
            flat.cast<bool>().map((b) => b ? 1 : 0).toList(),
          ).toJS;
    } else if (element is String) {
      type = 'string';
      values = flat.cast<String>().map((s) => s.toJS).toList().toJS;
    } else {
      throw ArgumentError(
        'Invalid or untyped empty tensor data. Use a typed list.',
      );
    }
    return OrtValueTensor._(type, values, selected);
  }

  static List<int> _inferShape(List list) {
    if (list.isEmpty) return [0];
    final first = list.first;
    final child = first is List ? _inferShape(first) : <int>[];
    for (final value in list) {
      final other = value is List ? _inferShape(value) : <int>[];
      if (child.length != other.length ||
          List.generate(
            child.length,
            (i) => child[i] != other[i],
          ).contains(true)) {
        throw ArgumentError('Cannot infer a rectangular shape.');
      }
    }
    return [list.length, ...child];
  }

  @override
  dynamic get value {
    _check();
    final type = _tensor.type.toDart;
    final shape = _tensor.dims.toDart.map((d) => d.toDartInt).toList();
    final data = _tensor.data;
    List flat;
    if (type == 'int64' || type == 'uint64') {
      final array = data as JSObject;
      final length = array.getProperty<JSNumber>('length'.toJS).toDartInt;
      flat = List<int>.generate(length, (i) {
        final text = _jsString(array.getProperty<JSBigInt>(i.toJS)).toDart;
        final integer = BigInt.parse(text);
        if (integer.abs() > BigInt.from(9007199254740991)) {
          throw UnsupportedError(
            'int64/uint64 output exceeds the exact Web Dart integer range.',
          );
        }
        return integer.toInt();
      });
    } else {
      flat = data.dartify() as List;
      if (type == 'bool') flat = flat.map((n) => n != 0).toList();
    }
    if (type == 'float16' || type == 'bfloat16') {
      throw UnsupportedError(
        'Extracting $type tensor values is not supported.',
      );
    }
    if (shape.isEmpty) return flat.single;
    if (type == 'float32' || type == 'float64') {
      return flat.reshape<double>(shape);
    }
    if (type == 'string') return flat.reshape<String>(shape);
    if (type == 'bool') return flat.reshape<bool>(shape);
    return flat.reshape<int>(shape);
  }

  @override
  void release() {
    if (_released) return;
    _tensor.dispose();
    super.release();
  }
}

class OrtValueSequence extends OrtValue {
  OrtValueSequence(Object ptr) {
    _nativeOnly('OrtValueSequence');
  }
  OrtValueSequence.fromAddress(int address) {
    _nativeOnly('OrtValueSequence.fromAddress');
  }
  @override
  List<OrtValue>? get value {
    _check();
    return _nativeOnly('OrtValueSequence.value');
  }
}

class OrtValueMap extends OrtValue {
  OrtValueMap(Object ptr) {
    _nativeOnly('OrtValueMap');
  }
  OrtValueMap.fromAddress(int address) {
    _nativeOnly('OrtValueMap.fromAddress');
  }
  @override
  Map get value {
    _check();
    return _nativeOnly('OrtValueMap.value');
  }

  int get size {
    _check();
    return _nativeOnly('OrtValueMap.size');
  }
}

class OrtValueSparseTensor extends OrtValue {
  OrtValueSparseTensor(Object ptr) {
    _nativeOnly('OrtValueSparseTensor');
  }
  OrtValueSparseTensor.fromAddress(int address) {
    _nativeOnly('OrtValueSparseTensor.fromAddress');
  }
  @override
  Object? get value {
    _check();
    return _nativeOnly('OrtValueSparseTensor.value');
  }
}

class OrtTensorTypeAndShapeInfo {
  OrtTensorTypeAndShapeInfo(Object ortValuePtr) {
    _nativeOnly('OrtTensorTypeAndShapeInfo');
  }
}
