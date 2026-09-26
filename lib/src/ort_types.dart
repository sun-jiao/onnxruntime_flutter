enum ONNXTensorElementDataType {
  undefined(0),
  float(1),
  uint8(2),
  int8(3),
  uint16(4),
  int16(5),
  int32(6),
  int64(7),
  string(8),
  bool(9),
  float16(10),
  double(11),
  uint32(12),
  uint64(13),
  complex64(14),
  complex128(15),
  bFloat16(16);

  final int value;

  const ONNXTensorElementDataType(this.value);

  static ONNXTensorElementDataType valueOf(int type) {
    switch (type) {
      case 1:
        return ONNXTensorElementDataType.float;
      case 2:
        return ONNXTensorElementDataType.uint8;
      case 3:
        return ONNXTensorElementDataType.int8;
      case 4:
        return ONNXTensorElementDataType.uint16;
      case 5:
        return ONNXTensorElementDataType.int16;
      case 6:
        return ONNXTensorElementDataType.int32;
      case 7:
        return ONNXTensorElementDataType.int64;
      case 8:
        return ONNXTensorElementDataType.string;
      case 9:
        return ONNXTensorElementDataType.bool;
      case 10:
        return ONNXTensorElementDataType.float16;
      case 11:
        return ONNXTensorElementDataType.double;
      case 12:
        return ONNXTensorElementDataType.uint32;
      case 13:
        return ONNXTensorElementDataType.uint64;
      case 14:
        return ONNXTensorElementDataType.complex64;
      case 15:
        return ONNXTensorElementDataType.complex128;
      case 16:
        return ONNXTensorElementDataType.bFloat16;
      default:
        return ONNXTensorElementDataType.undefined;
    }
  }
}

enum ONNXType {
  unknown(0),
  tensor(1),
  sequence(2),
  map(3),
  opaque(4),
  sparseTensor(5),
  optional(6);

  final int value;

  const ONNXType(this.value);

  static ONNXType valueOf(int type) {
    switch (type) {
      case 1:
        return ONNXType.tensor;
      case 2:
        return ONNXType.sequence;
      case 3:
        return ONNXType.map;
      case 4:
        return ONNXType.opaque;
      case 5:
        return ONNXType.sparseTensor;
      case 6:
        return ONNXType.optional;
      default:
        return ONNXType.unknown;
    }
  }
}

enum OrtSparseFormat {
  undefined(0),
  coo(1),
  csrc(2),
  blockSparse(4);

  final int value;

  const OrtSparseFormat(this.value);

  static OrtSparseFormat valueOf(int type) {
    switch (type) {
      case 1:
        return OrtSparseFormat.coo;
      case 2:
        return OrtSparseFormat.csrc;
      case 4:
        return OrtSparseFormat.blockSparse;
      default:
        return OrtSparseFormat.undefined;
    }
  }
}

