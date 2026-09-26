import 'dart:math' as math;
import 'dart:typed_data';

int _roundEven(double value) {
  final floor = value.floor();
  final fraction = value - floor;
  return fraction > 0.5 || (fraction == 0.5 && floor.isOdd) ? floor + 1 : floor;
}

Uint16List encodeHalf(List<double> values, {required bool bfloat}) {
  final result = Uint16List(values.length);
  final bytes = ByteData(4);
  for (var i = 0; i < values.length; i++) {
    final value = values[i];
    if (bfloat) {
      bytes.setFloat32(0, value, Endian.little);
      final bits = bytes.getUint32(0, Endian.little);
      result[i] =
          value.isNaN ? 0x7fc0 : ((bits + 0x7fff + ((bits >>> 16) & 1)) >>> 16);
      continue;
    }
    final sign = value.isNegative ? 0x8000 : 0;
    final magnitude = value.abs();
    if (value.isNaN) {
      result[i] = 0x7e00;
    } else if (magnitude >= 65520) {
      result[i] = sign | 0x7c00;
    } else if (magnitude < 1 / 16384) {
      result[i] = sign | _roundEven(magnitude * 16777216);
    } else {
      var exponent = (math.log(magnitude) / math.ln2).floor();
      var scale = math.pow(2, exponent).toDouble();
      while (magnitude < scale) {
        exponent--;
        scale /= 2;
      }
      while (magnitude >= scale * 2) {
        exponent++;
        scale *= 2;
      }
      var significand = _roundEven(magnitude / scale * 1024);
      if (significand == 2048) {
        significand = 1024;
        exponent++;
      }
      result[i] = sign | ((exponent + 15) << 10) | (significand - 1024);
    }
  }
  return result;
}

Float32List decodeHalf(Uint16List bits, {required bool bfloat}) {
  final result = Float32List(bits.length);
  final bytes = ByteData(4);
  for (var i = 0; i < bits.length; i++) {
    final raw = bits[i];
    if (bfloat) {
      bytes.setUint32(0, raw << 16, Endian.little);
      result[i] = bytes.getFloat32(0, Endian.little);
    } else {
      final exponent = (raw >>> 10) & 31, mantissa = raw & 1023;
      final sign = (raw & 0x8000) == 0 ? 1.0 : -1.0;
      result[i] =
          exponent == 31
              ? (mantissa == 0 ? sign * double.infinity : double.nan)
              : exponent == 0
              ? sign * mantissa / 16777216
              : sign * (1 + mantissa / 1024) * math.pow(2, exponent - 15);
    }
  }
  return result;
}

void validateTensorShape(List<int> shape, int count) {
  var product = BigInt.one;
  for (final dimension in shape) {
    if (dimension < 0) {
      throw ArgumentError('Tensor dimensions must be nonnegative.');
    }
    product *= BigInt.from(dimension);
  }
  if (product != BigInt.from(count)) {
    throw ArgumentError('Shape does not match data length.');
  }
}
