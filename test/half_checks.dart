import 'dart:typed_data';
import 'package:onnxruntime/src/util/half_float.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:onnxruntime/onnxruntime.dart';

void checkHalf({required bool web}) {
  final raw = Uint16List.fromList([
    0,
    0x8000,
    1,
    0x3e00,
    0x7bff,
    0x7c00,
    0xfc00,
    0x7e01,
  ]);
  final tensor = OrtValueTensor.fromFloat16Bits(raw, [2, 4]);
  try {
    expect(tensor.toHalfBits(), raw);
    final decoded = tensor.toFloat32List();
    expect(decoded.take(5), [0.0, -0.0, 1 / 16777216, 1.5, 65504.0]);
    expect(decoded[1].isNegative, isTrue);
    expect(decoded[5], double.infinity);
    expect(decoded[6], double.negativeInfinity);
    expect(decoded[7].isNaN, isTrue);
    raw[3] = 0;
    expect(tensor.toFloat32List()[3], 1.5);
    expect(() => tensor.value, throwsA(anything));
    expect(() => tensor.toTypedData(), throwsUnsupportedError);
    tensor.release();
    expect(decoded[3], 1.5);
    expect(() => tensor.toHalfBits(), throwsStateError);
  } finally {
    tensor.release();
  }
  final rounded = OrtValueTensor.fromFloat16(
    [1 + 1 / 2048, 1 + 3 / 2048, 1 / 33554432, 65520],
    [4],
  );
  try {
    expect(rounded.toHalfBits(), [0x3c00, 0x3c02, 0, 0x7c00]);
  } finally {
    rounded.release();
  }
  final empty = OrtValueTensor.fromFloat16Bits(Uint16List(0), [0]);
  try {
    expect(empty.toFloat32List(), isEmpty);
  } finally {
    empty.release();
  }
  expect(
    () => OrtValueTensor.fromFloat16Bits(Uint16List(2), [3]),
    throwsArgumentError,
  );
  if (web) {
    expect(() => OrtValueTensor.fromBFloat16([1], [1]), throwsUnsupportedError);
  } else {
    final bfloat = OrtValueTensor.fromBFloat16(
      [1.5, -2, -0.0, double.infinity, double.nan],
      [5],
    );
    try {
      expect(bfloat.toHalfBits().take(4), [0x3fc0, 0xc000, 0x8000, 0x7f80]);
      expect(bfloat.toFloat32List()[4].isNaN, isTrue);
      expect(() => bfloat.value, throwsA(anything));
    } finally {
      bfloat.release();
    }
  }
}

Future<void> checkHalfInference(
  OrtSession session, {
  bool bfloat = false,
}) async {
  final input =
      bfloat
          ? OrtValueTensor.fromBFloat16([1.5, -2], [2])
          : OrtValueTensor.fromFloat16([1.5, -2], [2]);
  final options = OrtRunOptions();
  try {
    final outputs = await session.runAsyncOrThrow(options, {'input': input});
    try {
      final output = outputs.single as OrtValueTensor;
      expect(output.toHalfBits(), input.toHalfBits());
      expect(output.toFloat32List(), [1.5, -2.0]);
    } finally {
      for (final value in outputs) {
        value?.release();
      }
    }
  } finally {
    await session.closeAsync();
    input.release();
    options.release();
  }
}

void checkHalfRoundtrip() {
  for (final bfloat in [false, true]) {
    final exponentMask = bfloat ? 0x7f80 : 0x7c00;
    final mantissaMask = bfloat ? 0x7f : 0x3ff;
    final bits = Uint16List.fromList(List.generate(65536, (i) => i).where((i) =>
      (i & exponentMask) != exponentMask || (i & mantissaMask) == 0).toList());
    final decoded = decodeHalf(bits, bfloat: bfloat);
    final encoded = encodeHalf(decoded, bfloat: bfloat);
    for (var i = 0; i < bits.length; i++) {
      if (encoded[i] != bits[i]) fail('Half roundtrip mismatch: bfloat=$bfloat, bits=${bits[i]}, actual=${encoded[i]}');
    }
  }
}
