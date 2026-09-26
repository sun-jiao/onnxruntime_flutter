import 'ort_types.dart';

/// Immutable model value description. Null shape means unknown rank; null
/// dimensions are dynamic. Non-tensor values expose their top-level kind only.
class OrtValueInfo {
  final String name;
  final ONNXType type;
  final ONNXTensorElementDataType? elementType;
  final List<int?>? shape;
  final List<String?>? symbolicDimensions;

  OrtValueInfo(
    this.name,
    this.type, {
    this.elementType,
    List<int?>? shape,
    List<String?>? symbolicDimensions,
  }) : shape = shape == null ? null : List.unmodifiable(shape),
       symbolicDimensions =
           symbolicDimensions == null
               ? null
               : List.unmodifiable(symbolicDimensions);
}

/// Checks required names, top-level kinds and tensor element types/shapes.
/// Nested sequence/map contents are left to the runtime. Symbolic dimensions
/// are descriptive and are not assumed to impose cross-input equality.
void validateOrtInputs(
  List<OrtValueInfo> expected,
  Map<String, OrtValueInfo> actual,
) {
  final names = expected.map((e) => e.name).toSet();
  for (final name in actual.keys) {
    if (!names.contains(name)) throw ArgumentError('Unknown input: $name');
  }
  for (final spec in expected) {
    final value = actual[spec.name];
    if (value == null) throw ArgumentError('Missing input: ${spec.name}');
    if (value.type != spec.type) {
      throw ArgumentError(
        'Input ${spec.name}: expected ${spec.type}, got ${value.type}',
      );
    }
    if (spec.type != ONNXType.tensor) continue;
    if (spec.elementType != null && spec.elementType != value.elementType) {
      throw ArgumentError(
        'Input ${spec.name}: expected ${spec.elementType}, got ${value.elementType}',
      );
    }
    final shape = spec.shape;
    if (shape == null) continue;
    final received = value.shape;
    if (received == null || shape.length != received.length) {
      throw ArgumentError(
        'Input ${spec.name}: expected rank ${shape.length}, got ${received?.length}',
      );
    }
    for (var i = 0; i < shape.length; i++) {
      if (shape[i] != null && shape[i] != received[i]) {
        throw ArgumentError(
          'Input ${spec.name} dimension $i: expected ${shape[i]}, got ${received[i]}',
        );
      }
    }
  }
}
