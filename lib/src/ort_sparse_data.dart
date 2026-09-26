import 'dart:typed_data';
import 'ort_types.dart';

/// Independent CPU copies. Arrays may be modified without changing the tensor.
/// Half-precision values use Uint16List raw bits, identified by elementType.
class OrtSparseTensorData {
  final OrtSparseFormat format;
  final ONNXTensorElementDataType elementType;
  final List<int> shape;
  final List<int> valuesShape;
  final TypedData values;

  /// Keys: coo, inner/outer (CSR), or block. COO/CSR use Int64List;
  /// block indices use Int32List and their shape is in blockIndicesShape.
  final Map<String, TypedData> indices;
  final List<int>? blockIndicesShape;
  OrtSparseTensorData(
    this.format,
    this.elementType,
    List<int> shape,
    List<int> valuesShape,
    this.values,
    Map<String, TypedData> indices, {
    List<int>? blockIndicesShape,
  }) : shape = List.unmodifiable(shape),
       valuesShape = List.unmodifiable(valuesShape),
       indices = Map.unmodifiable(indices),
       blockIndicesShape =
           blockIndicesShape == null
               ? null
               : List.unmodifiable(blockIndicesShape);
}
