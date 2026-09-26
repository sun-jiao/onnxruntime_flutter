/// Failure of a strict asynchronous inference request.
class OrtInferenceException implements Exception {
  final String message;
  /// Native ORT error code, or null for Dart/Web errors without an ORT code.
  final int? code;
  /// Session worker/queue request identifier, when the request was accepted.
  final int? requestId;
  final String backend;
  /// Original worker/backend stack, preserved across the isolate boundary.
  final String? remoteStackTrace;

  const OrtInferenceException(this.message,
      {this.code, this.requestId, required this.backend, this.remoteStackTrace});

  @override
  String toString() => 'OrtInferenceException(backend=$backend, '
      'requestId=$requestId, code=$code): $message';
}
