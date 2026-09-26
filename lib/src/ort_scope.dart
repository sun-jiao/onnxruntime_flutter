import 'dart:async';
import 'ort_session.dart' if (dart.library.js_interop) 'web/ort_web.dart';

/// Releases registered resources in reverse order, awaiting every disposer.
/// Register inputs/options before their session, so it drains before they free.
class OrtScope {
  final _disposers = <FutureOr<void> Function()>[];
  Future<void>? _closing;
  final _errors = <Object>[];
  List<Object> get disposalErrors => List.unmodifiable(_errors);

  T own<T>(T resource, FutureOr<void> Function(T) dispose) {
    defer(() => dispose(resource));
    return resource;
  }

  void defer(FutureOr<void> Function() dispose) {
    if (_closing != null) throw StateError('The scope is closing.');
    _disposers.add(dispose);
  }

  Future<void> closeAsync() {
    if (_closing != null) return _closing!;
    final done = Completer<void>();
    _closing = done.future;
    _drain().then(done.complete, onError: done.completeError);
    return _closing!;
  }

  Future<void> _drain() async {
    for (final dispose in _disposers.reversed) {
      try { await dispose(); } catch (error) { _errors.add(error); }
    }
    _disposers.clear();
    if (_errors.isNotEmpty) throw OrtScopeException(_errors);
  }
}

class OrtScopeException implements Exception {
  final List<Object> errors;
  OrtScopeException(List<Object> errors) : errors = List.unmodifiable(errors);
  @override
  String toString() => 'OrtScopeException: $errors';
}

/// On body failure, preserves that error after attempting all cleanup.
/// Cleanup failures remain available through scope.disposalErrors.
Future<T> usingOrtScope<T>(FutureOr<T> Function(OrtScope) body) async {
  final scope = OrtScope();
  late T result;
  try { result = await body(scope); }
  catch (error, stack) {
    try { await scope.closeAsync(); } catch (_) { /* Preserve body failure. */ }
    Error.throwWithStackTrace(error, stack);
  }
  await scope.closeAsync();
  return result;
}

/// Owns the session for this callback and awaits its shutdown on every exit.
Future<T> usingSession<T>(OrtSession session,
    FutureOr<T> Function(OrtSession) body) => usingOrtScope((scope) {
  scope.defer(session.closeAsync);
  return body(session);
});
