import 'dart:async';
import 'dart:collection';
import 'ort_session.dart' if (dart.library.js_interop) 'web/ort_web.dart';
import 'ort_value.dart' if (dart.library.js_interop) 'web/ort_web.dart';

enum OrtQueuePolicy { rejectNew, dropOldest, keepLatest }
enum OrtTaskStatus { completed, dropped, cancelled }

class OrtTaskResult<T> {
  final int id;
  final OrtTaskStatus status;
  final T? value;
  const OrtTaskResult(this.id, this.status, [this.value]);
}

/// Cancellation only succeeds while pending. Running native/JS work is drained.
class OrtTask<T> {
  final int id;
  final Future<OrtTaskResult<T>> result;
  final bool Function() _cancel;
  OrtTask._(this.id, this.result, this._cancel);
  bool cancel() => _cancel();
}

class _QueuedTask<T> {
  final int id;
  Future<T> Function()? body;
  final done = Completer<OrtTaskResult<T>>();
  _QueuedTask(this.id, this.body);
  void finish(OrtTaskStatus status) {
    body = null;
    done.complete(OrtTaskResult(id, status));
  }
}

/// A serial bounded queue. maxPending excludes the one active task.
/// keepLatest drops all older pending tasks on every new submission.
class OrtTaskQueue<T> {
  final int maxPending;
  final OrtQueuePolicy policy;
  final _pending = Queue<_QueuedTask<T>>();
  _QueuedTask<T>? _active;
  Completer<void>? _closed;
  var _nextId = 0;
  OrtTaskQueue({this.maxPending = 1, this.policy = OrtQueuePolicy.keepLatest}) {
    if (maxPending < 1) throw ArgumentError.value(maxPending, 'maxPending');
  }
  int get pendingCount => _pending.length;
  int get activeCount => _active == null ? 0 : 1;

  OrtTask<T> submit(Future<T> Function() body) {
    if (_closed != null) throw StateError('The queue is closing.');
    final task = _QueuedTask<T>(_nextId++, body);
    final ticket = OrtTask<T>._(task.id, task.done.future, () {
      if (!_pending.remove(task)) return false;
      task.finish(OrtTaskStatus.cancelled);
      _finishClose();
      return true;
    });
    if (policy == OrtQueuePolicy.keepLatest) {
      while (_pending.isNotEmpty) { _pending.removeFirst().finish(OrtTaskStatus.dropped); }
    } else if (_pending.length >= maxPending) {
      if (policy == OrtQueuePolicy.rejectNew) {
        task.finish(OrtTaskStatus.dropped);
        return ticket;
      }
      _pending.removeFirst().finish(OrtTaskStatus.dropped);
    }
    _pending.add(task);
    _pump();
    return ticket;
  }

  void _pump() {
    if (_active != null || _pending.isEmpty) { _finishClose(); return; }
    final task = _pending.removeFirst();
    _active = task;
    // Catch synchronous callback throws as well as asynchronous inference errors.
    Future<T>.sync(task.body!).then((value) {
      task.done.complete(OrtTaskResult(task.id, OrtTaskStatus.completed, value));
    }, onError: (Object error, StackTrace stack) {
      task.done.completeError(error, stack);
    }).whenComplete(() {
      task.body = null;
      _active = null;
      _pump();
    });
  }

  /// The first close determines whether to cancel pending tasks or drain them.
  /// The active task always finishes. No new tasks are accepted after this call.
  Future<void> closeAsync({bool cancelPending = false}) {
    if (_closed != null) return _closed!.future;
    _closed = Completer<void>();
    if (cancelPending) {
      while (_pending.isNotEmpty) { _pending.removeFirst().finish(OrtTaskStatus.cancelled); }
    }
    _finishClose();
    return _closed!.future;
  }

  void _finishClose() {
    if (_closed != null && !_closed!.isCompleted && _active == null && _pending.isEmpty) {
      _closed!.complete();
    }
  }
}

/// Real-time frame/chunk inference with explicit backpressure and cancellation.
/// Inputs/options remain caller-owned until each ticket settles (even if dropped).
/// Completed output values are owned by the consumer of that ticket's future.
class OrtRealtimeSession {
  final OrtSession session;
  final bool closeSession;
  final OrtTaskQueue<List<OrtValue?>> _queue;
  Future<void>? _closing;
  OrtRealtimeSession(this.session, {int maxPending = 1,
      OrtQueuePolicy policy = OrtQueuePolicy.keepLatest, this.closeSession = false})
      : _queue = OrtTaskQueue(maxPending: maxPending, policy: policy);
  int get pendingCount => _queue.pendingCount;
  int get activeCount => _queue.activeCount;
  OrtTask<List<OrtValue?>> enqueue(OrtRunOptions options,
      Map<String, OrtValue> inputs, [List<String>? outputNames]) {
    final feeds = Map<String, OrtValue>.of(inputs);
    final names = outputNames == null ? null : List<String>.of(outputNames);
    return _queue.submit(() => session.runAsyncOrThrow(options, feeds, names));
  }
  Future<void> closeAsync({bool cancelPending = false}) => _closing ??=
      _queue.closeAsync(cancelPending: cancelPending).then((_) async {
        if (closeSession) await session.closeAsync();
      });
}
