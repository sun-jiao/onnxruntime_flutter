import 'dart:async';
import 'dart:ffi';

import 'package:ffi/ffi.dart';

// Internal allocation boundary. A zone override allows deterministic allocation
// failure/leak tests without changing any exported API or the native runtime.
final Object nativeAllocatorZoneKey = Object();

Allocator get nativeAllocator =>
    Zone.current[nativeAllocatorZoneKey] as Allocator? ?? calloc;

R usingNative<R>(R Function(Arena arena) computation) =>
    using(computation, nativeAllocator);
