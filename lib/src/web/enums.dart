part of 'ort_web.dart';

enum OrtApiVersion {
  /// The initial release of the ORT API.
  api1(1),

  /// Post 1.0 builds of the ORT API.
  api2(2),

  /// Post 1.3 builds of the ORT API.
  api3(3),

  /// Post 1.6 builds of the ORT API.
  api7(7),

  /// Post 1.7 builds of the ORT API.
  api8(8),

  /// Post 1.10 builds of the ORT API.
  api11(11),

  /// Post 1.12 builds of the ORT API.
  api13(13),

  /// Post 1.13 builds of the ORT API.
  api14(14),

  /// The initial release of the ORT training API.
  trainingApi1(1);

  final int value;

  const OrtApiVersion(this.value);
}

/// An enumerated value of log's level.
enum OrtLoggingLevel {
  verbose(0),
  info(1),
  warning(2),
  error(3),
  fatal(4);

  final int value;

  const OrtLoggingLevel(this.value);
}

enum GraphOptimizationLevel {
  ortDisableAll(0),
  ortEnableBasic(1),
  ortEnableExtended(2),
  ortEnableAll(99);

  final int value;

  const GraphOptimizationLevel(this.value);
}
