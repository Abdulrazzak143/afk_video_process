import 'dart:io';

import 'package:archive/archive.dart';
import 'package:dio/dio.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

enum ModelStatus {
  checking,
  downloading,
  extracting,
  ready,
  error,
}

class ModelPaths {
  const ModelPaths({
    required this.rootDir,
    required this.encoder,
    required this.decoder,
    required this.tokens,
  });

  final String rootDir;
  final String encoder;
  final String decoder;
  final String tokens;
}

typedef ModelProgressCallback = void Function(
  ModelStatus status,
  double progress,
  String message,
);

class ModelManager {
  ModelManager({Dio? dio}) : _dio = dio ?? Dio();

  static const modelDownloadUrl =
      'https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.tar.bz2';

  static const modelDirName = 'sherpa-onnx-whisper-tiny';
  static const encoderFileName = 'tiny-encoder.int8.onnx';
  static const decoderFileName = 'tiny-decoder.int8.onnx';
  static const tokensFileName = 'tiny-tokens.txt';

  final Dio _dio;

  Future<Directory> _modelsRoot() async {
    final docs = await getApplicationDocumentsDirectory();
    return Directory(p.join(docs.path, modelDirName));
  }

  Future<ModelPaths?> findExistingModel() async {
    final root = await _modelsRoot();
    if (!await root.exists()) {
      return null;
    }

    final encoder = await _findFile(root, encoderFileName);
    final decoder = await _findFile(root, decoderFileName);
    final tokens = await _findFile(root, tokensFileName);

    if (encoder == null || decoder == null || tokens == null) {
      return null;
    }

    return ModelPaths(
      rootDir: root.path,
      encoder: encoder,
      decoder: decoder,
      tokens: tokens,
    );
  }

  Future<String?> _findFile(Directory root, String fileName) async {
    await for (final entity in root.list(recursive: true, followLinks: false)) {
      if (entity is File && p.basename(entity.path) == fileName) {
        return entity.path;
      }
    }
    return null;
  }

  Future<ModelPaths> ensureModelReady({
    required ModelProgressCallback onProgress,
  }) async {
    onProgress(ModelStatus.checking, 0, 'Checking for Whisper model...');

    final existing = await findExistingModel();
    if (existing != null) {
      onProgress(ModelStatus.ready, 1, 'Whisper model is ready.');
      return existing;
    }

    final root = await _modelsRoot();
    await root.create(recursive: true);

    final archivePath = p.join(root.path, 'sherpa-onnx-whisper-tiny.tar.bz2');
    final archiveFile = File(archivePath);

    try {
      onProgress(ModelStatus.downloading, 0, 'Downloading Whisper model...');

      await _dio.download(
        modelDownloadUrl,
        archivePath,
        onReceiveProgress: (received, total) {
          if (total <= 0) {
            onProgress(
              ModelStatus.downloading,
              0,
              'Downloading Whisper model... ($received bytes)',
            );
            return;
          }
          final fraction = received / total;
          onProgress(
            ModelStatus.downloading,
            fraction,
            'Downloading Whisper model... ${(fraction * 100).toStringAsFixed(0)}%',
          );
        },
      );

      onProgress(ModelStatus.extracting, 0, 'Extracting model files...');
      await _extractTarBz2(archiveFile, root);

      if (await archiveFile.exists()) {
        await archiveFile.delete();
      }

      final paths = await findExistingModel();
      if (paths == null) {
        throw StateError(
          'Model extraction finished but required ONNX files were not found.',
        );
      }

      onProgress(ModelStatus.ready, 1, 'Whisper model is ready.');
      return paths;
    } catch (e) {
      onProgress(ModelStatus.error, 0, 'Model setup failed: $e');
      rethrow;
    }
  }

  Future<void> _extractTarBz2(File archiveFile, Directory destDir) async {
    final compressed = await archiveFile.readAsBytes();
    final decompressed = BZip2Decoder().decodeBytes(compressed);
    final archive = TarDecoder().decodeBytes(decompressed);

    for (final entry in archive) {
      final normalized = entry.name.replaceAll('\\', '/');
      if (normalized.endsWith('/')) {
        await Directory(p.join(destDir.path, normalized))
            .create(recursive: true);
        continue;
      }

      final outFile = File(p.join(destDir.path, normalized));
      await outFile.parent.create(recursive: true);
      await outFile.writeAsBytes(entry.content as List<int>);
    }
  }
}
