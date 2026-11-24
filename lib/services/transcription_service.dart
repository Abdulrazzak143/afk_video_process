import 'dart:isolate';

import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

import 'model_manager.dart';

class TranscriptionService {
  /// Runs Whisper transcription off the UI thread.
  Future<String> transcribe({
    required String wavPath,
    required ModelPaths modelPaths,
  }) async {
    final encoder = modelPaths.encoder;
    final decoder = modelPaths.decoder;
    final tokens = modelPaths.tokens;

    return Isolate.run(() async {
      await sherpa_onnx.initBindingsAsync();
      return _transcribe(
        wavPath: wavPath,
        encoder: encoder,
        decoder: decoder,
        tokens: tokens,
      );
    });
  }

  static Future<String> _transcribe({
    required String wavPath,
    required String encoder,
    required String decoder,
    required String tokens,
  }) async {
    sherpa_onnx.OfflineRecognizer? recognizer;
    sherpa_onnx.OfflineStream? stream;

    try {
      final whisper = sherpa_onnx.OfflineWhisperModelConfig(
        encoder: encoder,
        decoder: decoder,
        language: '',
        task: 'transcribe',
        tailPaddings: 300,
      );

      final modelConfig = sherpa_onnx.OfflineModelConfig(
        whisper: whisper,
        tokens: tokens,
        modelType: 'whisper',
        numThreads: 2,
        debug: false,
      );

      final config = sherpa_onnx.OfflineRecognizerConfig(model: modelConfig);
      recognizer = sherpa_onnx.OfflineRecognizer(config);

      final wave = sherpa_onnx.readWave(wavPath);
      if (wave.sampleRate != 16000) {
        throw StateError(
          'Expected 16 kHz WAV input, got ${wave.sampleRate} Hz.',
        );
      }

      stream = recognizer.createStream();
      stream.acceptWaveform(
        samples: wave.samples,
        sampleRate: wave.sampleRate,
      );
      recognizer.decode(stream);

      final result = recognizer.getResult(stream);
      final text = result.text.trim();

      if (text.isEmpty) {
        return '(No speech detected in the audio track.)';
      }

      return text;
    } catch (e) {
      throw Exception('Transcription failed: $e');
    } finally {
      stream?.free();
      recognizer?.free();
    }
  }
}
