import tempfile
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import MidTermFeatures

def analyze_audio(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(file.read())
        filename = tmp.name

    [Fs, x] = audioBasicIO.read_audio_file(filename)
    F, f_names = MidTermFeatures.mid_feature_extraction(x, Fs, 1.0, 1.0)
    mean_features = F.mean(axis=1)

    return {
        "mean_pitch": round(float(mean_features[0]), 4),
        "energy": round(float(mean_features[1]), 4),
        "spectral_centroid": round(float(mean_features[2]), 4),
        "status": "Modelo de predição ainda em desenvolvimento (fase de extração de features)"
    }
