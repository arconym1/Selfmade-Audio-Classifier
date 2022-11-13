# We need to input Spectograms in our CNN.
import os

import librosa
import numpy
import skimage.io

path = "../PathToYourDataDir"


def scale_minmax(X, min=0.0, max=1.0):
    print("X : ", X)
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    print("X_std : ", X_std)
    print("X scaled : ", X_scaled)
    return X_scaled


def spectrogram_image(y, sr, out, hop_length, n_mels):
    # use log-melspectrogram
    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                          n_fft=hop_length * 2, hop_length=hop_length)  # y = window
    mels = numpy.log(mels + 1e-9)  # add small number to avoid log(0)
    print("Mels : ", mels)

    # min-max scale to fit inside 8-bit range
    img = scale_minmax(mels, 0, 255).astype(numpy.uint8)  # 8 byte
    print("Img : ", img)
    img = numpy.flip(img, axis=0)  # put low frequencies at the bottom in image
    img = 255 - img  # invert. make black==more energy
    print("Img = 255 - img : ", img)
    skimage.io.imsave(out, img)


def main():
    # settings
    hop_length = 512  # number of samples per time-step in spectrogram
    n_mels = 64  # number of bins (unterteilte frequenz) in spectrogram. Height of image
    time_steps = 64  # number of time-steps. Width of image
    n = 0
    for file in os.listdir(path):
        n += 1
        y, sr = librosa.load(path +
                             file, offset=1.0, duration=10.0, sr=22050)  # sr = 22050 hz
        out = f"Spectograms/Bund{n}.png"
        print("File : ", file)

        # extract a fixed length window
        start_sample = 0  # starting at beginning
        length_samples = time_steps * hop_length
        window = y[start_sample:start_sample + length_samples]  # scaled timeline / window
        print("Window : ", window)

        spectrogram_image(window, sr=sr, out=out, hop_length=hop_length, n_mels=n_mels)
        print('Wrote file : ', out)


if __name__ == '__main__':
    main()
