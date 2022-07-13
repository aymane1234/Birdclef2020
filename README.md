# Birdclef2020
This is the implementation of the BirdCLEF 2020 submission by.... within the ... project.
# 0-Preliminaries

"--samplerate",  "44100" := # samplerate

"--nfft",              "1024" := # FFt window length

"--hoplength",   "256" := # FFT hop-length

"--mels",            "128",    :=      # number of Mel Filters

"--fmax",           "15000",  :=    # Max Frequency

"--fmin",            "2000",    :=    # Min Frequency

"--mono",                     :=     # Convert to single channel audio

"--offset_sec",  "0",         :=    # offset from audiofile start (in seconds)

"--length_frames", "192" :=   # final length of extracted segment (in dimensions / number of frames)

final data shape or keras input shape = (128,192,1)
# 0.1- Setup
input:
audio (MelSpec)

Metadata (Numerically encode)

output: These should draw or guide the latent intermediate representations to identify higher concepts from which the specialized ones are derived.

1500 species (one-hot-encoded (OHE))

families (OHE)

orders (OHE)

If the confusion is within the same family, then the model has in principle identified the corrrect patterns, but only failed to identify the specific discriminating features for these species. But the error should not be considered equally impacting than a confusion with a different order

we can define a loss function which is agnostic of the taxonomy and weights the mis-classification based on a graph distance measure (e.g. max. hop length).
# 1-First experiment results

Quality is suboptimal. There are two important issues:

Track B is only silence (background noise).
Track B is correctly predicted and the lime explanation shows where the relevant information comes from

this shows us, that we have the problem that the audio features we not correctly extracted and the model tries to find patterns in the background noise.
We need to add an onset detector to the feature extraction to get only segments with bird sounds.

Solution: Segment the recording to extract only audio segments where birds are actually singing. Otherwise the input to the model will be silence (silent white noise) which is a huge problem.
One possible approach:

1-use the intermediate latent representation, the current model has learned

2-reduce its dimensions using PCA

3-use density based clustering to identify significant spectrogram patterns

4-use these clusters to aggregate a training set

5-train a bird-song detector

6-Using this birdsong/silence detector to extract only MelSpecs with birdsongs. This should dramatically improve the model "Better data, better output"

Remark: 
We use the taxonomy tree to predict the audio inputs according different granuliarities. We use one input to get three predictions (order, family, species). The outputs derive from different layers of the model. The hypothesis is, that by this granular guideance, the model learns more specific features. The feedback the model gets (the loss of the model) is not just right/wrong species, but more "wrong species, but right family and right order" so the CNN layers have only to adapt the species specific features.
