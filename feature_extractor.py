#!/usr/bin/env python
# coding: utf-8

# # Configuration

# In[ ]:
#Important: 
#For the pp.preprocess function to work we need to adjust our soundfile to how it was implemented. To achieve this the soundfiles should be:
#1-Turn mp3 to wavs then resample them to 22050 Hz
#2-They need to be mono "one channel"
#3-Sample width 16 bits
#This code might still need some adjusments, so that you integrate the split and do the feature extraction but i have investigate the pp.... function and it should work if one do these steps :)  
config = [
    
# General Processing Parameters
# =============================
    
# Path to trackid partition file.
'--tidfile', "./files_to_extract_HPC1.csv", 

# Path to directory to store intermediate features
"--dst", "/home/schindlera/sshfs/spark_master_schindlera/MSD/melspec_128_10seconds_2ch/",

"--workers",   "12", # Number of processes for feature extraction.
"--precision", "32", # Store features with 16bit or 32bit precision
    
"--crop",            # Crop longer audio files (comment/uncomment)
"--pad",             # Zero-pad shorter audio files (comment/uncomment)
"--skip",            # Skip if feature files already exist. (comment/uncomment)
#"--test"             # Development parameter. Only process one file (comment/uncomment)
    

# Audio Extraction (FFT, Mel Filter) Parameters
# =============================================
    
"--samplerate",  "22050",  # samplerate
"--nfft",        "1024",   # FFt window length
"--hoplength",   "256",    # FFT hop-length
"--mels",        "128",    # number of Mel Filters
#"--fmax",        "",       # Max Frequency
#"--fmin",        "0.0",    # Min Frequency
#"--mono",                  # Convert to single channel audio

"--length_sec",  "11",     # length of audio segment (in seconds)
"--offset_sec",  "3",      # offset from audiofile start (in seconds)
"--length_frames", "880"   # final length of extracted segment (in dimensions / number of frames)
    
]


# # Imports

# In[ ]:


from multiprocessing import Pool
import pandas as pd
import librosa
import audioread
import os
import sys
import numpy as np
from tqdm.auto import tqdm
import traceback
import warnings
import argparse
import logging
from bird import preprocessing as pp


# Parse Configuration

# In[ ]:


parser = argparse.ArgumentParser()

parser.add_argument('--tidfile',    help="Path to trackid partition file.",                  type=str)
parser.add_argument('--class_dir',    help="Path to class dir.",                             type=str)
parser.add_argument('--noise_dir',    help="Path to noise dir.",                             type=str)
parser.add_argument('--dst',        help="Path to directory to store intermediate features", type=str)
parser.add_argument('--workers',    help="Number of processes for feature extraction.",      type=int)
parser.add_argument('--crop',       help="Crop longer audio files",                          action='store_true')
parser.add_argument('--pad',        help="Zero-pad shorter audio files",                     action='store_true')
parser.add_argument('--skip',       help="Skip if feature files already exist.",             action='store_true')
parser.add_argument('--precision',  help="Store features with 16bit or 32bit precision",     type=int, default=32)
parser.add_argument("--log-level",  help="Configure the logging level.",                     default=logging.DEBUG, type=lambda x: getattr(logging, x))
parser.add_argument('--test',       help="Development parameter. Only process one file",     action='store_true')

parser.add_argument("--samplerate",    help="Audio Samplerate (for resampling)",             type=int, default=44100)
parser.add_argument("--nfft",          help="FFT window length",                             type=int, default=1024)
parser.add_argument("--hoplength",     help="FFT hop-length",                                type=int, default=512)
parser.add_argument("--mels",          help="number of Mel Filters",                         type=int, default=80)
parser.add_argument("--fmax",          help="Max Frequency",                                 type=int, default=None)
parser.add_argument("--fmin",          help="Min Frequency",                                 type=float, default=0.0)
parser.add_argument("--mono",          help="Convert to single channel audio",               action='store_true')
parser.add_argument("--length_sec",    help="length of audio segment (in seconds)",          type=int)
parser.add_argument("--offset_sec",    help="offset from audiofile start (in seconds)",      type=int)
parser.add_argument("--length_frames", help="final length of extracted segment (in dimensions / number of frames)", type=int)


if sys.argv[0].find("ipykernel_launcher") != -1:
    args = parser.parse_args(config)
else:
    args = parser.parse_args()
 


# Lib configuration

# In[ ]:


warnings.filterwarnings('ignore')


logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')
logging.basicConfig(level=args.log_level)


# # Methods

# In[ ]:


def extract_melspec(y, sample_rate):

    mel_spec = librosa.feature.melspectrogram(y          = y, 
                                              sr         = args.samplerate, 
                                              n_fft      = args.nfft, 
                                              hop_length = args.hoplength, 
                                              n_mels     = args.mels,
                                              fmin       = args.fmin,
                                              fmax       = args.fmax)

    mel_spec = librosa.core.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec



def extract(track_id, f_name):
    dst_fname = args.dst + "/" + track_id + ".npz"
    success   = False
    msg       = None
    
    if not (args.skip and os.path.exists(dst_fname)):

        try:
                                
            wave_data, sample_rate = librosa.core.load(f_name, 
                                                       sr       = args.samplerate, 
                                                       mono     = args.mono)
                                                       
                                                       
            if not args.mono and (len(wave_data.shape) != 2):
                wave_data = np.asarray([wave_data,wave_data])
                
            
            
            start  = args.offset_sec * sample_rate
            length = sample_rate * args.length_sec
            end    = start + length
            
            if args.mono:
            
                if (wave_data.shape[0] > end):
                    wave_data = wave_data[start:end]
                else:
                    wave_data = wave_data[:length]
                    
            else:
            
                if (wave_data.shape[1] > end):
                    wave_data = wave_data[:,start:end]
                else:
                    wave_data = wave_data[:,:length]
                

            if args.mono:
                
                mel_spec = extract_melspec(np.asfortranarray(wave_data), sample_rate) 
                
                if args.crop:
                    mel_spec = mel_spec[:,:args.length_frames]
                mel_spec = np.expand_dims(mel_spec, 2)
                                                
            else:
                
                mel_spec_ch1 = extract_melspec(np.asfortranarray(wave_data[0,:]), sample_rate)
                mel_spec_ch2 = extract_melspec(np.asfortranarray(wave_data[1,:]), sample_rate)
                
                if args.crop:
                    mel_spec_ch1 = mel_spec_ch1[:,:args.length_frames]
                    mel_spec_ch2 = mel_spec_ch2[:,:args.length_frames]
                
                mel_spec_ch1 = np.expand_dims(mel_spec_ch1, 2)
                mel_spec_ch2 = np.expand_dims(mel_spec_ch2, 2)
                
                mel_spec = np.concatenate([mel_spec_ch1, mel_spec_ch2], axis=2)

            
            if (args.pad) and (mel_spec.shape[1] < args.length_frames):
                                
                zeros = np.zeros((mel_spec.shape[0],SEG_DIM,mel_spec.shape[2]), dtype=np.float32)
                zeros[:mel_spec.shape[0], :mel_spec.shape[1], :mel_spec.shape[2]] = mel_spec
                    
                mel_spec = zeros            
            
            
            if args.precision == 16:
                mel_spec = mel_spec.astype(np.float32)
            elif args.precision == 16:
                mel_spec = mel_spec.astype(np.float16)
            
            np.savez(dst_fname, data=mel_spec)
            
            success = True
            
        except Exception as e:
            msg = e.msg
            
    else:
        # skip
        success   = True
        msg       = "skip"

    return track_id, success, msg


# # Run Feature Extraction

# In[ ]:


# read partition file
audiofile_metadata         = pd.read_csv(args.tidfile, header=None)
audiofile_metadata.columns = ["track_id", "filename"]
class_dir = args.class_dir
noise_dir = args.noise_dir

# In[ ]:


# create process pool
#pool = Pool(args.workers)

#results = []

if not args.test:
    pbar = tqdm(total=audiofile_metadata.shape[0])
else:
    pbar = tqdm(total=10)

#def update(*a):
#    pbar.update()
#    results.append(a[0])




for i in range(pbar.total):

   pp.preprocess_sound_file("./converted/audio_files2020/Sporophilaangolensis_Chestnut-belliedSeed-Finch/XC390533.wav", "./converted/class_dir/", "./converted/noise_dir/", 3)
#    pool.apply_async(extract, args=(audiofile_metadata.iloc[i].track_id, 
#                                   audiofile_metadata.iloc[i].filename,), callback=update)
    
#pool.close()
#pool.join()


# In[ ]:


#results = pd.DataFrame(results, columns=["trackid", "success", "error_msg"])
#results = results.set_index("trackid")
#results.to_csv(args.tidfile + ".melspec_extract.log.csv")

#print("Mel-Spectrograms sucessfully extracted : %d " % results[results.success].shape[0])
#print("Audio files failed to process          : %d " % (audiofile_metadata.shape[0] -results[results.success].shape[0]))

