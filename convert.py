import os
import soundfile
import sys
## This script does the proper steps to the audio files in order to run pp.preprocess properly
root = sys.argv[1]
for path, subdirs, files in os.walk(root):
    for name in files:
        if (name.endswith('.mp3')):
           a = path + "/" + name
           os.system("mkdir ./converted")
           os.system("mkdir ./converted" + path[1:])
           os.system("ffmpeg -i " + a + " -ar 22050 " "-ac 1 " + "./converted/"+ a[2:-3]  + "wav")                 
for path, subdirs, files in os.walk("./converted/"):
    for name in files:
        if (name.endswith('.wav')):           
           data, samplerate = soundfile.read(path + "/" + name)                
           soundfile.write(path + "/" + name, data, samplerate, subtype='PCM_16', )
           print(path + "/" + name + "converted")
sys.exit()