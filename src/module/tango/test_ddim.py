import IPython
import soundfile as sf
from tango import Tango

tango = Tango("declare-lab/tango2")
print(tango.model.__dict__)
import pdb

pdb.set_trace()


prompt = "An audience cheering and clapping"
audio = tango.generate(prompt)
sf.write(f"ddpm.wav", audio, samplerate=16000)
IPython.display.Audio(data=audio, rate=16000)
