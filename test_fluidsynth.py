import time
import sys
import fluidsynth

# Default paths for soundfonts across different systems to test
SOUNDFONTS = [
    "/usr/share/soundfonts/freepats-general-midi.sf2", # Arch standard
    "/usr/share/sounds/sf2/FluidR3_GM.sf2", # Ubuntu standard
    "/Library/Audio/Sounds/Banks/FluidR3_GM.sf2" # Mac hypothetical
]

fs = fluidsynth.Synth()
fs.start(driver="pulseaudio") # Try pulse first, ALSA backup

# Load first available soundfont
sfid = -1
loaded_sf = None
for sf in SOUNDFONTS:
    import os
    if os.path.exists(sf):
        sfid = fs.sfload(sf)
        loaded_sf = sf
        break

if sfid == -1:
    print("No soundfont found. You need a .sf2 file to use fluidsynth.")
    sys.exit(1)
    
print(f"Loaded soundfont: {loaded_sf} with ID {sfid}")

# Select acoustic grand piano (bank 0, program 0)
fs.program_select(0, sfid, 0, 0)

print("Playing C chord for 2 seconds...")

# Play chord C E G (MIDI notes 60, 64, 67)
fs.noteon(0, 60, 100)
fs.noteon(0, 64, 100)
fs.noteon(0, 67, 100)

time.sleep(2.0)

# Stop notes
fs.noteoff(0, 60)
fs.noteoff(0, 64)
fs.noteoff(0, 67)

time.sleep(0.5)

fs.delete()
print("Test complete.")
