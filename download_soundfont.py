import os
import urllib.request
import tqdm

def download_soundfont():
    # Standard local path for the soundfont
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        
    target_path = os.path.join(models_dir, "soundfont.sf2")
    
    # URL to a reliable, lightweight GM soundfont (TimGM6mb.sf2 from GitHub)
    url = "https://raw.githubusercontent.com/bratpeki/soundfonts/master/TimGM6mb.sf2"
    
    print(f"Downloading soundfont to {target_path}...")
    
    try:
        class TqdmUpTo(tqdm.tqdm):
            def update_to(self, b=1, bsize=1, tsize=None):
                if tsize is not None:
                    self.total = tsize
                self.update(b * bsize - self.n)

        with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc="Soundfont") as t:
            urllib.request.urlretrieve(url, filename=target_path, reporthook=t.update_to)
        
        print("\nDownload complete!")
        return True
    except Exception as e:
        print(f"\nError downloading soundfont: {e}")
        return False

if __name__ == "__main__":
    download_soundfont()
