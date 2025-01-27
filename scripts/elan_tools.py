import pathlib as p
import sys

def get_wav_names(eaf):
    wav_names = []
    for descriptor in eaf.media_descriptors:
        if "audio" in descriptor["MIME_TYPE"]:
            url = None
            if "RELATIVE_MEDIA_URL" in descriptor:
                url = descriptor["RELATIVE_MEDIA_URL"]
            elif "MEDIA_URL" in descriptor:
                url = descriptor["MEDIA_URL"] # :(
            if url is not None:
                wav_names.append(p.Path(url).name)
    return wav_names

if __name__ == '__main__':
    globals()[sys.argv[1]]()