import ssl

from nltk import download

def init_nltk():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    for pkg in ['wordnet', 'omw-1.4']:
        download(pkg)

if __name__ == '__main__':
    init_nltk()