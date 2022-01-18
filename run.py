import os

import Annotation
import Video

def main():
    '''
    RUN CONFIGURATION:
    '''
    config = {
        'compress': False,
    }
    
    print(f'\nWorking Directory: {os.getcwd()}\n')
    print(f'Configurations:')
    for key, val in config.items():
        print(f'\t{key}: {val}')
    print()



    '''
    BODY:
    '''
    if config['compress']:
        _ = Video.VideoCompressor('./examples/videos', './examples/npy_videos')
        _ = Annotation.AnnotationsCompressor('./examples/alignments', './examples/npy_alignments')



    '''
    RESULTS:
    '''
    print('\nfinished.')


if __name__=='__main__':
    main()