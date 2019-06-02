import sys
import os
import tarfile
import shutil
import numpy as np
from imageio import *


def file_create(path):
    if  not os.path.exists(path):
        os.mkdir(path)


url = 'http://vision.stanford.edu/aditya86/ImageNetDogs/'
last_percent_reported = None
data_root = '.'
num_classes = 120
image_size = 224
num_channels = 3
np.random.seed(133)

def download_progress_hook(count, blockSize, totalSize):
    """
    A hook to report the progress of a download. This is mostly intended for users with
    slow internet connections. Reports every 5% change in download progress.
    """
    global last_percent_reported
    percent = int(count * blockSize * 100 / totalSize)

    if last_percent_reported != percent:
        if percent % 5 == 0:
            sys.stdout.write("%s%%" % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write(".")
            sys.stdout.flush()

    last_percent_reported = percent

def maybe_download(filename, expected_bytes, force=False):
	dest_filename = os.path.join(data_root, filename)
	if force or not os.path.exists(dest_filename):
		print('Attempting to download:', filename)	
		filename, _ = urlretrieve(url + filename, dest_filename, reporthook=download_progress_hook)
		print('\nDownload Complete!')
	statinfo = os.stat(dest_filename)
	if statinfo.st_size == expected_bytes:
		print('Found and verified', dest_filename)
	else:
		raise Exception('Failed to verify ' + dest_filename + '. Can you get to it with a browser?')
	return dest_filename

def maybe_extract(filename, check_classes=True, force=False):
    root = os.path.splitext(filename)[0]  # remove .tar
    if os.path.isdir(root) and not force:
        print('%s already present - Skipping extraction of %s.' % (root, filename))
    else:
        print('Extracting data for %s. This may take a while. Please wait.' % root)
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall(data_root)
        tar.close()
    if check_classes:
        data_folders = [os.path.join(root, d) for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]
        if len(data_folders) != num_classes:
            raise Exception('Expected %d folders, one per class. Found %d instead.' % (num_classes, len(data_folders)))
        print('Completed extraction of %s.' % filename)
        return data_folders
    else:
        print('Completed extraction of %s.' % filename)

def move_data_files(image_list, new_folder):
    for file in image_list:
        if os.path.exists('Images/'+file[0][0]):
            shutil.move('Images/'+file[0][0],new_folder+'/'+file[0][0])
        elif not os.path.exists(new_folder+'/'+file[0][0]):
           print('%s does not exist, it may be missing' % os.path.exists('./images/'+file[0][0]))
    return [new_folder+'/'+d for d in sorted(os.listdir(new_folder)) if os.path.isdir(os.path.join(new_folder, d))]



def load_breed(folder):
    """
    Load the data for a single breed label.
    """
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size,num_channels), dtype=np.float32)
    print(folder)
    num_images = 0
    for image in image_files:
        image_file = folder+'/'+image
        try:
            
            image_data = misc.imread(image_file)
            
            annon_file = 'Annotation' + '/' + folder.split('/')[-1] + '/' + image.split('.')[0]
            annon_xml = minidom.parse(annon_file)
            xmin = int(annon_xml.getElementsByTagName('xmin')[0].firstChild.nodeValue)
            ymin = int(annon_xml.getElementsByTagName('ymin')[0].firstChild.nodeValue)
            xmax = int(annon_xml.getElementsByTagName('xmax')[0].firstChild.nodeValue)
            ymax = int(annon_xml.getElementsByTagName('ymax')[0].firstChild.nodeValue)
            
            new_image_data = image_data[ymin:ymax,xmin:xmax,:]
            new_image_data = misc.imresize(new_image_data, (image_size, image_size))
            misc.imsave('cropped/' + folder + '/' + image, new_image_data)
            dataset[num_images, :, :, :] = new_image_data
            num_images = num_images + 1
        except IOError as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

    dataset = dataset[0:num_images, :, :, :]

    print('Full dataset tensor:', dataset.shape)
    return dataset




def maybe_pickle(data_folders, force=False):
    dataset_names = []
    for folder in data_folders:
        set_filename = folder + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            print('%s already present - Skipping pickling.' % set_filename)
        else:
            print('Pickling %s.' % set_filename)
            dataset = load_breed(folder)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)
  
    return dataset_names


def make_arrays(nb_rows, img_size):
    if nb_rows:
        dataset = np.ndarray((nb_rows,img_size, img_size,num_channels), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size=0, even_size=True):
    num_classes = len(pickle_files)
    valid_dataset, valid_labels = make_arrays(valid_size, image_size)
    train_dataset, train_labels = make_arrays(train_size, image_size)
    vsize_per_class = valid_size // num_classes
    tsize_per_class = train_size // num_classes
    
    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, tsize_per_class
    end_l = vsize_per_class+tsize_per_class
    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, 'rb') as f:
                breed_set = pickle.load(f)
                np.random.shuffle(breed_set)
                
            if not even_size:
                tsize_per_class,end_l = len(breed_set),len(breed_set)
                end_t = start_t + tsize_per_class
                
            if valid_dataset is not None:
                valid_breed = breed_set[:vsize_per_class, :, :, :]
                valid_dataset[start_v:end_v, :, :, :] = valid_breed
                valid_labels[start_v:end_v] = label
                start_v += vsize_per_class
                end_v += vsize_per_class

            
            train_breed = breed_set[vsize_per_class:end_l, :, :, :]
            train_dataset[start_t:end_t, :, :, :] = train_breed
            train_labels[start_t:end_t] = label
            start_t += tsize_per_class
            end_t += tsize_per_class
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise
    
    return valid_dataset, valid_labels, train_dataset, train_labels



images_filename = maybe_download('images.tar', 793579520)
annotation_filename = maybe_download('annotation.tar', 21852160)
#lists_filename = maybe_download('lists.tar', 481280)
images_filename = 'images.tar'
annotation_filename = 'annotation.tar'
images_folders = maybe_extract(images_filename)
annotation_folders = maybe_extract(annotation_filename)