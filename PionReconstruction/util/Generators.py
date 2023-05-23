import numpy as np
import uproot as ur
import gzip
import pickle
import awkward as ak
import random

import os
import time

from multiprocessing import Process, Queue, set_start_method

import tensorflow as tf

import matplotlib.pyplot as plt

class Generator:
    """ """
    def __init__(self, generator_name, save_type, file_list, cellGeo, batch_size, 
                 labeled=True, shuffle=True, num_procs=32, preprocess=False, output_dir=None, noisy=False):
        """ Initialization """
        self.name = generator_name
        self.save_type = save_type
        self.labeled = labeled
        self.preprocess = preprocess
        self.output_dir = output_dir
        self.noisy = noisy
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.file_list = file_list
        self.num_files = len(self.file_list)
        
        if self.shuffle: np.random.shuffle(self.file_list)
        
        self.num_procs = np.min([num_procs, self.num_files])
        self.procs = []
        if self.preprocess and self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
            if isinstance(cellGeo, str):
                with ur.open(cellGeo) as file:
                    self.geo_dict = loadGraphDictionary(file['CellGeo'])
            else:
                self.geo_dict = cellGeo
            self.preprocess_data()

    def preprocessor(self, worker_id):
        """ Abstract Method """
        pass  
    
    def preprocess_data(self):
        """ """
        print('\nPreprocessing and saving data to {}'.format(self.output_dir))
        for i in range(self.num_procs):
            p = Process(target=self.preprocessor, args=(i,), daemon=True)
            p.start()
            self.procs.append(p)
            
        for p in self.procs:
            p.join()
    
    def worker(self, worker_id, batch_queue):
        """ """
        batch_data = []
        batch_targets = []
        
        file_num = worker_id
        while file_num < self.num_files:
            with gzip.open(self.output_dir + f'{self.name}_{file_num:03d}.{self.save_type}', 'rb') as f:
                file_data = pickle.load(f)
            
            for i in range(len(file_data)):
                batch_data.append(file_data[i][0])
                batch_targets.append(file_data[i][1])
            
                if len(batch_data) == self.batch_size:
                    batch_targets = np.reshape(np.array(batch_targets), [-1, 3]).astype(np.float32)
                    
                    batch_queue.put((np.array(batch_data), batch_targets))
                    
                    batch_data = []
                    batch_targets = []
                    
            file_num += self.num_procs
        
        if len(batch_data) > 0:
            batch_targets = np.reshape(np.array(batch_targets), [-1,3]).astype(np.float32)
            batch_queue.put((np.array(batch_data), batch_targets))
    
    def check_procs(self):
        """ """
        for p in self.procs:
            if p.is_alive(): return True
        return False
        
    def kill_procs(self):
        """ """
        for p in self.procs:
            p.kill()
            
        self.procs = []
    
    def generator(self):
        """ Generator that returns processed batches during training """
        
        batch_queue = Queue(2 * self.num_procs)
        
        for i in range(self.num_procs):
            p = Process(target=self.worker, args=(i, batch_queue), daemon=True)
            p.start()
            self.procs.append(p)
            
        while self.check_procs() or not batch_queue.empty():
            try:
                batch = batch_queue.get(True, 0.0001)
            except:
                continue
                
            yield batch
            
        for p in self.procs:
            p.join()
            
class garnetDataGenerator(Generator):
    """ """
    def __init__(self, file_list, cellGeo_file, batch_size,
                labeled=False, shuffle=True, num_procs=32, preprocess=False, output_dir=None, noisy=False):
        """ """
        super().__init__('garnet', 'p', file_list, cellGeo_file, batch_size,
                         labeled, shuffle, num_procs, preprocess, output_dir, noisy)
        
    def preprocessor(self, worker_id):
        """ """
        file_num = worker_id
        while file_num < self.num_files:
            try: 
                if self.noisy:
                    print(f'Processing file {file_num}')
                if self.labeled:
                    file = ur.open(self.file_list[file_num][0])
                    label = self.file_list[file_num][1]
                else:
                    file = ur.open(self.file_list[file_num])
                tree = file['EventTree']

                preprocessed_data = []

                num_events = len(tree.arrays(tree.keys()[0]))
                event_data = tree.arrays(library='np')
                for event in range(num_events):
                    num_clusters = event_data['nCluster'][event]
                    for cluster in range(num_clusters):

                        cluster_E = event_data['cluster_E'][event][cluster]
                        target_E = event_data['cluster_ENG_CALIB_TOT'][event][cluster]

                        cluster_eta = event_data['cluster_Eta'][event][cluster]
                        cluster_phi = event_data['cluster_Phi'][event][cluster]

                        cell_e = event_data['cluster_cell_E'][event][cluster]

                        cellIDs = event_data['cluster_cell_ID'][event][cluster]
                        # Converting Cell IDs
                        cell_eta = np.vectorize(self.geo_dict['cell_geo_eta'].get)(np.nan_to_num(cellIDs))
                        cell_phi = np.vectorize(self.geo_dict['cell_geo_phi'].get)(np.nan_to_num(cellIDs))
                        cell_samp = np.vectorize(self.geo_dict['cell_geo_sampling'].get)(np.nan_to_num(cellIDs))
                        # Normalizing
                        cell_eta = cell_eta - cluster_eta
                        cell_phi = cell_phi - cluster_phi
                        with np.errstate(divide='ignore', invalid='ignore'):
                            cell_e = np.nan_to_num(np.log10(cell_e), nan=0.0, posinf=0.0, neginf=0.0)
                            target_E = np.nan_to_num(np.log10(target_E), nan=0.0, posinf=0.0, neginf=0.0)
                        # Clipping and Padding
                        PADLENGTH = 128
                        data = np.stack((cell_eta, cell_phi, cell_samp, cell_e), axis=-1)
                        data = np.pad(data[0:128], [(0, max(0, PADLENGTH-len(data))), (0, 0)], 'constant')
                        if not self.labeled:
                            label = np.round(event_data['cluster_EM_PROBABILITY'][event][cluster])
                        target = [label, int(not label), target_E]
                        if cluster_E > 0.5:
                            preprocessed_data.append((data, target))

                random.shuffle(preprocessed_data)

                # Saving
                with gzip.open(self.output_dir + f'{self.name}_{file_num:03d}.{self.save_type}', 'wb') as f:
                    pickle.dump(preprocessed_data, f)
                if self.noisy:
                    print(f'Finished processing file {file_num}')
            except:
                if self.labeled:
                    print(f'{self.file_list[file_num][0]} could not be opened')
                else:
                    print(f'{self.file_list[file_num]} could not be opened')
            finally:
                file_num += self.num_procs
                
                

# A quick implemention of Dilia's idea for converting the geoTree into a dict
def loadGraphDictionary(graphTree):
    globalDict = {}
    print("\nLoading Geo Dictionary...")

    arrays = graphTree.arrays()
    keys = graphTree.keys()
    for key in keys:
        if key not in ['cell_geo_sampling', 'cell_geo_eta', 'cell_geo_phi']: 
            continue
        branchDict = {}
        print(f'\tStarting on {key}')
        
        for iter, ID in enumerate(arrays['cell_geo_ID'][0]):
            branchDict[ID] = arrays[key][0][iter]

        if key == 'cell_geo_sampling':
            mask = 0
        else:
            mask = None

        branchDict[0] = mask
        branchDict[4308257264] = mask #Magic Number ID???
        
        globalDict[key] = branchDict
    print('Finished loading Geo Dictionary')
    return globalDict
    
        