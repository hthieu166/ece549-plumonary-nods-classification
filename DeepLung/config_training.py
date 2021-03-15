config = {'train_data_path':['/home/hthieu/data/luna16/subsets/subset0/',
                             '/home/hthieu/data/luna16/subsets/subset1/',
                             '/home/hthieu/data/luna16/subsets/subset2/',
                             '/home/hthieu/data/luna16/subsets/subset3/',
                             '/home/hthieu/data/luna16/subsets/subset4/',
                             '/home/hthieu/data/luna16/subsets/subset5/',
                             '/home/hthieu/data/luna16/subsets/subset6/',
                             '/home/hthieu/data/luna16/subsets/subset7/',
                             '/home/hthieu/data/luna16/subsets/subset8/'],
          'val_data_path':['/home/hthieu/cig_luna16_experiments/luna16/subsets/subset9/'], 
          'test_data_path':['/home/hthieu/cig_luna16_experiments/luna16/subsets/subset9/'], 
          
          'train_preprocess_result_path':'$LUNA16PROPOCESSPATH/', # contains numpy for the data and label, which is generated by prepare.py
          'val_preprocess_result_path':'$LUNA16PROPOCESSPATH/',  # make sure copy all the numpy into one folder after prepare.py
          'test_preprocess_result_path':'$LUNA16PROPOCESSPATH/',
          
          'train_annos_path':'/home/hthieu/data/luna16/annotations.csv',
          'val_annos_path'  :'/home/hthieu/data/luna16/annotations.csv',
          'test_annos_path' :'/home/hthieu/data/luna16/annotations.csv',

          'black_list':[],
          
          'preprocessing_backend':'python',

          'luna_segment':'/home/hthieu/data/luna16/seg-lungs-LUNA16/', # download from https://luna16.grand-challenge.org/data/
          'preprocess_result_path':'/media/DATA/LUNA16/preprocess/',
          'luna_data':'/home/hthieu/data/luna16/subsets/',
          'luna_label':'/home/hthieu/data/luna16/annotations.csv'
         } 
