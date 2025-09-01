import os
import pickle
import numpy as np
import pandas as pd
import random
import torch
import math
from torch.autograd import Variable
import logging

# ログ設定
logger = logging.getLogger(__name__)

class DataLoader():
    """
    データローダークラス。
    データセットの読み込み、前処理、バッチへの分割、訓練/検証セットへの分割を担当。
    """

    def __init__(self, data_dir, batch_size=5, seq_length=20, validation_size=0.2,
                 test_dataset_name=None, forcePreProcess=False, infer=False):
        '''
        DataLoaderの初期化
        :param data_dir: データセットが格納されているルートディレクトリ
        :param batch_size: バッチサイズ
        :param seq_length: シーケンス長
        :param validation_size: 検証データとして使用するデータの割合
        :param test_dataset_name: テスト用に除外するデータセットのファイル名に含まれる文字列
        :param forcePreProcess: Trueの場合、キャッシュがあっても強制的に前処理を実行
        :param infer: Trueの場合、テストモード (data_dirをテストデータセットとして使用)
        '''
        logger.info("🔧 DataLoader初期化開始")
        
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.infer = infer

        all_files = self.get_dataset_path(self.data_dir)
        if not all_files:
            logger.error(f"❌ データが見つかりません。'{self.data_dir}'にデータを配置してください。")
            raise FileNotFoundError(f"データがありません: {self.data_dir}")

        if not infer:
            # テスト用データセットを除外
            if test_dataset_name:
                train_val_files = [f for f in all_files if test_dataset_name not in os.path.basename(f)]
                self.test_dataset_files = [f for f in all_files if test_dataset_name in os.path.basename(f)]
            else:
                train_val_files = all_files
                self.test_dataset_files = []

            random.shuffle(train_val_files)
            split_index = math.ceil(len(train_val_files) * (1 - validation_size))
            self.train_dataset = train_val_files[:split_index]
            self.validation_dataset = train_val_files[split_index:]
            logger.info(f"📊 全{len(train_val_files)}ファイルを、訓練用({len(self.train_dataset)}個)と検証用({len(self.validation_dataset)}個)に分割しました。")
        else:
            self.test_dataset_files = all_files
            self.train_dataset = []
            self.validation_dataset = []

        self.cache_dir = os.path.join(self.data_dir, '.cache')
        self.data_file_tr = os.path.join(self.cache_dir, "trajectories_train.cpkl")
        self.data_file_vl = os.path.join(self.cache_dir, "trajectories_val.cpkl")
        self.data_file_te = os.path.join(self.cache_dir, "trajectories_test.cpkl")
        
        self.process_data(forcePreProcess)
        
        self.reset_batch_pointer(valid=False)
        self.reset_batch_pointer(valid=True)
        
        logger.info(f"✅ DataLoader初期化完了")

    def process_data(self, forcePreProcess):
        """データ前処理の実行"""
        if self.infer:
            if not os.path.exists(self.data_file_te) or forcePreProcess:
                self.frame_preprocess(self.test_dataset_files, self.data_file_te)
            self.load_preprocessed(self.data_file_te, is_validation=False) # テストデータをメインのdataにロード
        else:
            if self.train_dataset and (not os.path.exists(self.data_file_tr) or forcePreProcess):
                self.frame_preprocess(self.train_dataset, self.data_file_tr)

            if self.validation_dataset and (not os.path.exists(self.data_file_vl) or forcePreProcess):
                self.frame_preprocess(self.validation_dataset, self.data_file_vl)
            
            if os.path.exists(self.data_file_tr):
                self.load_preprocessed(self.data_file_tr, is_validation=False)
            if os.path.exists(self.data_file_vl):
                 self.load_preprocessed(self.data_file_vl, is_validation=True)

    def frame_preprocess(self, data_dirs, data_file):
        '''フレーム前処理'''
        logger.info(f"🔄 フレーム前処理開始: {len(data_dirs)}ファイル")
        
        all_frame_data, frameList_data, numPeds_data, pedsList_data, target_ids_data, orig_data = [], [], [], [], [], []
        
        for dataset_index, directory in enumerate(data_dirs):
            logger.info(f"📂 処理中: {directory}")
            try:
                df = pd.read_csv(directory, dtype={'frame_num': 'int', 'ped_id': 'int'},
                                 delimiter='\t', header=None, names=['frame_num', 'ped_id', 'x', 'y'])
                if df.empty: continue
            except Exception as e:
                logger.error(f"❌ データ読み込みエラー {directory}: {e}")
                continue

            data = df[['ped_id', 'x', 'y']].to_numpy()
            frame_ids = df['frame_num'].to_numpy()
            
            frames = np.unique(frame_ids)
            peds = np.unique(data[:, 0])
            
            all_frame_data.append([])
            frameList_data.append(frames)
            numPeds_data.append([])
            pedsList_data.append([])
            target_ids_data.append(peds)
            orig_data.append(df.to_numpy())
            
            for frame in frames:
                peds_in_frame_data = data[frame_ids == frame]
                all_frame_data[dataset_index].append(peds_in_frame_data)
                pedsList_data[dataset_index].append(peds_in_frame_data[:, 0].tolist())
                numPeds_data[dataset_index].append(peds_in_frame_data.shape[0])
                
        os.makedirs(os.path.dirname(data_file), exist_ok=True)
        with open(data_file, "wb") as f:
            pickle.dump((all_frame_data, frameList_data, numPeds_data, pedsList_data, target_ids_data, orig_data), f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"✅ 前処理完了: {data_file}")

    def load_preprocessed(self, data_file, is_validation=False):
        '''前処理済みデータをロード'''
        with open(data_file, 'rb') as f:
            raw_data = pickle.load(f)
        
        if is_validation:
            self.valid_data, self.valid_frameList, self.valid_numPedsList, self.valid_pedsList, self.valid_target_ids, self.valid_orig_data = raw_data
            counter = sum(max(0, len(frame_list) - self.seq_length) for frame_list in self.valid_data if frame_list)
            self.valid_num_batches = int(counter / self.batch_size)
        else:
            self.data, self.frameList, self.numPedsList, self.pedsList, self.target_ids, self.orig_data = raw_data
            counter = sum(max(0, len(frame_list) - self.seq_length) for frame_list in self.data if frame_list)
            self.num_batches = int(counter / self.batch_size)

    def next_batch(self): return self._get_batch_from_pointer(valid=False)
    def next_valid_batch(self): return self._get_batch_from_pointer(valid=True)

    def _get_batch_from_pointer(self, valid=False):
        '''ポインタからバッチデータを取得する共通ロジック'''
        x_batch, y_batch, d_batch, numPedsList_batch, PedsList_batch, target_ids_batch = [], [], [], [], [], []
        
        data_source = self.valid_data if valid else self.data
        peds_list_source = self.valid_pedsList if valid else self.pedsList
        num_peds_source = self.valid_numPedsList if valid else self.numPedsList
        
        current_dataset_pointer = self.valid_dataset_pointer if valid else self.dataset_pointer
        current_frame_pointer = self.valid_frame_pointer if valid else self.frame_pointer

        while len(x_batch) < self.batch_size:
            if current_dataset_pointer >= len(data_source): break 
            
            if current_frame_pointer + self.seq_length >= len(data_source[current_dataset_pointer]):
                current_dataset_pointer, current_frame_pointer = self._tick_batch_pointer(valid, current_dataset_pointer)
                if current_dataset_pointer == 0 and current_frame_pointer == 0: break 
                continue

            x_seq = data_source[current_dataset_pointer][current_frame_pointer : current_frame_pointer + self.seq_length]
            y_seq = data_source[current_dataset_pointer][current_frame_pointer + 1 : current_frame_pointer + self.seq_length + 1]
            peds_list_seq = peds_list_source[current_dataset_pointer][current_frame_pointer : current_frame_pointer + self.seq_length]
            num_peds_seq = num_peds_source[current_dataset_pointer][current_frame_pointer : current_frame_pointer + self.seq_length]
            
            possible_targets = peds_list_seq[0]
            if not possible_targets:
                current_frame_pointer += 1
                continue

            target_id = random.choice(possible_targets)
            x_batch.append(x_seq); y_batch.append(y_seq); d_batch.append(current_dataset_pointer)
            numPedsList_batch.append(num_peds_seq); PedsList_batch.append(peds_list_seq); target_ids_batch.append(target_id)
            
            current_frame_pointer += 1

        if valid:
            self.valid_dataset_pointer, self.valid_frame_pointer = current_dataset_pointer, current_frame_pointer
        else:
            self.dataset_pointer, self.frame_pointer = current_dataset_pointer, current_frame_pointer

        return x_batch, y_batch, d_batch, numPedsList_batch, PedsList_batch, target_ids_batch

    def _tick_batch_pointer(self, valid, current_pointer):
        current_pointer += 1
        if valid:
            if current_pointer >= len(self.valid_data): current_pointer = 0
        else:
            if current_pointer >= len(self.data): current_pointer = 0
        return current_pointer, 0
            
    def reset_batch_pointer(self, valid=False):
        if valid:
            self.valid_dataset_pointer = 0; self.valid_frame_pointer = 0
        else:
            self.dataset_pointer = 0; self.frame_pointer = 0
    
    def get_dataset_path(self, directory):
        dataset = []
        if os.path.exists(directory):
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.endswith('.txt'): dataset.append(os.path.join(root, file))
        return dataset


