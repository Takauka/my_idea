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
        
        # 基本的な設定
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.infer = infer

        # --- MODIFIED: 指定されたdata_dirから全ファイルを取得 ---
        all_files = self.get_dataset_path(self.data_dir)
        if not all_files:
            logger.error(f"❌ データが見つかりません。'{self.data_dir}'にデータを配置してください。")
            raise FileNotFoundError(f"データがありません: {self.data_dir}")

        if not infer:
            # テスト用データセットを除外
            if test_dataset_name:
                train_val_files = [f for f in all_files if test_dataset_name not in f]
                self.test_dataset = [f for f in all_files if test_dataset_name in f]
            else:
                train_val_files = all_files
                self.test_dataset = []

            # データをシャッフルして訓練用と検証用に分割
            random.shuffle(train_val_files)
            split_index = math.ceil(len(train_val_files) * (1 - validation_size))
            self.train_dataset = train_val_files[:split_index]
            self.validation_dataset = train_val_files[split_index:]
            logger.info(f"📊 全{len(train_val_files)}ファイルを、訓練用({len(self.train_dataset)}個)と検証用({len(self.validation_dataset)}個)に分割しました。")
        else:
            # 推論モードでは、指定されたディレクトリの全ファイルをテスト用と見なす
            self.test_dataset = all_files
            self.train_dataset = []
            self.validation_dataset = []

        # 使用するデータセット決定
        self.data_dirs = self.test_dataset if infer else self.train_dataset
        
        # 前処理キャッシュファイルパス
        # キャッシュはデータディレクトリ内に作成
        self.cache_dir = os.path.join(self.data_dir, '.cache')
        self.data_file_tr = os.path.join(self.cache_dir, "trajectories_train.cpkl")
        self.data_file_vl = os.path.join(self.cache_dir, "trajectories_val.cpkl")
        self.data_file_te = os.path.join(self.cache_dir, "trajectories_test.cpkl")
        
        # データ前処理
        self.process_data(forcePreProcess)
        
        # ポインタ初期化
        self.reset_batch_pointer(valid=False)
        self.reset_batch_pointer(valid=True)
        
        logger.info(f"✅ DataLoader初期化完了")


    def process_data(self, forcePreProcess):
        """データ前処理の実行"""
        if self.infer:
            if not os.path.exists(self.data_file_te) or forcePreProcess:
                logger.info("📊 テストデータ前処理中...")
                self.frame_preprocess(self.test_dataset, self.data_file_te)
            self.load_preprocessed(self.data_file_te)
        else:
            if self.train_dataset and (not os.path.exists(self.data_file_tr) or forcePreProcess):
                logger.info("📊 訓練データ前処理中...")
                self.frame_preprocess(self.train_dataset, self.data_file_tr)

            if self.validation_dataset and (not os.path.exists(self.data_file_vl) or forcePreProcess):
                logger.info("📊 検証データ前処理中...")
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
                                 delimiter=' ', header=None, names=['frame_num', 'ped_id', 'y', 'x'])
                if df.empty:
                    logger.warning(f"⚠️ 空のデータファイル: {directory}")
                    continue
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
        logger.info(f"📖 前処理データ読み込み: {data_file}")
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"前処理ファイルが見つかりません: {data_file}")
            
        with open(data_file, 'rb') as f:
            raw_data = pickle.load(f)
        
        # --- FIX: is_validation=True の場合のアンパッキングを修正 ---
        if is_validation:
            self.valid_data, self.valid_frameList, self.valid_numPedsList, self.valid_pedsList, self.valid_target_ids, self.valid_orig_data = raw_data
            counter = sum(len(frame_list) - self.seq_length + 1 for frame_list in self.valid_data if frame_list)
            self.valid_num_batches = int(counter / self.batch_size)
        else:
            self.data, self.frameList, self.numPedsList, self.pedsList, self.target_ids, self.orig_data = raw_data
            counter = sum(len(frame_list) - self.seq_length + 1 for frame_list in self.data if frame_list)
            self.num_batches = int(counter / self.batch_size)


    def next_batch(self):
        '''次の訓練バッチを取得'''
        return self._get_batch_from_pointer(valid=False)

    def next_valid_batch(self):
        '''次の検証バッチを取得'''
        return self._get_batch_from_pointer(valid=True)

    def _get_batch_from_pointer(self, valid=False):
        '''ポインタからバッチデータを取得する共通ロジック'''
        x_batch, y_batch, d, numPedsList_batch, PedsList_batch, target_ids_batch = [], [], [], [], [], []
        
        data_source = self.valid_data if valid else self.data
        peds_list_source = self.valid_pedsList if valid else self.pedsList
        num_peds_source = self.valid_numPedsList if valid else self.numPedsList
        
        pointer = self.valid_dataset_pointer if valid else self.dataset_pointer
        frame_pointer = self.valid_frame_pointer if valid else self.frame_pointer

        for _ in range(self.batch_size):
            if pointer >= len(data_source) or frame_pointer + self.seq_length >= len(data_source[pointer]):
                self._tick_batch_pointer(valid)
                pointer = self.valid_dataset_pointer if valid else self.dataset_pointer
                frame_pointer = self.valid_frame_pointer if valid else self.frame_pointer
                
                if (_ > 0 and pointer == 0 and frame_pointer == 0) or (_ == 0):
                    break
                continue

            x_seq = data_source[pointer][frame_pointer : frame_pointer + self.seq_length]
            y_seq = data_source[pointer][frame_pointer + 1 : frame_pointer + self.seq_length + 1]
            peds_list_seq = peds_list_source[pointer][frame_pointer : frame_pointer + self.seq_length]
            num_peds_seq = num_peds_source[pointer][frame_pointer : frame_pointer + self.seq_length]
            
            possible_targets = [ped_id for ped_id in peds_list_seq[0]]
            target_id = random.choice(possible_targets) if possible_targets else 0

            x_batch.append(x_seq)
            y_batch.append(y_seq)
            d.append(pointer)
            numPedsList_batch.append(num_peds_seq)
            PedsList_batch.append(peds_list_seq)
            target_ids_batch.append(target_id)
            
            if valid: self.valid_frame_pointer += 1
            else: self.frame_pointer += 1

        return x_batch, y_batch, d, numPedsList_batch, PedsList_batch, target_ids_batch

    def _tick_batch_pointer(self, valid=False):
        '''データセットポインタとフレームポインタを更新'''
        if valid:
            self.valid_dataset_pointer += 1
            if self.valid_dataset_pointer >= len(self.valid_data):
                self.valid_dataset_pointer = 0
            self.valid_frame_pointer = 0
        else:
            self.dataset_pointer += 1
            if self.dataset_pointer >= len(self.data):
                self.dataset_pointer = 0
            self.frame_pointer = 0
            
    def reset_batch_pointer(self, valid=False):
        '''ポインタをリセット'''
        if valid:
            self.valid_dataset_pointer = 0
            self.valid_frame_pointer = 0
        else:
            self.dataset_pointer = 0
            self.frame_pointer = 0
    
    def get_dataset_path(self, directory):
        '''指定されたパス内の.txtファイルを取得'''
        dataset = []
        if os.path.exists(directory):
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.endswith('.txt'):
                        dataset.append(os.path.join(root, file))
        return dataset


