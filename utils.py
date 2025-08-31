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

    def __init__(self, f_prefix, batch_size=5, seq_length=20, validation_size=0.2,
                 forcePreProcess=False, infer=False):
        '''
        DataLoaderの初期化
        :param f_prefix: データフォルダへのプレフィックス
        :param batch_size: バッチサイズ
        :param seq_length: シーケンス長
        :param validation_size: 検証データとして使用するデータの割合 (0.0 to 1.0)
        :param forcePreProcess: Trueの場合、キャッシュがあっても強制的に前処理を実行
        :param infer: Trueの場合、テストモード (data/test/ を使用)
        '''
        logger.info("🔧 DataLoader初期化開始")
        
        # 基本的な設定
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.infer = infer

        # 基本パス設定
        self.base_train_path = 'data/train/'
        self.base_test_path = 'data/test/'
        
        # --- MODIFIED: データセットの取得と分割 ---
        if not infer:
            all_train_files = self.get_dataset_path(self.base_train_path, f_prefix)
            if not all_train_files:
                logger.error(f"❌ 訓練データが見つかりません。'{os.path.join(f_prefix, self.base_train_path)}'にデータを配置してください。")
                raise FileNotFoundError("訓練データがありません。")

            # データをシャッフルして訓練用と検証用に分割
            random.shuffle(all_train_files)
            split_index = math.ceil(len(all_train_files) * (1 - validation_size))
            self.train_dataset = all_train_files[:split_index]
            self.validation_dataset = all_train_files[split_index:]
            logger.info(f"📊 全{len(all_train_files)}データセットを、訓練用({len(self.train_dataset)}個)と検証用({len(self.validation_dataset)}個)に分割しました。")
        else:
            self.test_dataset = self.get_dataset_path(self.base_test_path, f_prefix)
            self.train_dataset = []
            self.validation_dataset = []

        # 使用するデータセット決定
        self.data_dirs = self.test_dataset if infer else self.train_dataset
        self.num_datasets = len(self.data_dirs)
        
        # データファイルパス
        self.train_data_dir = os.path.join(f_prefix, self.base_train_path)
        self.test_data_dir = os.path.join(f_prefix, self.base_test_path)
        self.val_data_dir = os.path.join(f_prefix, 'data/validation/') # 検証用キャッシュの保存場所
        
        # 前処理ファイルパス
        self.data_file_tr = os.path.join(self.train_data_dir, "trajectories_train.cpkl")
        self.data_file_te = os.path.join(self.test_data_dir, "trajectories_test.cpkl")
        self.data_file_vl = os.path.join(self.val_data_dir, "trajectories_val.cpkl")
        
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
            if not os.path.exists(self.data_file_tr) or forcePreProcess:
                logger.info("📊 訓練データ前処理中...")
                self.frame_preprocess(self.train_dataset, self.data_file_tr)

            if self.validation_dataset:
                if not os.path.exists(self.data_file_vl) or forcePreProcess:
                    logger.info("📊 検証データ前処理中...")
                    self.frame_preprocess(self.validation_dataset, self.data_file_vl)
            
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

            # NumPy配列に変換
            data = df[['ped_id', 'x', 'y']].to_numpy()
            frame_ids = df['frame_num'].to_numpy()
            
            # このデータセットの全フレームIDと歩行者ID
            frames = np.unique(frame_ids)
            peds = np.unique(data[:, 0])
            
            all_frame_data.append([])
            frameList_data.append(frames)
            numPeds_data.append([])
            pedsList_data.append([])
            target_ids_data.append(peds)
            orig_data.append(df.to_numpy())
            
            for frame in frames:
                # このフレームにいる歩行者のデータを抽出
                peds_in_frame_data = data[frame_ids == frame]
                
                # [ped_id, x, y] の形式で格納
                all_frame_data[dataset_index].append(peds_in_frame_data)
                pedsList_data[dataset_index].append(peds_in_frame_data[:, 0].tolist())
                numPeds_data[dataset_index].append(peds_in_frame_data.shape[0])
                
        # データ保存
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
        
        if is_validation:
            self.valid_data, self.valid_frameList, self.valid_numPedsList, self.valid_pedsList, self.valid_target_ids, _ = raw_data
            counter = sum(len(frame_list) - self.seq_length + 1 for frame_list in self.valid_data if frame_list)
            self.valid_num_batches = int(counter / self.batch_size)
        else:
            self.data, self.frameList, self.numPedsList, self.pedsList, self.target_ids, self.orig_data = raw_data
            counter = sum(len(frame_list) - self.seq_length + 1 for frame_list in self.data if frame_list)
            self.num_batches = int(counter / self.batch_size)

