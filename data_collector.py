# data_collector.py
import torch
import ccxt
import numpy as np
from sklearn.preprocessing import StandardScaler
from config import EXCHANGE, SYMBOL, ORDERBOOK_LIMIT, SEQUENCE_LENGTH
import logging


class DataCollector:
    def __init__(self):
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future'  # 使用期货市场
            }
        })
        self.symbol = 'BTC/USDT'  # Binance期货使用这种格式
        self.scaler = StandardScaler()

    def get_exchange(self):
        return self.exchange

    def fetch_orderbook(self):
        try:
            orderbook = self.exchange.fetch_order_book(self.symbol, ORDERBOOK_LIMIT)
            bids = orderbook['bids']
            asks = orderbook['asks']

            features = []
            for i in range(ORDERBOOK_LIMIT):
                if i < len(bids):
                    features.extend(bids[i])
                else:
                    features.extend([0, 0])
                if i < len(asks):
                    features.extend(asks[i])
                else:
                    features.extend([0, 0])

            return features
        except Exception as e:
            logging.error(f"获取订单簿数据时发生错误: {str(e)}")
            return None

    def calculate_additional_features(self, orderbook_data):
        features = []
        for data in orderbook_data:
            bid_prices = data[0::4][:ORDERBOOK_LIMIT]
            bid_volumes = data[1::4][:ORDERBOOK_LIMIT]
            ask_prices = data[2::4][:ORDERBOOK_LIMIT]
            ask_volumes = data[3::4][:ORDERBOOK_LIMIT]

            bid_ask_spread = ask_prices[0] - bid_prices[0]
            volume_imbalance = sum(bid_volumes) / (sum(bid_volumes) + sum(ask_volumes))
            bid_price_slope = (bid_prices[0] - bid_prices[-1]) / len(bid_prices) if len(bid_prices) > 1 else 0
            ask_price_slope = (ask_prices[-1] - ask_prices[0]) / len(ask_prices) if len(ask_prices) > 1 else 0
            bid_depth = sum(bid_volumes)
            ask_depth = sum(ask_volumes)
            mid_price = (bid_prices[0] + ask_prices[0]) / 2

            features.append(
                [bid_ask_spread, volume_imbalance, bid_price_slope, ask_price_slope, bid_depth, ask_depth, mid_price])

        return np.array(features)

    def prepare_data(self):
        data = []
        for _ in range(SEQUENCE_LENGTH):
            features = self.fetch_orderbook()
            if features is None:
                return None, None
            data.append(features)
        data = np.array(data)
        additional_features = self.calculate_additional_features(data)
        combined_data = np.hstack((data, additional_features))

        scaled_data = self.scaler.fit_transform(combined_data)
        return scaled_data, self.scaler

    def collect_train_data(self, n_samples, sequence_length):
        train_data = []
        for _ in range(n_samples):
            sample, _ = self.prepare_data()
            if sample is not None:
                train_data.append(sample)
        return train_data

    def collect_validation_data(self):
        data, _ = self.prepare_data()
        if data is None:
            return None, None
        inputs = torch.FloatTensor(data[:-1]).unsqueeze(0)
        targets = torch.FloatTensor([data[-1, 0] - data[-2, 0]])
        return inputs, targets

    def get_scaler(self):
        return self.scaler
