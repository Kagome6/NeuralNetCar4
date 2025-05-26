# main_server_refactored.py
# マルチエージェント追跡・回避 RLシミュレート環境 (Python主導アーキテクチャ版)

import asyncio
import websockets
import json
import logging
import random
import os
import numpy as np
import math
import time
from pathlib import Path
from dataclasses import dataclass, field
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Set, Deque, Literal, Callable

# --- 環境設定 ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # TensorFlowのINFO/WARNINGログを抑制
import tensorflow as tf

# --- ロギング設定 ---
# DEBUGレベルで詳細情報を出力可能にする
logging.basicConfig(
    level=logging.INFO, # 通常はINFO、デバッグ時はDEBUGに変更
    format='%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s'
)
tf_logger = tf.get_logger()
tf_logger.setLevel(logging.ERROR) # TensorFlow の不要なログは抑制

# ==============================
# グローバル設定値
# ==============================

# --- サーバー設定 ---
HOST: str = "localhost"
PORT: int = 8765
SERVER_TICK_RATE: int = 60 # 1秒あたりの目標シミュレーション更新回数 (実行モード用)
SERVER_SLEEP_TIME: float = 1.0 / SERVER_TICK_RATE

# --- シミュレーション領域 ---
SIM_WIDTH: float = 800.0
SIM_HEIGHT: float = 600.0
MAX_FRAMES_PER_EPISODE: int = 3000 # エピソード最大フレーム数 (約50秒 @60fps)

# --- エージェント共通設定 ---
AGENT_SIZE: float = 15.0            # 描画サイズ目安
AGENT_SPEED: float = 1.8            # フレームあたりの移動距離
AGENT_COLLISION_RADIUS: float = AGENT_SIZE * 0.6 # 衝突判定半径
AGENT_TURN_RATE: float = 0.1        # フレームあたりの旋回角度 (ラジアン)
AgentType = Literal['pursuer', 'evader'] # エージェントタイプ定義

# --- Agent 0: Pursuer (追跡者) 設定 ---
PURSUER_ID: int = 0
PURSUER_START_POS: Tuple[float, float] = (SIM_WIDTH / 2, 50.0)
PURSUER_START_ANGLE: float = -math.pi / 2 # 北向き (-Y方向)
PURSUER_ACTION_SIZE: int = 6 # 0:L, 1:S, 2:R, 3:L+F, 4:S+F, 5:R+F
PURSUER_BULLET_COOLDOWN: int = 45 # フレーム数 (0.75秒 @60fps)
PURSUER_STATE_COMPONENTS: Dict[str, int] = {
    'obstacle_sensors': 7,
    'target_sensors': 3,
}
PURSUER_STATE_SIZE: int = sum(PURSUER_STATE_COMPONENTS.values())
PURSUER_SENSOR_ANGLES_OBSTACLE: np.ndarray = np.array([0, -math.pi/6, math.pi/6, -math.pi/3, math.pi/3, -math.pi*0.8, math.pi*0.8], dtype=np.float32)
PURSUER_SENSOR_RANGE_OBSTACLE: float = 130.0
PURSUER_SENSOR_ANGLES_TARGET: np.ndarray = np.array([0, -math.pi/8, math.pi/8], dtype=np.float32)
PURSUER_SENSOR_RANGE_TARGET: float = PURSUER_SENSOR_RANGE_OBSTACLE * 1.5

# --- Agent 1: Evader (回避者) 設定 ---
EVADER_ID: int = 1
EVADER_START_POS: Tuple[float, float] = (SIM_WIDTH / 2, SIM_HEIGHT- 50.0)
EVADER_START_ANGLE: float = -math.pi / 2 # 北向き (-Y方向)
EVADER_ACTION_SIZE: int = 3 # 0:L, 1:S, 2:R
EVADER_INITIAL_HEALTH: int = 10
EVADER_STATE_COMPONENTS: Dict[str, int] = {
    'obstacle_sensors': 7,
    'target_sensors': 8, # 全方位センサー
}
EVADER_STATE_SIZE: int = sum(EVADER_STATE_COMPONENTS.values())
EVADER_SENSOR_ANGLES_OBSTACLE: np.ndarray = PURSUER_SENSOR_ANGLES_OBSTACLE # 障害物センサーは共通
EVADER_SENSOR_RANGE_OBSTACLE: float = PURSUER_SENSOR_RANGE_OBSTACLE
# 全方位ターゲットセンサーの角度を計算
EVADER_SENSOR_ANGLES_TARGET: np.ndarray = np.linspace(
    0, 2 * math.pi, EVADER_STATE_COMPONENTS['target_sensors'], endpoint=False, dtype=np.float32
)
EVADER_SENSOR_RANGE_TARGET: float = PURSUER_SENSOR_RANGE_OBSTACLE * 1.5 # 範囲は追跡者と同じ

# --- 弾 設定 ---
BULLET_SPEED: float = 5.0
BULLET_RADIUS: float = 3.0
BULLET_LIFETIME: int = 90 # フレーム数 (1.5秒 @60fps)
BULLET_HIT_DAMAGE: int = 1 # 弾1発あたりのダメージ

# --- 障害物設定 ---
OBSTACLE_RADIUS: float = 7.0
INITIAL_OBSTACLES: int = 8     # 初期配置数
MAX_OBSTACLES: int = 30        # 最大存在数
OBSTACLE_GEN_PROB: float = 0.02  # 各フレームでの新規生成確率
OBSTACLE_MIN_DIST_AGENT_START: float = 100.0 # エージェント初期位置からの最小距離
OBSTACLE_MIN_DIST_OBSTACLE: float = OBSTACLE_RADIUS * 6 # 障害物同士の最小距離 (初期)
OBSTACLE_MIN_DIST_RUNTIME: float = OBSTACLE_RADIUS * 5 # 障害物同士の最小距離 (動的生成)
OBSTACLE_MIN_DIST_AGENT_RUNTIME: float = OBSTACLE_RADIUS + AGENT_COLLISION_RADIUS + 30 # 動的生成時のエージェントからの最小距離

# --- 強化学習 (DQN) 設定 ---
# 共通設定
MEMORY_SIZE: int = 100_000         # リプレイメモリサイズ
BATCH_SIZE: int = 64              # 学習時のバッチサイズ
GAMMA: float = 0.99               # 割引率
LEARNING_RATE: float = 0.001      # 学習率
TARGET_UPDATE_FREQ: int = 10      # Targetネットワーク更新頻度 (エピソード数)
MODEL_SAVE_FREQ: int = 200          # モデル保存頻度 (エピソード数)
REPLAY_START_SIZE: int = 1000     # このメモリサイズに達してから学習開始
REPLAY_FREQUENCY: int = 4         # フレームごとの学習頻度 (nフレームに1回学習)
# Epsilon Greedy 設定
EPSILON_START: float = 1.0        # 初期ε値
EPSILON_MIN: float = 0.05         # 最小ε値
EPSILON_DECAY_RATE: float = 0.9995 # ε減衰率 (エピソードごと)
# モデルファイルパス
MODEL_DIR: str = "models"
MODEL_PATH_A0: str = f"{MODEL_DIR}/rl_pursuer_model.weights.h5"
MODEL_PATH_A1: str = f"{MODEL_DIR}/rl_evader_model.weights.h5"
# DQNネットワーク構造
DQN_HIDDEN_UNITS: List[int] = [128, 128] # 隠れ層のユニット数

# --- 報酬設定 ---
# Agent 0 (Pursuer)
RWD_A0_SURVIVAL: float = 0.005            # 生存ボーナス
RWD_A0_FORWARD_MOVE: float = 0.3          # 前進（目標方向への移動）ボーナス係数
RWD_A0_ENEMY_PROXIMITY: float = 0.2       # 敵への近接ボーナス係数
RWD_A0_HIT_TARGET: float = 50.0           # 敵に弾を命中させたボーナス
RWD_A0_CATCH_TARGET: float = 20.0         # 敵に接触（捕獲）したボーナス
PNL_A0_OBSTACLE_PROXIMITY: float = -0.5   # 障害物への近接ペナルティ係数
PNL_A0_OBSTACLE_COLLISION: float = -25.0  # 障害物との衝突ペナルティ
PNL_A0_TURN: float = -0.05                # 旋回ペナルティ
PNL_A0_FIRE_ACTION: float = -0.1          # 発射アクションペナルティ
PROXIMITY_THRESHOLD_RATIO: float = 0.20   # 近接ペナルティ/ボーナスが発生するセンサー範囲に対する割合

# Agent 1 (Evader)
RWD_A1_SURVIVAL: float = 0.005
RWD_A1_FORWARD_MOVE: float = 0.5          # 前進（目標方向への移動）ボーナス係数 (追跡者から離れる方向)
RWD_A1_PURSUER_DISTANCE: float = 0.1      # 追跡者からの距離ボーナス係数
PNL_A1_HIT_BY_BULLET: float = -75.0       # 弾に被弾したペナルティ
PNL_A1_CAUGHT: float = -30.0              # 追跡者に接触（捕獲）されたペナルティ
PNL_A1_OBSTACLE_PROXIMITY: float = -0.5
PNL_A1_OBSTACLE_COLLISION: float = -25.0
PNL_A1_TURN: float = -0.05

# --- 学習/実行 ループ設定 ---
TOTAL_EPISODES: int = 120 # 学習総エピソード数 (十分な学習のために増やす)
# 実行モード時のPythonループスリープ時間（フレームレート制御用）
# 値を小さくするとCPU負荷は上がるが、よりリアルタイムに近くなる
RUNNING_MODE_SLEEP_TIME: float = 0.05 # 約 ms (200Hz相当、JSの描画が律速になる想定)

# ==============================
# ユーティリティ & Quadtree
# ==============================

# --- Geometry Helpers ---
def distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """2点間のユークリッド距離を計算"""
    return math.hypot(x2 - x1, y2 - y1)

def get_line_circle_intersection(
    x1: float, y1: float, x2: float, y2: float,
    cx: float, cy: float, r: float
) -> Optional[Tuple[float, float]]:
    """線分 (x1,y1)-(x2,y2) と円 (cx,cy,r) の最も近い交点を計算"""
    dx, dy = x2 - x1, y2 - y1
    fx, fy = x1 - cx, y1 - cy
    a = dx*dx + dy*dy
    if a < 1e-9: return None # 線分の長さがほぼゼロ
    b = 2*(fx*dx + fy*dy)
    c = fx*fx + fy*fy - r*r
    discriminant = b*b - 4*a*c
    if discriminant < 0: return None # 交点なし

    discriminant = math.sqrt(discriminant)
    t1 = (-b - discriminant) / (2*a)
    t2 = (-b + discriminant) / (2*a)

    # 線分上の交点のみを考慮 (0 <= t <= 1)
    valid_t = []
    if 0 <= t1 <= 1: valid_t.append(t1)
    if 0 <= t2 <= 1: valid_t.append(t2)

    if not valid_t: return None # 線分上には交点なし
    closest_t = min(valid_t) # 線分の始点に最も近い交点を選ぶ

    intersection_x = x1 + dx * closest_t
    intersection_y = y1 + dy * closest_t
    return (intersection_x, intersection_y)

# --- Quadtree 実装 ---
@dataclass(frozen=True)
class PointData:
    """Quadtreeに格納する点のデータ"""
    x: float
    y: float
    radius: float
    # dataフィールドには参照元のオブジェクト情報などを格納
    data: Dict[str, Any] = field(repr=False, hash=False, compare=False)

@dataclass(frozen=True)
class AABB:
    """Axis-Aligned Bounding Box (軸並行境界ボックス)"""
    center_x: float
    center_y: float
    half_width: float
    half_height: float

    @property
    def left(self) -> float: return self.center_x - self.half_width
    @property
    def right(self) -> float: return self.center_x + self.half_width
    @property
    def top(self) -> float: return self.center_y + self.half_height # Y軸上向きが正
    @property
    def bottom(self) -> float: return self.center_y - self.half_height

    def contains_point(self, point: PointData) -> bool:
        """点がAABB内に含まれるか (境界線上は含まない)"""
        return (self.left <= point.x < self.right and
                self.bottom <= point.y < self.top)

    def intersects_aabb(self, other: 'AABB') -> bool:
        """他のAABBと交差するか"""
        return not (other.left >= self.right or other.right <= self.left or
                    other.bottom >= self.top or other.top <= self.bottom)

    def intersects_circle(self, cx: float, cy: float, r: float) -> bool:
        """円と交差するか (SATもどき)"""
        closest_x = max(self.left, min(cx, self.right))
        closest_y = max(self.bottom, min(cy, self.top))
        dist_sq = (cx - closest_x)**2 + (cy - closest_y)**2
        return dist_sq <= r**2

class Quadtree:
    """衝突判定効率化のためのQuadtree"""
    def __init__(self, boundary: AABB, capacity: int = 4):
        self.boundary: AABB = boundary
        self.capacity: int = capacity # 1ノードあたりの最大要素数
        self.points: List[PointData] = []
        self.divided: bool = False
        # 子ノード (None または Quadtree インスタンス)
        self.northwest: Optional['Quadtree'] = None
        self.northeast: Optional['Quadtree'] = None
        self.southwest: Optional['Quadtree'] = None
        self.southeast: Optional['Quadtree'] = None

    def _subdivide(self):
        """現在のノードを4つの子ノードに分割する"""
        cx, cy = self.boundary.center_x, self.boundary.center_y
        hw, hh = self.boundary.half_width / 2, self.boundary.half_height / 2

        # Y軸上向きが正なので、NorthはYが大きい方
        nw_aabb = AABB(cx - hw, cy + hh, hw, hh)
        ne_aabb = AABB(cx + hw, cy + hh, hw, hh)
        sw_aabb = AABB(cx - hw, cy - hh, hw, hh)
        se_aabb = AABB(cx + hw, cy - hh, hw, hh)

        self.northwest = Quadtree(nw_aabb, self.capacity)
        self.northeast = Quadtree(ne_aabb, self.capacity)
        self.southwest = Quadtree(sw_aabb, self.capacity)
        self.southeast = Quadtree(se_aabb, self.capacity)
        self.divided = True

        # 分割前の点を適切な子ノードに再分配
        old_points = self.points
        self.points = []
        for point in old_points:
            # いずれかの子ノードに挿入を試みる
            if not (self.northwest.insert(point) or
                    self.northeast.insert(point) or
                    self.southwest.insert(point) or
                    self.southeast.insert(point)):
                # 稀に境界線上などの理由でどの子ノードにも入らない場合があるかもしれない
                # その場合は分割元のノードに残す (通常は発生しないはず)
                self.points.append(point)
                logging.warning(f"Point ({point.x}, {point.y}) could not be inserted into subnodes.")

    def insert(self, point: PointData) -> bool:
        """点をQuadtreeに挿入する"""
        # 点がこのノードの境界内にない場合は挿入しない
        if not self.boundary.contains_point(point):
            return False

        # 容量に余裕があり、まだ分割されていない場合 -> このノードに追加
        if len(self.points) < self.capacity and not self.divided:
            self.points.append(point)
            return True

        # 容量オーバーの場合、または既に分割済みの場合
        if not self.divided:
            self._subdivide()

        # 適切な子ノードに挿入を試みる
        if self.northwest.insert(point): return True
        if self.northeast.insert(point): return True
        if self.southwest.insert(point): return True
        if self.southeast.insert(point): return True

        # どのサブノードにも挿入できなかった場合 (通常は起こらないはず)
        logging.error(f"Quadtree insert failed for point ({point.x}, {point.y}). Boundary issue?")
        return False

    def query_range_circle(self, cx: float, cy: float, r: float) -> List[PointData]:
        """指定された円範囲内にある点を検索してリストで返す"""
        found_points: List[PointData] = []

        # 検索範囲の円がこのノードの境界と交差しない場合は探索不要
        if not self.boundary.intersects_circle(cx, cy, r):
            return found_points

        # このノード内の点をチェック
        for point in self.points:
            # 点と検索円の中心との距離が、半径の合計以下かチェック
            if distance(point.x, point.y, cx, cy) <= r + point.radius:
                found_points.append(point)

        # 分割されている場合は、子ノードも再帰的に検索
        if self.divided:
            found_points.extend(self.northwest.query_range_circle(cx, cy, r))
            found_points.extend(self.northeast.query_range_circle(cx, cy, r))
            found_points.extend(self.southwest.query_range_circle(cx, cy, r))
            found_points.extend(self.southeast.query_range_circle(cx, cy, r))

        return found_points

    def clear(self):
        """Quadtreeの内容をクリアする"""
        self.points = []
        self.divided = False
        self.northwest = None
        self.northeast = None
        self.southwest = None
        self.southeast = None

# ==============================
# PyAgent クラス
# ==============================
class PyAgent:
    """シミュレーション内のエージェントを表すクラス"""
    def __init__(self, agent_id: int, agent_type: AgentType, x: float, y: float, angle: float):
        self.id: int = agent_id
        self.type: AgentType = agent_type
        self.x: float = x
        self.y: float = y
        self.angle: float = angle # ラジアン (-pi ~ pi)
        self.speed: float = AGENT_SPEED
        self.size: float = AGENT_SIZE
        self.collision_radius: float = AGENT_COLLISION_RADIUS

        # 状態フラグ
        self.is_active: bool = True
        self.collided_obstacle_this_step: bool = False # このフレームで障害物に衝突したか
        self.collided_agent_this_step: bool = False    # このフレームで他エージェントに衝突したか
        self.hit_by_bullet_this_step: bool = False     # このフレームで弾に当たったか (Evader用)

        # センサー値 (np.ndarray)
        self.sensors_obstacle: Optional[np.ndarray] = None
        self.sensors_target: Optional[np.ndarray] = None

        # エージェントタイプ別の設定
        if self.type == 'pursuer':
            if self.id != PURSUER_ID: logging.warning("Pursuer ID mismatch")
            self.action_size: int = PURSUER_ACTION_SIZE
            self.state_size: int = PURSUER_STATE_SIZE
            self.num_obstacle_sensors: int = PURSUER_STATE_COMPONENTS['obstacle_sensors']
            self.num_target_sensors: int = PURSUER_STATE_COMPONENTS['target_sensors']
            self.sensor_angles_obstacle: np.ndarray = PURSUER_SENSOR_ANGLES_OBSTACLE
            self.sensor_range_obstacle: float = PURSUER_SENSOR_RANGE_OBSTACLE
            self.sensor_angles_target: np.ndarray = PURSUER_SENSOR_ANGLES_TARGET
            self.sensor_range_target: float = PURSUER_SENSOR_RANGE_TARGET
            self.bullet_cooldown: int = 0
            self.health: Optional[int] = None # Pursuerに体力は不要
        elif self.type == 'evader':
            if self.id != EVADER_ID: logging.warning("Evader ID mismatch")
            self.action_size: int = EVADER_ACTION_SIZE
            self.state_size: int = EVADER_STATE_SIZE
            self.num_obstacle_sensors: int = EVADER_STATE_COMPONENTS['obstacle_sensors']
            self.num_target_sensors: int = EVADER_STATE_COMPONENTS['target_sensors']
            self.sensor_angles_obstacle: np.ndarray = EVADER_SENSOR_ANGLES_OBSTACLE
            self.sensor_range_obstacle: float = EVADER_SENSOR_RANGE_OBSTACLE
            self.sensor_angles_target: np.ndarray = EVADER_SENSOR_ANGLES_TARGET
            self.sensor_range_target: float = EVADER_SENSOR_RANGE_TARGET
            self.bullet_cooldown: Optional[int] = None # Evaderは弾を撃たない
            self.health: int = EVADER_INITIAL_HEALTH
        else:
            raise ValueError(f"Unknown agent type: {self.type}")

        # センサー配列の初期化 (最初は最大レンジで埋める)
        self.sensors_obstacle = np.full(self.num_obstacle_sensors, self.sensor_range_obstacle, dtype=np.float32)
        self.sensors_target = np.full(self.num_target_sensors, self.sensor_range_target, dtype=np.float32)

        self.last_action_taken: Optional[int] = None # 最後に実行した行動 (デバッグ/描画用)
        self.last_position: Tuple[float, float] = (x, y) # 1フレーム前の位置 (移動量計算用)

    def reset_state(self, x: float, y: float, angle: float):
        """エージェントの状態を初期化 (リセット時)"""
        self.x = x
        self.y = y
        self.angle = angle
        self.is_active = True
        self.collided_obstacle_this_step = False
        self.collided_agent_this_step = False
        self.hit_by_bullet_this_step = False
        self.sensors_obstacle.fill(self.sensor_range_obstacle)
        self.sensors_target.fill(self.sensor_range_target)
        self.last_action_taken = None
        self.last_position = (x, y)
        if self.type == 'pursuer':
            self.bullet_cooldown = 0
        elif self.type == 'evader':
            self.health = EVADER_INITIAL_HEALTH

    def update_last_position(self):
        """現在の位置を保存 (次のフレームの移動量計算のため)"""
        self.last_position = (self.x, self.y)

    def move(self):
        """エージェントを現在の角度と速度で移動させる (トーラス空間)"""
        if not self.is_active: return

        delta_x: float = self.speed * math.cos(self.angle)
        delta_y: float = self.speed * math.sin(self.angle)

        self.x = (self.x + delta_x + SIM_WIDTH) % SIM_WIDTH
        self.y = (self.y + delta_y + SIM_HEIGHT) % SIM_HEIGHT

    def turn(self, action: int) -> bool:
        """指定されたアクションに基づいて旋回する。旋回した場合はTrueを返す"""
        if not self.is_active: return False

        is_turning = False
        turn_direction = 0 # -1: 左, 0: 直進, 1: 右

        if self.type == 'pursuer':
            # Pursuer Actions: 0:L, 1:S, 2:R, 3:L+F, 4:S+F, 5:R+F
            if action in [0, 3]: turn_direction = -1
            elif action in [2, 5]: turn_direction = 1
        elif self.type == 'evader':
            # Evader Actions: 0:L, 1:S, 2:R
            if action == 0: turn_direction = -1
            elif action == 2: turn_direction = 1

        if turn_direction != 0:
            self.angle += turn_direction * AGENT_TURN_RATE
            # 角度を -pi から pi の範囲に正規化
            self.angle = (self.angle + math.pi) % (2 * math.pi) - math.pi
            is_turning = True

        return is_turning

    def can_fire(self) -> bool:
        """Pursuerが弾を発射できる状態か"""
        return self.type == 'pursuer' and self.is_active and self.bullet_cooldown == 0

    def attempt_fire(self, action: int) -> bool:
        """指定されたアクションが発射を含むか判定し、可能ならクールダウンをリセット"""
        if not self.can_fire(): return False

        is_fire_action = (self.type == 'pursuer' and action in [3, 4, 5])
        if is_fire_action:
            self.bullet_cooldown = PURSUER_BULLET_COOLDOWN
            return True
        return False

    def decrement_cooldown(self):
        """Pursuerの弾クールダウンを1減らす"""
        if self.type == 'pursuer' and self.bullet_cooldown > 0:
            self.bullet_cooldown -= 1

    def take_damage(self, damage: int):
        """Evaderがダメージを受ける"""
        if self.type == 'evader' and self.is_active:
            self.health -= damage
            if self.health <= 0:
                self.health = 0
                self.is_active = False # 体力がなくなったら非アクティブ化
                logging.debug(f"Agent {self.id} (Evader) health depleted.")

    def update_sensors(self, world_quadtree: Quadtree, other_agent: 'PyAgent'):
        """センサー値を更新 (Quadtreeと他エージェントの位置を利用)"""
        if not self.is_active:
            # 非アクティブなエージェントのセンサー値は0とする
            self.sensors_obstacle.fill(0.0)
            self.sensors_target.fill(0.0)
            return

        start_x, start_y = self.x, self.y

        # --- 1. 障害物センサー ---
        sensor_angles_abs_obs = self.angle + self.sensor_angles_obstacle
        # 各センサー方向の終点座標 (最大レンジ)
        end_x_base_obs = start_x + self.sensor_range_obstacle * np.cos(sensor_angles_abs_obs)
        end_y_base_obs = start_y + self.sensor_range_obstacle * np.sin(sensor_angles_abs_obs)

        # Quadtreeを使って近傍の障害物候補を効率的に検索
        # 検索範囲はセンサーレンジ + 最大障害物半径 + 少しのマージン
        search_radius_obs = self.sensor_range_obstacle + OBSTACLE_RADIUS + 5.0
        potential_obstacles_data: List[PointData] = []
        # トーラス空間を考慮して9区画を検索
        for dx_world in [-1, 0, 1]:
            for dy_world in [-1, 0, 1]:
                offset_x = dx_world * SIM_WIDTH
                offset_y = dy_world * SIM_HEIGHT
                points_in_range = world_quadtree.query_range_circle(
                    start_x + offset_x, start_y + offset_y, search_radius_obs
                )
                # 障害物タイプのみを抽出
                potential_obstacles_data.extend([p for p in points_in_range if p.data.get('type') == 'obstacle'])

        # IDで重複を除去 (トーラス空間で同じ障害物が複数回ヒットするため)
        unique_obstacle_candidates = {p.data['id']: p for p in potential_obstacles_data}

        # 各障害物センサーについて、最も近い障害物までの距離を計算
        for i in range(self.num_obstacle_sensors):
            sensor_val = self.sensor_range_obstacle # 初期値は最大レンジ
            sx, sy = start_x, start_y
            ex, ey = end_x_base_obs[i], end_y_base_obs[i] # このセンサー方向の終点

            for obs_point_data in unique_obstacle_candidates.values():
                obs_data = obs_point_data.data
                obs_x_orig, obs_y_orig = obs_point_data.x, obs_point_data.y
                obs_r = obs_point_data.radius

                # トーラス空間を考慮して、各障害物の仮想的な位置との交差判定
                for dx_wrap in [-1, 0, 1]:
                    for dy_wrap in [-1, 0, 1]:
                         ox_virtual = obs_x_orig + dx_wrap * SIM_WIDTH
                         oy_virtual = obs_y_orig + dy_wrap * SIM_HEIGHT
                         # 線分と円の交差判定
                         intersection = get_line_circle_intersection(sx, sy, ex, ey, ox_virtual, oy_virtual, obs_r)
                         if intersection:
                             # 交点が見つかったら、始点からの距離を計算し、最小値を更新
                             dist_to_intersection = distance(sx, sy, intersection[0], intersection[1])
                             sensor_val = min(sensor_val, dist_to_intersection)

            self.sensors_obstacle[i] = sensor_val

        # --- 2. ターゲットセンサー ---
        if other_agent:
            sensor_angles_abs_target = self.angle + self.sensor_angles_target
            end_x_base_target = start_x + self.sensor_range_target * np.cos(sensor_angles_abs_target)
            end_y_base_target = start_y + self.sensor_range_target * np.sin(sensor_angles_abs_target)

            target_x_orig, target_y_orig = other_agent.x, other_agent.y
            target_r = other_agent.collision_radius

            for i in range(self.num_target_sensors):
                sensor_val = self.sensor_range_target # 初期値
                sx, sy = start_x, start_y
                ex, ey = end_x_base_target[i], end_y_base_target[i]

                # トーラス空間を考慮
                for dx_wrap in [-1, 0, 1]:
                    for dy_wrap in [-1, 0, 1]:
                        tx_virtual = target_x_orig + dx_wrap * SIM_WIDTH
                        ty_virtual = target_y_orig + dy_wrap * SIM_HEIGHT
                        intersection = get_line_circle_intersection(sx, sy, ex, ey, tx_virtual, ty_virtual, target_r)
                        if intersection:
                            dist_to_intersection = distance(sx, sy, intersection[0], intersection[1])
                            sensor_val = min(sensor_val, dist_to_intersection)

                self.sensors_target[i] = sensor_val
        else:
            # ターゲットがいない場合は最大レンジ
            self.sensors_target.fill(self.sensor_range_target)


    def get_state(self) -> np.ndarray:
        """現在の状態（正規化されたセンサー値の結合）を取得"""
        if not self.is_active:
            # 非アクティブな場合はゼロベクトルを返す
            return np.zeros(self.state_size, dtype=np.float32)

        # センサー値を 0.0 (遠い) ~ 1.0 (近い) に正規化
        # sensor_range が 0 の場合に備えて微小値を追加
        norm_sensors_obs = 1.0 - np.minimum(self.sensors_obstacle / (self.sensor_range_obstacle + 1e-6), 1.0)
        norm_sensors_target = 1.0 - np.minimum(self.sensors_target / (self.sensor_range_target + 1e-6), 1.0)

        # 状態ベクトルとして結合
        state = np.concatenate((norm_sensors_obs, norm_sensors_target))

        # 念のためサイズチェック
        if state.shape[0] != self.state_size:
            logging.error(f"Agent {self.id} state size mismatch! Expected {self.state_size}, Got {state.shape[0]}")
            # サイズが違う場合はゼロベクトルを返すなどエラー処理が必要
            return np.zeros(self.state_size, dtype=np.float32)

        return state.astype(np.float32)

    def get_render_data(self) -> Dict[str, Any]:
         """フロントエンド描画用のデータを返す"""
         data = {
            'id': self.id,
            'type': self.type,
            'x': self.x,
            'y': self.y,
            'angle': self.angle,
            'is_active': self.is_active,
            # このフレームでの衝突状態も送る (描画に反映させるため)
            'collided_obstacle': self.collided_obstacle_this_step,
            'collided_agent': self.collided_agent_this_step,
            'hit_by_bullet': self.hit_by_bullet_this_step if self.type == 'evader' else False,
            # 現在のクールダウンや体力
            'cooldown': self.bullet_cooldown if self.type == 'pursuer' else None,
            'health': self.health if self.type == 'evader' else None,
            # 生のセンサー値 (デバッグ表示用)
            'sensors_obstacle': self.sensors_obstacle.tolist(),
            'sensors_target': self.sensors_target.tolist(),
            # 最後に取った行動
            'last_action': self.last_action_taken,
         }
         return data

# ==============================
# SimulationEnv クラス
# ==============================
class SimulationEnv:
    """マルチエージェント追跡・回避のためのシミュレーション環境"""
    def __init__(self):
        self.agents: Dict[int, PyAgent] = {}
        self.obstacles: List[Dict[str, Any]] = []
        self.bullets: List[Dict[str, Any]] = []
        self.frame_count: int = 0

        # オブジェクトIDカウンター
        self._obstacle_id_counter: int = 0
        self._bullet_id_counter: int = 0

        # Quadtree設定
        self.boundary = AABB(center_x=SIM_WIDTH / 2, center_y=SIM_HEIGHT / 2,
                             half_width=SIM_WIDTH / 2, half_height=SIM_HEIGHT / 2)
        self.world_quadtree = Quadtree(self.boundary, capacity=4)

        logging.info("SimulationEnv initialized.")

    def _get_next_id(self, counter_name: str) -> int:
        """内部カウンターをインクリメントして新しいIDを返す"""
        if counter_name == 'obstacle':
            self._obstacle_id_counter += 1
            return self._obstacle_id_counter
        elif counter_name == 'bullet':
            self._bullet_id_counter += 1
            return self._bullet_id_counter
        else:
            raise ValueError(f"Unknown counter name: {counter_name}")

    def reset(self) -> Dict[int, np.ndarray]:
        """環境を初期状態にリセットし、各エージェントの初期状態を返す"""
        self.frame_count = 0
        self._obstacle_id_counter = 0
        self._bullet_id_counter = 0
        self.bullets = []
        self.agents = {} # エージェント辞書をクリア

        # エージェントの生成と初期化
        agent0 = PyAgent(agent_id=PURSUER_ID, agent_type='pursuer',
                         x=PURSUER_START_POS[0], y=PURSUER_START_POS[1], angle=PURSUER_START_ANGLE)
        agent1 = PyAgent(agent_id=EVADER_ID, agent_type='evader',
                         x=EVADER_START_POS[0], y=EVADER_START_POS[1], angle=EVADER_START_ANGLE)
        self.agents = {PURSUER_ID: agent0, EVADER_ID: agent1}

        # 障害物の初期生成
        self.obstacles = self._generate_initial_obstacles()

        # Quadtreeの更新
        self._update_quadtree()

        # エージェントのセンサー初期化
        for agent in self.agents.values():
            other_agent = self.agents[1 - agent.id] # 相手エージェントを取得
            agent.update_sensors(self.world_quadtree, other_agent)

        logging.debug(f"Environment reset. Initial obstacles: {len(self.obstacles)}")
        # 各エージェントの初期状態を返す
        initial_states = {agent_id: agent.get_state() for agent_id, agent in self.agents.items()}
        return initial_states

    def _generate_initial_obstacles(self) -> List[Dict[str, Any]]:
        """初期配置用の障害物を生成する"""
        obs_list: List[Dict[str, Any]] = []
        attempts = 0
        max_attempts = INITIAL_OBSTACLES * 20 # 試行回数を増やす

        while len(obs_list) < INITIAL_OBSTACLES and attempts < max_attempts:
            radius = OBSTACLE_RADIUS
            # 障害物が壁際に寄りすぎないようにパディングを設定
            padding = radius + 10
            x = random.uniform(padding, SIM_WIDTH - padding)
            y = random.uniform(padding, SIM_HEIGHT - padding)

            # 1. エージェントの初期位置に近すぎないかチェック
            too_close_to_start = False
            for agent in self.agents.values():
                if distance(x, y, agent.x, agent.y) < OBSTACLE_MIN_DIST_AGENT_START:
                    too_close_to_start = True; break
            if too_close_to_start:
                attempts += 1; continue

            # 2. 他の既に配置された障害物に近すぎないかチェック
            valid_pos = True
            for existing_obs in obs_list:
                if distance(x, y, existing_obs['x'], existing_obs['y']) < OBSTACLE_MIN_DIST_OBSTACLE:
                    valid_pos = False; break
            if not valid_pos:
                attempts += 1; continue

            # 有効な位置なら障害物を追加
            obs_id = self._get_next_id('obstacle')
            obs_list.append({
                'id': obs_id,
                'type': 'obstacle',
                'x': x, 'y': y,
                'radius': radius
            })
            attempts += 1 # 配置成功でも試行回数は増やす

        if len(obs_list) < INITIAL_OBSTACLES:
            logging.warning(f"Could only generate {len(obs_list)} initial obstacles after {max_attempts} attempts.")
        return obs_list

    def _generate_obstacle_runtime(self):
        """シミュレーション実行中に確率で障害物を生成する"""
        if len(self.obstacles) >= MAX_OBSTACLES or random.random() >= OBSTACLE_GEN_PROB:
            return

        radius = OBSTACLE_RADIUS
        padding = radius + 5
        valid_pos = False
        attempts = 0
        max_runtime_attempts = 15
        x, y = 0.0, 0.0

        while not valid_pos and attempts < max_runtime_attempts:
            x = random.uniform(padding, SIM_WIDTH - padding)
            y = random.uniform(padding, SIM_HEIGHT - padding)
            valid_pos = True

            # 1. 既存の障害物との距離チェック
            for obs in self.obstacles:
                if distance(x, y, obs['x'], obs['y']) < OBSTACLE_MIN_DIST_RUNTIME:
                    valid_pos = False; break
            if not valid_pos: attempts += 1; continue

            # 2. アクティブなエージェントとの距離チェック
            for agent in self.agents.values():
                if agent.is_active and distance(x, y, agent.x, agent.y) < OBSTACLE_MIN_DIST_AGENT_RUNTIME:
                    valid_pos = False; break
            if not valid_pos: attempts += 1; continue

            # 有効な位置が見つかった
            attempts += 1 # ループ終了条件のため

        if valid_pos:
            obs_id = self._get_next_id('obstacle')
            new_obs = {'id': obs_id, 'type': 'obstacle', 'x': x, 'y': y, 'radius': radius}
            self.obstacles.append(new_obs)
            # Quadtreeにも追加 (次のフレームの_update_quadtreeで再構築されるので必須ではないが、念のため)
            # self.world_quadtree.insert(PointData(x, y, radius, new_obs))
            logging.debug(f"Generated runtime obstacle {obs_id} at ({x:.1f}, {y:.1f})")

    def _update_quadtree(self):
        """現在のすべてのオブジェクトでQuadtreeを再構築する"""
        self.world_quadtree.clear()
        # 障害物を挿入
        for obs in self.obstacles:
            self.world_quadtree.insert(PointData(obs['x'], obs['y'], obs['radius'], obs))
        # エージェントを挿入
        for agent in self.agents.values():
            # アクティブでないエージェントも判定対象になる可能性があるので、挿入はする
            agent_data = {'type': 'agent', 'id': agent.id, 'ref': agent} # エージェント本体への参照を含める
            self.world_quadtree.insert(PointData(agent.x, agent.y, agent.collision_radius, agent_data))
        # 弾を挿入
        for bullet in self.bullets:
            bullet_data = {'type': 'bullet', 'id': bullet['id'], 'owner_id': bullet['owner_id'], 'ref': bullet}
            self.world_quadtree.insert(PointData(bullet['x'], bullet['y'], bullet['radius'], bullet_data))

    def _move_bullets(self):
        """すべての弾を移動させ、寿命切れや画面外の弾を除去する"""
        active_bullets: List[Dict[str, Any]] = []
        for bullet in self.bullets:
            bullet['lifetime'] -= 1
            if bullet['lifetime'] <= 0:
                continue # 寿命切れ

            dx = BULLET_SPEED * math.cos(bullet['angle'])
            dy = BULLET_SPEED * math.sin(bullet['angle'])
            bullet['x'] += dx
            bullet['y'] += dy

            # 画面外に出た弾も除去 (トーラスではない)
            if not (0 <= bullet['x'] < SIM_WIDTH and 0 <= bullet['y'] < SIM_HEIGHT):
                continue

            active_bullets.append(bullet)
        self.bullets = active_bullets

    def _check_collisions(self) -> Tuple[Set[int], Set[Tuple[int, int]], Set[int]]:
        """
        衝突判定を実行し、衝突情報を返す。
        エージェントの衝突フラグとEvaderの体力を更新する。
        Returns:
            Tuple[Set[int], Set[Tuple[int, int]], Set[int]]:
            (障害物に衝突したエージェントIDセット, エージェント間衝突ペアセット, 命中して消滅する弾IDセット)
        """
        collided_obstacle_agents: Set[int] = set()
        collided_agent_pairs: Set[Tuple[int, int]] = set() # (小さいID, 大きいID)
        hit_bullets_to_remove: Set[int] = set()

        # 各エージェントについて衝突判定
        for agent_id, agent in self.agents.items():
            # 衝突フラグをリセット
            agent.collided_obstacle_this_step = False
            agent.collided_agent_this_step = False
            agent.hit_by_bullet_this_step = False

            # 非アクティブなエージェントは衝突を起こさないが、衝突される可能性はあるので判定は行う
            # if not agent.is_active: continue

            agent_x, agent_y = agent.x, agent.y
            agent_r = agent.collision_radius

            # Quadtreeを使って近傍のオブジェクト候補を検索
            # 検索半径は自エージェント半径 + 起こりうる最大の相手半径 + マージン
            max_other_radius = max(OBSTACLE_RADIUS, AGENT_COLLISION_RADIUS, BULLET_RADIUS)
            search_radius = agent_r + max_other_radius + 5.0

            candidate_points: List[PointData] = []
            # トーラス空間を考慮 (9区画検索)
            for dx_world in [-1, 0, 1]:
                for dy_world in [-1, 0, 1]:
                    offset_x = dx_world * SIM_WIDTH
                    offset_y = dy_world * SIM_HEIGHT
                    points_in_range = self.world_quadtree.query_range_circle(
                        agent_x + offset_x, agent_y + offset_y, search_radius
                    )
                    # 自分自身以外のオブジェクトを候補に追加
                    candidate_points.extend([p for p in points_in_range if p.data.get('id') != agent_id or p.data.get('type') != 'agent'])

            # (タイプ, ID) のタプルをキーにして重複を除去
            unique_candidates: Dict[Tuple[str, int], PointData] = {}
            for p in candidate_points:
                key = (p.data.get('type', 'unknown'), p.data.get('id', -1))
                if key not in unique_candidates: unique_candidates[key] = p

            # 各候補オブジェクトとの衝突判定
            for point_data in unique_candidates.values():
                obj_data = point_data.data
                obj_type = obj_data.get('type')
                obj_id = obj_data.get('id')
                obj_r = point_data.radius
                obj_x_orig, obj_y_orig = point_data.x, point_data.y

                # トーラス空間での距離判定
                collided_this_obj = False
                for dx_wrap in [-1, 0, 1]:
                    for dy_wrap in [-1, 0, 1]:
                        ox_virtual = obj_x_orig + dx_wrap * SIM_WIDTH
                        oy_virtual = obj_y_orig + dy_wrap * SIM_HEIGHT
                        dist = distance(agent_x, agent_y, ox_virtual, oy_virtual)
                        if dist < agent_r + obj_r:
                            collided_this_obj = True
                            break
                    if collided_this_obj: break

                if collided_this_obj:
                    # --- 衝突タイプに応じた処理 ---
                    if obj_type == 'obstacle':
                        # アクティブなエージェントが障害物に衝突した場合のみ記録
                        if agent.is_active:
                            agent.collided_obstacle_this_step = True
                            agent.is_active = False # 障害物衝突で即非アクティブ化
                            collided_obstacle_agents.add(agent_id)
                            logging.debug(f"Agent {agent_id} collided with obstacle {obj_id}. Deactivated.")

                    elif obj_type == 'agent':
                        other_agent_id = obj_id
                        other_agent = self.agents.get(other_agent_id)
                        # 両エージェントがアクティブな場合のみ衝突とする
                        if agent.is_active and other_agent and other_agent.is_active:
                            agent.collided_agent_this_step = True
                            other_agent.collided_agent_this_step = True
                            agent.is_active = False # エージェント衝突で両者非アクティブ化
                            other_agent.is_active = False
                            # ペアをソートして記録 (小さいID, 大きいID)
                            pair = tuple(sorted((agent_id, other_agent_id)))
                            collided_agent_pairs.add(pair)
                            logging.debug(f"Agent {agent_id} and Agent {other_agent_id} collided. Both deactivated.")

                    elif obj_type == 'bullet':
                        # Evader が自分以外のエージェントが発射した弾に当たった場合
                        if agent.type == 'evader' and agent.is_active and obj_data.get('owner_id') != agent_id:
                            agent.hit_by_bullet_this_step = True
                            agent.take_damage(BULLET_HIT_DAMAGE) # ダメージ処理 (内部で非アクティブ化も判定)
                            hit_bullets_to_remove.add(obj_id) # 命中した弾は除去対象
                            logging.debug(f"Agent {agent_id} (Evader) hit by bullet {obj_id}. Health: {agent.health}")
                            # take_damageの結果、非アクティブになった場合、is_activeフラグは更新されている

        # 衝突した弾をリストから除去
        if hit_bullets_to_remove:
            self.bullets = [b for b in self.bullets if b['id'] not in hit_bullets_to_remove]

        return collided_obstacle_agents, collided_agent_pairs, hit_bullets_to_remove

    def _calculate_rewards(self, agent: PyAgent, other_agent: PyAgent) -> float:
        """指定されたエージェントのこのステップでの報酬を計算する"""
        reward = 0.0
        agent_id = agent.id

        # --- 衝突によるペナルティ/ボーナス (衝突判定結果を反映) ---
        if agent.collided_obstacle_this_step:
            reward += PNL_A0_OBSTACLE_COLLISION if agent_id == 0 else PNL_A1_OBSTACLE_COLLISION
        elif agent.collided_agent_this_step:
            # エージェント間衝突の場合
            if agent_id == 0: reward += RWD_A0_CATCH_TARGET
            else: reward += PNL_A1_CAUGHT
        elif agent.type == 'evader' and agent.hit_by_bullet_this_step:
            reward += PNL_A1_HIT_BY_BULLET
        elif agent.type == 'pursuer' and other_agent.hit_by_bullet_this_step:
            # 自分が撃った弾が相手に命中した場合 (相手のhitフラグを見る)
            reward += RWD_A0_HIT_TARGET

        # --- アクティブ状態での報酬/ペナルティ ---
        # エージェントがこのステップの開始時にアクティブだった場合のみ適用
        # (衝突などで非アクティブになった場合でも、そのフレームでの行動に基づく報酬は与える)
        # is_active は step 開始時の状態を参照すべきだが、ここでは簡易的に現在の is_active で代用
        # → 正確には step 関数の最初で is_active を記録しておくべき
        # ここでは、衝突で is_active=False になった場合は生存報酬などを与えないようにする
        if agent.is_active:
            if agent_id == 0: # Pursuer
                reward += RWD_A0_SURVIVAL
                # 前進報酬: 目標 (Evader) 方向への移動量に応じてボーナス
                # Y軸負方向 (-1) が目標
                current_pos = np.array([agent.x, agent.y])
                last_pos = np.array(agent.last_position)
                movement = current_pos - last_pos
                # 角度による目標方向ベクトル (北向き = [0, -1])
                target_dir = np.array([0.0, -1.0])
                # 移動ベクトルと目標方向の内積を計算 (正規化)
                move_norm = np.linalg.norm(movement)
                if move_norm > 1e-6:
                    forward_progress = np.dot(movement / move_norm, target_dir)
                    reward += max(0, forward_progress) * RWD_A0_FORWARD_MOVE

                # 敵への近接ボーナス
                enemy_dist = np.min(agent.sensors_target)
                proximity_threshold = agent.sensor_range_target * PROXIMITY_THRESHOLD_RATIO
                if enemy_dist < proximity_threshold:
                    # 近いほどボーナス増加 (最大1.0)
                    bonus = (1.0 - min(1.0, enemy_dist / proximity_threshold))
                    reward += bonus * RWD_A0_ENEMY_PROXIMITY

                # 障害物への近接ペナルティ
                min_obs_dist = np.min(agent.sensors_obstacle)
                obs_proximity_threshold = agent.sensor_range_obstacle * PROXIMITY_THRESHOLD_RATIO
                if min_obs_dist < obs_proximity_threshold:
                    penalty = (1.0 - min(1.0, min_obs_dist / obs_proximity_threshold))
                    reward += penalty * PNL_A0_OBSTACLE_PROXIMITY

            else: # Evader (Agent 1)
                reward += RWD_A1_SURVIVAL
                # 前進報酬: 追跡者から離れる方向への移動量
                # 追跡者の方向ベクトル (自分 -> 追跡者)
                pursuer_vec = np.array([other_agent.x - agent.x, other_agent.y - agent.y])
                # トーラス空間考慮 (簡易版: 差が半分以上なら逆方向から)
                if abs(pursuer_vec[0]) > SIM_WIDTH / 2: pursuer_vec[0] -= np.sign(pursuer_vec[0]) * SIM_WIDTH
                if abs(pursuer_vec[1]) > SIM_HEIGHT / 2: pursuer_vec[1] -= np.sign(pursuer_vec[1]) * SIM_HEIGHT
                pursuer_dir = pursuer_vec / (np.linalg.norm(pursuer_vec) + 1e-6)
                # 離れる方向 = -pursuer_dir
                escape_dir = -pursuer_dir

                current_pos = np.array([agent.x, agent.y])
                last_pos = np.array(agent.last_position)
                movement = current_pos - last_pos
                move_norm = np.linalg.norm(movement)
                if move_norm > 1e-6:
                    forward_progress = np.dot(movement / move_norm, escape_dir)
                    reward += max(0, forward_progress) * RWD_A1_FORWARD_MOVE

                # 追跡者からの距離ボーナス
                pursuer_dist = np.min(agent.sensors_target)
                # 遠いほどボーナス増加 (最大1.0)
                bonus = min(1.0, pursuer_dist / agent.sensor_range_target)
                reward += bonus * RWD_A1_PURSUER_DISTANCE

                # 障害物への近接ペナルティ (Pursuerと同様)
                min_obs_dist = np.min(agent.sensors_obstacle)
                obs_proximity_threshold = agent.sensor_range_obstacle * PROXIMITY_THRESHOLD_RATIO
                if min_obs_dist < obs_proximity_threshold:
                    penalty = (1.0 - min(1.0, min_obs_dist / obs_proximity_threshold))
                    reward += penalty * PNL_A1_OBSTACLE_PROXIMITY

            # 旋回ペナルティ (旋回した場合)
            # turn() メソッドで旋回したかどうかのフラグを使う必要がある -> stepメソッドで管理
            # ここでは簡易的に last_action を見て判断
            if agent.last_action_taken is not None:
                if agent_id == 0: # Pursuer
                    # 旋回ペナルティ
                    if agent.last_action_taken in [0, 2, 3, 5]: # L, R, L+F, R+F
                        reward += PNL_A0_TURN
                    # 発射アクションペナルティ (命中・非命中問わず)
                    if agent.last_action_taken in [3, 4, 5]: # L+F, S+F, R+F
                        reward += PNL_A0_FIRE_ACTION
                elif agent_id == 1: # Evader
                    # 旋回ペナルティ
                    if agent.last_action_taken in [0, 2]: # L, R
                        reward += PNL_A1_TURN

        return reward


    def step(self, actions: Dict[int, int], pursuer_q_values: Optional[np.ndarray] = None) -> Tuple[Dict[int, np.ndarray], Dict[int, float], bool, Dict]:
        """
        環境を1ステップ進める。
        Args:
            actions (Dict[int, int]): 各エージェントIDとその行動の辞書。
            pursuer_q_values (Optional[np.ndarray]): 追跡者の現在の状態に対するQ値配列 (発射判断用)。
        Returns:
            Tuple[Dict[int, np.ndarray], Dict[int, float], bool, Dict]:
            (next_states, rewards, done, info)
            - next_states: 各エージェントの次の状態。
            - rewards: 各エージェントがこのステップで得た報酬。
            - done: エピソードが終了したかどうか。
            - info: エピソード終了理由などの追加情報。
        """
        self.frame_count += 1
        rewards = {PURSUER_ID: 0.0, EVADER_ID: 0.0}
        done = False
        info = {'episode_end_reason': 'Unknown'}

        # --- 0. ステップ開始前の状態を記録 ---
        for agent in self.agents.values():
            agent.update_last_position() # 移動量計算のために前フレームの位置を保存

        # --- 1. 行動適用 (旋回、発射試行、クールダウン) ---
        bullets_fired_this_step: List[Dict[str, Any]] = []
        for agent_id, action in actions.items():
            agent = self.agents.get(agent_id)
            if agent and agent.is_active:
                agent.last_action_taken = action # 行動を記録
                agent.turn(action) # 旋回処理

                # Pursuerの発射試行と弾生成
                if agent.type == 'pursuer' and agent.attempt_fire(action):
                    bullet_id = self._get_next_id('bullet')
                    # 弾の初期位置をエージェントの少し前に設定
                    spawn_dist = agent.size * 0.8
                    bullet_x = agent.x + spawn_dist * math.cos(agent.angle)
                    bullet_y = agent.y + spawn_dist * math.sin(agent.angle)
                    new_bullet = {
                        'id': bullet_id, 'type': 'bullet', 'owner_id': agent.id,
                        'x': bullet_x, 'y': bullet_y, 'angle': agent.angle,
                        'radius': BULLET_RADIUS, 'lifetime': BULLET_LIFETIME
                    }
                    self.bullets.append(new_bullet)
                    bullets_fired_this_step.append(new_bullet)
                    logging.debug(f"[Frame {self.frame_count}] Agent {agent_id} fired bullet {bullet_id}.")

        # Pursuerのクールダウン進行
        pursuer = self.agents.get(PURSUER_ID)
        if pursuer: pursuer.decrement_cooldown()

        # --- 2. 物理演算 (移動) ---
        self._move_bullets()
        for agent in self.agents.values():
            agent.move()

        # --- 3. 障害物生成 ---
        self._generate_obstacle_runtime()

        # --- 4. クアッドツリー更新 ---
        # 移動後のオブジェクト位置でQuadtreeを再構築
        self._update_quadtree()

        # --- 5. 衝突判定 ---
        # この関数内でエージェントの衝突フラグ、状態 (is_active)、体力 (Evader) が更新される
        collided_obs_agents, collided_agent_pairs, removed_bullets = self._check_collisions()

        # --- 6. センサー更新 ---
        # 衝突判定 *後* の状態でセンサーを更新する
        for agent in self.agents.values():
            other_agent = self.agents[1 - agent.id]
            agent.update_sensors(self.world_quadtree, other_agent)

        # --- 7. 報酬計算 ---
        # 衝突結果とセンサー値に基づいて報酬を計算
        for agent_id, agent in self.agents.items():
            other_agent = self.agents[1 - agent_id]
            rewards[agent_id] = self._calculate_rewards(agent, other_agent)

        # --- 8. エピソード終了判定 ---
        agent0 = self.agents[PURSUER_ID]
        agent1 = self.agents[EVADER_ID]

        if self.frame_count >= MAX_FRAMES_PER_EPISODE:
            done = True
            info['episode_end_reason'] = 'max_frames_reached'
        # is_active フラグで判定 (衝突処理でFalseになる)
        elif not agent0.is_active and agent0.collided_obstacle_this_step:
             done = True
             info['episode_end_reason'] = 'pursuer_obstacle_collision'
        elif not agent1.is_active and agent1.collided_obstacle_this_step:
             done = True
             info['episode_end_reason'] = 'evader_obstacle_collision'
        elif not agent1.is_active and (agent1.health is not None and agent1.health <= 0):
             done = True
             info['episode_end_reason'] = 'evader_health_depleted'
        elif not agent0.is_active and not agent1.is_active and (PURSUER_ID, EVADER_ID) in collided_agent_pairs:
             done = True
             info['episode_end_reason'] = 'agent_collision'
        # 片方だけ非アクティブになるケースは？ (現状のロジックでは障害物衝突か体力切れ)
        # elif not agent0.is_active or not agent1.is_active:
        #     # どちらか一方が非アクティブになったら終了（上記で個別に判定済みのはず）
        #     # done = True
        #     # info['episode_end_reason'] = 'one_agent_inactive'
        #     pass

        # --- 9. 次の状態を取得 ---
        next_states = {agent_id: agent.get_state() for agent_id, agent in self.agents.items()}

        # --- 10. 返り値 ---
        # logging.debug(f"Step {self.frame_count}: A0_R={rewards[0]:.3f}, A1_R={rewards[1]:.3f}, Done={done}")
        return next_states, rewards, done, info

    def get_render_data(self) -> Dict[str, Any]:
        """フロントエンド描画用の環境状態を返す"""
        return {
            'frame': self.frame_count,
            'agents': [agent.get_render_data() for agent in self.agents.values()],
            'obstacles': self.obstacles,
            'bullets': self.bullets,
            # 必要なら他の情報も追加 (例: ゲームスコアなど)
        }

# ==============================
# DQNAgent クラス
# ==============================
class DQNAgent:
    """DQNアルゴリズムを実装した強化学習エージェント"""
    def __init__(self, state_size: int, action_size: int, agent_id: int):
        self.agent_id = agent_id
        self.state_size = state_size
        self.action_size = action_size

        # リプレイメモリ (deque)
        self.memory: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=MEMORY_SIZE)

        # DQN ハイパーパラメータ
        self.gamma = GAMMA            # 割引率
        self.epsilon = EPSILON_START  # ε-greedy法のε初期値
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY_RATE
        self.learning_rate = LEARNING_RATE

        # モデル構築
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model() # 初期同期

        self._train_step_counter = 0 # train_on_batch の呼び出し回数 (デバッグ用)

        logging.info(f"DQNAgent {self.agent_id} initialized (State: {state_size}, Action: {action_size}).")

    def _build_model(self) -> tf.keras.Model:
        """Qネットワークモデルを構築する"""
        input_layer = tf.keras.layers.Input(shape=(self.state_size,), name=f"agent{self.agent_id}_input")
        x = input_layer
        for i, units in enumerate(DQN_HIDDEN_UNITS):
            x = tf.keras.layers.Dense(units, activation='relu', name=f"agent{self.agent_id}_dense{i+1}")(x)
        output_layer = tf.keras.layers.Dense(self.action_size, activation='linear', name=f"agent{self.agent_id}_output")(x)

        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(loss='mse', optimizer=optimizer) # Mean Squared Error損失
        # モデル概要をログに出力 (DEBUGレベル)
        # stringlist = []
        # model.summary(print_fn=lambda x: stringlist.append(x))
        # model_summary = "\n".join(stringlist)
        # logging.debug(f"Agent {self.agent_id} Model Summary:\n{model_summary}")
        return model

    def update_target_model(self):
        """ターゲットネットワークの重みを現在のモデルの重みで更新する"""
        self.target_model.set_weights(self.model.get_weights())
        logging.debug(f"Agent {self.agent_id} target model updated.")

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """経験をリプレイメモリに保存する"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: np.ndarray, training: bool = True) -> int:
        """ε-greedy方策に基づいて行動を選択する"""
        if training and np.random.rand() <= self.epsilon:
            # εの確率でランダムに行動を選択
            return random.randrange(self.action_size)
        else:
            # Q値が最大となる行動を選択
            # stateを (1, state_size) の形状に変形して入力
            state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
            act_values = self.model(state_tensor, training=False) # 推論モード
            return int(np.argmax(act_values[0].numpy())) # Q値最大の行動のインデックス

    def replay(self, batch_size: int) -> float:
        """リプレイメモリからミニバッチを取得し、モデルを学習する"""
        if len(self.memory) < REPLAY_START_SIZE or len(self.memory) < batch_size:
            return 0.0 #十分な経験がない場合は学習しない

        # メモリからランダムにミニバッチをサンプリング
        minibatch = random.sample(self.memory, batch_size)

        # ミニバッチから各要素を抽出してNumPy配列に変換
        states = np.array([transition[0] for transition in minibatch], dtype=np.float32)
        actions = np.array([transition[1] for transition in minibatch]) # 行動インデックス
        rewards = np.array([transition[2] for transition in minibatch], dtype=np.float32)
        next_states = np.array([transition[3] for transition in minibatch], dtype=np.float32)
        dones = np.array([transition[4] for transition in minibatch]) # bool配列

        # 1. ターゲットQ値の計算
        # 次の状態に対するターゲットモデルのQ値予測 (shape: batch_size, action_size)
        target_q_next = self.target_model.predict_on_batch(next_states)
        # 次の状態で取りうる行動の中で最大のQ値 (shape: batch_size,)
        max_q_next = np.max(target_q_next, axis=1)
        # ターゲットQ値: R + γ * max_a Q_target(s', a)  (エピソード終了時は R のみ)
        target_q_values = rewards + self.gamma * max_q_next * (1 - dones.astype(int))

        # 2. 現在のモデルのQ値予測とターゲットの更新
        # 現在の状態に対する現在のモデルのQ値予測 (shape: batch_size, action_size)
        current_q_values = self.model.predict_on_batch(states)

        # 実際に取った行動に対応するQ値だけをターゲットQ値で更新
        batch_indices = np.arange(batch_size, dtype=np.int32)
        current_q_values[batch_indices, actions] = target_q_values

        # 3. モデルの学習 (状態sを入力、更新されたQ値を教師データとして学習)
        loss = self.model.train_on_batch(states, current_q_values)
        self._train_step_counter += 1
        if self._train_step_counter % 100 == 0: # 定期的にデバッグログ出力
             logging.debug(f"Agent {self.agent_id} training step {self._train_step_counter}, Loss: {loss:.4f}")

        return float(loss)

    def decay_epsilon(self):
        """ε値を減衰させる"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            # self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def load(self, filepath: str) -> bool:
        """モデルの重みをファイルから読み込む"""
        model_file = Path(filepath)
        if not model_file.is_file():
            logging.warning(f"Agent {self.agent_id} model weights file not found at {filepath}. Starting with random weights.")
            return False
        try:
            self.model.load_weights(filepath)
            self.update_target_model() # ターゲットモデルも同期
            # 読み込み成功時はεを実行用に設定（オプション）
            self.epsilon = self.epsilon_min
            logging.info(f"Agent {self.agent_id} model weights loaded successfully from {filepath}. Epsilon set to {self.epsilon:.3f}.")
            return True
        except Exception as e:
            logging.error(f"Failed to load model weights for agent {self.agent_id} from {filepath}: {e}", exc_info=True)
            # ロード失敗した場合も学習は継続できるよう、Falseを返す
            return False

    def save(self, filepath: str):
        """モデルの重みをファイルに保存する"""
        try:
            # 保存先ディレクトリが存在しない場合は作成
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            self.model.save_weights(filepath)
            logging.info(f"Agent {self.agent_id} model weights saved successfully to {filepath}")
        except Exception as e:
            logging.error(f"Failed to save model weights for agent {self.agent_id} to {filepath}: {e}", exc_info=True)

# ==============================
# WebSocket サーバー & ループ管理
# ==============================

# --- グローバル状態変数 ---
# Thread-safeではないが、asyncioのシングルスレッドイベントループ内では基本的に問題ない
connected_client: Optional[websockets.WebSocketServerProtocol] = None
server_mode: Literal['idle', 'training', 'running', 'error'] = 'idle'
simulation_task: Optional[asyncio.Task] = None
# DQNエージェントインスタンス (モード開始時に初期化)
dqn_agent0: Optional[DQNAgent] = None
dqn_agent1: Optional[DQNAgent] = None
# 環境インスタンス (モード開始時に初期化)
simulation_env: Optional[SimulationEnv] = None

# --- WebSocket メッセージ送信 ---
async def send_message(websocket: Optional[websockets.WebSocketServerProtocol], data: Dict[str, Any]):
    """指定されたWebSocketクライアントにJSONメッセージを送信する"""
    # 事前の接続状態チェック (open/closed属性へのアクセスを削除)
    if websocket: # websocket オブジェクトが None でないことだけを確認
        try:
            json_str = json.dumps(data)
            await websocket.send(json_str) # 送信を試みる
            log_level = logging.DEBUG if data.get('type') == 'update_state' else logging.INFO
            logging.log(log_level, f"Sent message: type={data.get('type')}")
        except websockets.exceptions.ConnectionClosed:
            # send() 実行時に接続が閉じていた場合にこの例外が発生する
            logging.warning(f"Failed to send message (type={data.get('type')}): Connection was closed.")
            # グローバルなクライアント参照をクリア
            global connected_client
            if connected_client == websocket:
                connected_client = None
                logging.info("Cleared connected client reference due to ConnectionClosed during send.")
        except TypeError as e:
            logging.error(f"JSON Serialize Error for type={data.get('type')}: {e}. Data: {data}")
        except Exception as e:
            # その他の送信時エラー (ネットワーク問題など)
            logging.error(f"Failed to send message (type={data.get('type')}): {e}", exc_info=True)
    else:
        logging.debug(f"Cannot send message (type={data.get('type')}): WebSocket is None.")

# --- モード変更関数 ---
async def change_mode(new_mode: Literal['idle', 'training', 'running', 'error'], message: str = ""):
    """サーバーのモードを変更し、クライアントに通知する"""
    global server_mode
    if server_mode == new_mode:
        return # モード変更なし

    logging.info(f"Changing mode from '{server_mode}' to '{new_mode}'. Reason: {message}")
    server_mode = new_mode
    await send_message(connected_client, {
        'type': 'mode_changed',
        'mode': server_mode,
        'message': message or f"{new_mode.capitalize()} mode started."
    })

# --- 学習ループ ---
async def training_loop(websocket: websockets.WebSocketServerProtocol):
    """強化学習のメインループ"""
    global dqn_agent0, dqn_agent1, simulation_env, simulation_task

    logging.info(f"Starting Training Loop for {TOTAL_EPISODES} episodes...")
    if not (dqn_agent0 and dqn_agent1 and simulation_env):
        logging.error("Training loop started but agents or environment not initialized.")
        await change_mode('error', "Initialization failed.")
        return

    agents = {PURSUER_ID: dqn_agent0, EVADER_ID: dqn_agent1}
    model_paths = {PURSUER_ID: MODEL_PATH_A0, EVADER_ID: MODEL_PATH_A1}

    total_steps = 0
    episode_rewards_history = {PURSUER_ID: deque(maxlen=100), EVADER_ID: deque(maxlen=100)} # 直近100エピソードの報酬記録用
    episode_losses_history = {PURSUER_ID: deque(maxlen=100), EVADER_ID: deque(maxlen=100)} # 直近100エピソードの平均損失記録用

    start_time_total = time.time()

    try:
        for episode in range(1, TOTAL_EPISODES + 1):
            if server_mode != 'training':
                logging.info(f"Training stopped externally at episode {episode}.")
                break

            # --- エピソード開始 ---
            states = simulation_env.reset() # 環境リセット、初期状態取得
            # JS側に初期状態を送信
            initial_render_data = simulation_env.get_render_data()
            await send_message(websocket, {'type': 'init_state', **initial_render_data})

            episode_rewards = {PURSUER_ID: 0.0, EVADER_ID: 0.0}
            episode_losses = {PURSUER_ID: [], EVADER_ID: []}
            done = False
            frame = 0
            episode_start_time = time.time()

            while not done and server_mode == 'training':
                frame += 1
                total_steps += 1

                # 1. 行動選択 (ε-greedy)
                actions = {agent_id: agent.act(states[agent_id], training=True)
                           for agent_id, agent in agents.items()}

                # 2. 環境を1ステップ進める
                next_states, rewards, done, info = simulation_env.step(actions)

                # 3. 報酬と経験を記録
                for agent_id in agents.keys():
                    episode_rewards[agent_id] += rewards[agent_id]
                    # 経験をリプレイバッファに記憶
                    agents[agent_id].remember(states[agent_id], actions[agent_id], rewards[agent_id], next_states[agent_id], done)

                # 4. 状態を更新
                states = next_states

                # 5. モデル学習 (リプレイ) - 一定頻度で実行
                if total_steps % REPLAY_FREQUENCY == 0:
                    current_batch_losses = {PURSUER_ID: 0.0, EVADER_ID: 0.0}
                    for agent_id, agent in agents.items():
                        loss_value = agent.replay(BATCH_SIZE)
                        if loss_value > 1e-7: # 小さすぎる損失は記録しない
                            episode_losses[agent_id].append(loss_value)
                            current_batch_losses[agent_id] = loss_value # training_status送信用

                # 6. フロントエンドに状態と進捗を送信 (フレームごとに送信)
                #    (負荷が高ければ送信頻度を調整する)
                render_data = simulation_env.get_render_data()
                await send_message(websocket, {'type': 'update_state', **render_data})

                # 定期的に学習ステータスも送信 (例: 10フレームごと)
                if frame % 10 == 0:
                    avg_ep_losses = {aid: (sum(ls)/len(ls)) if ls else 0.0 for aid, ls in episode_losses.items()}
                    status_data = {
                        'episode': episode,
                        'frame': frame,
                        'rewards_current_ep': {aid: round(r, 3) for aid, r in episode_rewards.items()},
                        'epsilons': {aid: round(ag.epsilon, 3) for aid, ag in agents.items()},
                        'losses_last_batch': {aid: round(l, 5) if l > 0 else None for aid, l in current_batch_losses.items()},
                        'progress': round(min(1.0, episode / TOTAL_EPISODES), 3),
                    }
                    await send_message(websocket, {'type': 'training_status', 'status': status_data})

                # 非同期処理の実行機会を与える
                await asyncio.sleep(0) # ほぼ待たないが、他のコルーチンに制御を渡す

            # --- エピソード終了 ---
            episode_duration = time.time() - episode_start_time
            for aid in agents.keys():
                 episode_rewards_history[aid].append(episode_rewards[aid])
                 avg_loss = (sum(episode_losses[aid]) / len(episode_losses[aid])) if episode_losses[aid] else 0.0
                 episode_losses_history[aid].append(avg_loss)
                 agents[aid].decay_epsilon() # ε減衰

            # エピソード終了情報をJSに送信
            await send_message(websocket, {'type': 'episode_end', 'reason': info.get('episode_end_reason', 'N/A')})
            # 少し待機してJS側の処理時間を確保 (任意)
            # await asyncio.sleep(0.02)

            # ログ出力 (平均報酬なども表示)
            avg_rewards_str = ", ".join([f"A{aid}: {sum(hist)/len(hist):.2f}" for aid, hist in episode_rewards_history.items() if hist])
            avg_losses_str = ", ".join([f"A{aid}: {sum(hist)/len(hist):.4f}" for aid, hist in episode_losses_history.items() if hist])
            logging.info(
                f"Ep {episode}/{TOTAL_EPISODES} | Frames: {frame} | Dur: {episode_duration:.2f}s | "
                f"Rews: (A0:{episode_rewards[0]:.2f}, A1:{episode_rewards[1]:.2f}) | "
                f"Avg Rews (last 100): [{avg_rewards_str}] | Eps: (A0:{agents[0].epsilon:.3f}, A1:{agents[1].epsilon:.3f}) | "
                f"Avg Loss (last 100): [{avg_losses_str}] | End: {info.get('episode_end_reason', 'N/A')}"
            )

            # ターゲットネットワーク更新
            if episode % TARGET_UPDATE_FREQ == 0:
                for agent in agents.values(): agent.update_target_model()

            # モデルの中間保存
            if episode % MODEL_SAVE_FREQ == 0 and episode > 0:
                for agent_id, agent in agents.items(): agent.save(model_paths[agent_id])

        # --- 全エピソード終了後 ---
        if episode >= TOTAL_EPISODES:
            logging.info(f"Training finished naturally after {TOTAL_EPISODES} episodes.")
            await change_mode('idle', '学習完了')
        else:
            # ループが break で抜けた場合 (外部から停止されたなど)
             await change_mode('idle', '学習停止')

    except asyncio.CancelledError:
        logging.info("Training loop was cancelled.")
        await change_mode('idle', '学習キャンセル')
    except Exception as e:
        logging.error(f"Error in training loop: {e}", exc_info=True)
        await change_mode('error', f'学習エラー: {type(e).__name__}')
    finally:
        logging.info("Training loop finished or terminated. Saving final models...")
        # 最終モデルを保存
        if dqn_agent0: dqn_agent0.save(MODEL_PATH_A0)
        if dqn_agent1: dqn_agent1.save(MODEL_PATH_A1)
        # グローバル参照クリア
        simulation_task = None
        # simulation_env = None # 環境は維持しても良いかも？
        # dqn_agent0 = None
        # dqn_agent1 = None

# --- 実行ループ ---
async def running_loop(websocket: websockets.WebSocketServerProtocol):
    """学習済みモデルを使った実行ループ"""
    global dqn_agent0, dqn_agent1, simulation_env, simulation_task

    logging.info("Starting Running Loop...")
    if not (dqn_agent0 and dqn_agent1 and simulation_env):
        logging.error("Running loop started but agents or environment not initialized.")
        await change_mode('error', "Initialization failed.")
        return

    agents = {PURSUER_ID: dqn_agent0, EVADER_ID: dqn_agent1}
    # 実行モードではεを最小値に固定
    for agent in agents.values(): agent.epsilon = agent.epsilon_min

    try:
        # 環境リセットと初期状態送信
        states = simulation_env.reset()
        initial_render_data = simulation_env.get_render_data()
        await send_message(websocket, {'type': 'init_state', **initial_render_data})

        done = False
        while not done and server_mode == 'running':
            start_frame_time = time.monotonic() # フレーム時間計測開始

            # 1. 行動選択 (greedy)
            actions = {agent_id: agent.act(states[agent_id], training=False)
                       for agent_id, agent in agents.items()}

            # 2. 環境を1ステップ進める (報酬や終了判定も取得)
            next_states, rewards, done, info = simulation_env.step(actions)

            # 3. 状態を更新
            states = next_states

            # 4. フロントエンドに状態送信
            render_data = simulation_env.get_render_data()
            await send_message(websocket, {'type': 'update_state', **render_data})

            # エピソード終了判定 (done フラグをチェック)
            if done:
                logging.info(f"Running episode finished. Reason: {info.get('episode_end_reason', 'N/A')}")
                # エピソード終了メッセージを送信 (オプション)
                await send_message(websocket, {'type': 'episode_end', 'reason': info.get('episode_end_reason', 'N/A')})
                # 少し待ってリセット、または停止
                await asyncio.sleep(1.0) # 1秒待って次のエピソードへ
                if server_mode == 'running': # モードが変わっていなければリセット
                    states = simulation_env.reset()
                    initial_render_data = simulation_env.get_render_data()
                    await send_message(websocket, {'type': 'init_state', **initial_render_data})
                    done = False # ループ継続
                else:
                     break # モードが変わっていたらループ終了

            # 5. フレームレート制御
            elapsed_time = time.monotonic() - start_frame_time
            sleep_duration = max(0, RUNNING_MODE_SLEEP_TIME - elapsed_time)
            if sleep_duration == 0 and SERVER_TICK_RATE > 0:
                 logging.warning(f"Running loop took longer ({elapsed_time:.4f}s) than target frame time ({RUNNING_MODE_SLEEP_TIME:.4f}s).")
            await asyncio.sleep(sleep_duration)

        # --- ループ終了後 ---
        if server_mode == 'running': # 自然終了ではなく外部要因で停止した場合
             await change_mode('idle', '実行停止')

    except asyncio.CancelledError:
        logging.info("Running loop was cancelled.")
        await change_mode('idle', '実行キャンセル')
    except Exception as e:
        logging.error(f"Error in running loop: {e}", exc_info=True)
        await change_mode('error', f'実行エラー: {type(e).__name__}')
    finally:
        logging.info("Running loop finished or terminated.")
        simulation_task = None
        # simulation_env = None # 維持
        # dqn_agent0 = None
        # dqn_agent1 = None

# --- WebSocket メッセージハンドラ ---
async def handle_message(websocket: websockets.WebSocketServerProtocol, message_str: str):
    """受信したWebSocketメッセージを処理する"""
    global server_mode, simulation_task, dqn_agent0, dqn_agent1, simulation_env

    try:
        message: Dict[str, Any] = json.loads(message_str)
        msg_type: Optional[str] = message.get('type')
        logging.debug(f"Received message: type={msg_type}, data={message}")

        if msg_type == 'client_hello':
            logging.info(f"Client hello: {message.get('message')}")
            await send_message(websocket, {'type': 'server_hello', 'message': 'Python Server (Python-Driven Arch) Connected'})
            # 接続時に現在のモードを通知
            await send_message(websocket, {'type': 'mode_changed', 'mode': server_mode, 'message': f'Connected to server in {server_mode} mode.'})

        elif msg_type == 'command':
            command: Optional[str] = message.get('command')
            logging.info(f"Command received: {command}")

            # --- コマンド処理 ---
            if command == 'start_training':
                if server_mode == 'idle' and (simulation_task is None or simulation_task.done()):
                    # 1. エージェントと環境を初期化
                    logging.info("Initializing agents and environment for training...")
                    dqn_agent0 = DQNAgent(PURSUER_STATE_SIZE, PURSUER_ACTION_SIZE, PURSUER_ID)
                    dqn_agent1 = DQNAgent(EVADER_STATE_SIZE, EVADER_ACTION_SIZE, EVADER_ID)
                    simulation_env = SimulationEnv()
                    # 2. 保存済みモデルがあれば読み込む
                    dqn_agent0.load(MODEL_PATH_A0)
                    dqn_agent1.load(MODEL_PATH_A1)
                    # 3. モード変更とループ開始
                    await change_mode('training', '学習開始')
                    simulation_task = asyncio.create_task(training_loop(websocket))
                    simulation_task.add_done_callback(lambda t: logging.info(f"Training task finished. Result={t.result()}, Exception={t.exception()}"))
                else:
                    err_msg = "Cannot start training. Server not idle or task already running."
                    logging.warning(err_msg)
                    await send_message(websocket, {'type': 'error', 'message': err_msg})

            elif command == 'stop_training':
                if server_mode == 'training' and simulation_task and not simulation_task.done():
                    logging.info("Requesting to stop training task...")
                    simulation_task.cancel()
                    # キャンセル完了を待たずに idle に戻す (finallyブロックで処理される)
                    # await change_mode('idle', '学習停止リクエスト済み')
                else:
                    err_msg = "Cannot stop training. Not in training mode or no task found."
                    logging.warning(err_msg)
                    await send_message(websocket, {'type': 'error', 'message': err_msg})

            elif command == 'start_running':
                if server_mode == 'idle' and (simulation_task is None or simulation_task.done()):
                    # 1. エージェントと環境を初期化
                    logging.info("Initializing agents and environment for running...")
                    dqn_agent0 = DQNAgent(PURSUER_STATE_SIZE, PURSUER_ACTION_SIZE, PURSUER_ID)
                    dqn_agent1 = DQNAgent(EVADER_STATE_SIZE, EVADER_ACTION_SIZE, EVADER_ID)
                    simulation_env = SimulationEnv()
                    # 2. 学習済みモデルをロード (必須)
                    load0_ok = dqn_agent0.load(MODEL_PATH_A0)
                    load1_ok = dqn_agent1.load(MODEL_PATH_A1)
                    if not (load0_ok and load1_ok):
                        missing = []
                        if not load0_ok: missing.append(Path(MODEL_PATH_A0).name)
                        if not load1_ok: missing.append(Path(MODEL_PATH_A1).name)
                        err_msg = f"Cannot start running. Failed to load required models: {', '.join(missing)}"
                        logging.error(err_msg)
                        await send_message(websocket, {'type': 'error', 'message': err_msg})
                        # エラー時はインスタンスをクリア
                        dqn_agent0 = None; dqn_agent1 = None; simulation_env = None
                        return
                    # 3. モード変更とループ開始
                    await change_mode('running', '実行開始')
                    simulation_task = asyncio.create_task(running_loop(websocket))
                    simulation_task.add_done_callback(lambda t: logging.info(f"Running task finished. Result={t.result()}, Exception={t.exception()}"))
                else:
                    err_msg = "Cannot start running. Server not idle or task already running."
                    logging.warning(err_msg)
                    await send_message(websocket, {'type': 'error', 'message': err_msg})

            elif command == 'stop_running':
                if server_mode == 'running' and simulation_task and not simulation_task.done():
                    logging.info("Requesting to stop running task...")
                    simulation_task.cancel()
                else:
                    err_msg = "Cannot stop running. Not in running mode or no task found."
                    logging.warning(err_msg)
                    await send_message(websocket, {'type': 'error', 'message': err_msg})

            else:
                logging.warning(f"Received unknown command: {command}")
                await send_message(websocket, {'type': 'error', 'message': f"Unknown command: {command}"})

        # --- 他のメッセージタイプ (必要なら追加) ---
        # 例: JSから次のフレーム要求を受け取る場合
        # elif msg_type == 'request_next_frame':
        #     if server_mode == 'running':
        #         # running_loop側でこのイベントを待つか、フラグを立てる
        #         pass
        #     else:
        #         logging.warning("Received request_next_frame but not in running mode.")

        else:
            logging.warning(f"Unknown message type received: {msg_type}")

    except json.JSONDecodeError:
        logging.error(f"Failed to decode JSON from message: {message_str}")
        await send_message(websocket, {'type': 'error', 'message': 'Invalid JSON format.'})
    except Exception as e:
        logging.error(f"Error handling message: {e}", exc_info=True)
        try:
            await send_message(websocket, {'type': 'error', 'message': f'Server internal error: {type(e).__name__}'})
        except Exception as send_err:
            logging.error(f"Failed to send error message back to client: {send_err}")


# --- WebSocket 接続ハンドラ ---
async def connection_handler(websocket: websockets.WebSocketServerProtocol):
    """新しいWebSocket接続要求を処理するハンドラ"""
    global connected_client, server_mode, simulation_task, dqn_agent0, dqn_agent1, simulation_env
    remote_addr = websocket.remote_address
    logging.info(f"Connection attempt from: {remote_addr}")

    # --- シングルクライアント接続制限 ---
    if connected_client is not None:
        logging.warning(f"Rejecting connection from {remote_addr}: Server busy with {connected_client.remote_address}.")
        await websocket.close(code=1008, reason="Server busy")
        return

    connected_client = websocket
    logging.info(f"Client connected: {remote_addr}")

    try:
        # 接続確立後、クライアントからのメッセージを待ち受け、処理するループ
        async for message in websocket:
            await handle_message(websocket, str(message))

    except websockets.exceptions.ConnectionClosedError as e:
        logging.info(f"Connection closed uncleanly with {remote_addr}: code={e.code}, reason='{e.reason}'")
    except websockets.exceptions.ConnectionClosedOK:
        logging.info(f"Connection closed cleanly by {remote_addr}.")
    except Exception as e:
        # 予期せぬエラー (ネットワーク関連など)
        logging.error(f"Unexpected error on connection with {remote_addr}: {e}", exc_info=True)
    finally:
        logging.info(f"Cleaning up connection for {remote_addr}")
        # グローバルなクライアント参照をクリア
        if connected_client == websocket:
            connected_client = None
            logging.info("Cleared connected client reference.")

            # クライアント切断時に実行中のタスクがあればキャンセル
            if simulation_task and not simulation_task.done():
                logging.info("Client disconnected, cancelling the ongoing simulation task...")
                simulation_task.cancel()
                # キャンセル完了を待つか？ (待たない方が早くリソース解放できるかも)
                # try:
                #     await asyncio.wait_for(simulation_task, timeout=1.0)
                # except asyncio.TimeoutError:
                #     logging.warning("Timeout waiting for task cancellation.")
                # except asyncio.CancelledError:
                #     pass # 正常なキャンセル
                simulation_task = None

            # クライアント切断時にモードをアイドルに戻す
            if server_mode != 'idle':
                # change_mode は接続が必要なので直接変更
                server_mode = 'idle'
                logging.info("Server mode set to idle due to client disconnection.")

            # エージェントや環境インスタンスをクリアするかどうかはポリシーによる
            # 学習途中の状態を保持したい場合はクリアしない
            # dqn_agent0 = None
            # dqn_agent1 = None
            # simulation_env = None
        else:
             # 理論上ここには来ないはずだが、念のためログ
             logging.warning(f"Cleanup called for {remote_addr}, but global client was already different or None.")

# ==============================
# サーバー起動
# ==============================
async def main():
    """サーバーアプリケーションのメインエントリーポイント"""
    # 保存ディレクトリ作成
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

    # WebSocketサーバー起動
    # ping_interval/timeout で無応答接続を切断
    server = await websockets.serve(
        connection_handler,
        HOST, PORT,
        ping_interval=20, ping_timeout=20,
        # メッセージサイズ上限を増やす (必要な場合)
        # max_size=2**20 # 1MB
    )
    server_address = f"ws://{HOST}:{PORT}"
    logging.info(f"WebSocket server listening on {server_address}")
    logging.info("Server ready and waiting for a client connection...")
    # サーバーが停止するまで待機
    await server.wait_closed()
    logging.info("WebSocket server has been stopped.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Server shutdown requested (KeyboardInterrupt).")
    except Exception as e:
        # サーバー起動時の致命的なエラー
        logging.critical(f"Server failed to start or run: {e}", exc_info=True)